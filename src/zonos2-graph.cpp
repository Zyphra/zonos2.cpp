// zonos2-graph.cpp — ggml prefill graph + numeric validation runner.
// Built up phase by phase: P1 embed+emb_norm, P2 attention+dense FFN, P3 MoE, P4 head.
#include "zonos2.h"
#include "zonos2-sampler.h"
#include "npy.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

struct gctx {
    const zonos2_model * m = nullptr;
    ggml_context * ctx = nullptr;
    int n = 0;
    std::vector<ggml_tensor *> ids;
    ggml_tensor * pos = nullptr;
    ggml_tensor * router_states = nullptr; // EDA state threaded across MoE layers

    // speaker conditioning: overwrite the embedding at column spk_pos with the
    // projected speaker vector (before emb_norm). spk_pos < 0 => no speaker.
    ggml_tensor * spk = nullptr; // input speaker embedding [spk_dim]
    int spk_pos = -1;
    bool capture = true;                   // false during generation (lean graph)
    std::vector<std::pair<std::string, ggml_tensor *>> caps;

    // KV cache
    std::vector<ggml_tensor *> * kc = nullptr;  // per-layer K cache [head_dim, max_seq, n_head_kv]
    std::vector<ggml_tensor *> * vc = nullptr;  // per-layer V cache (same layout)
    bool decode = false;                        // true: attend over the full cache window via mask
    int  max_kv = 0;                            // cache window size
    ggml_tensor * mask = nullptr;               // [max_kv, 1] additive mask (decode only)
    ggml_cgraph * gf = nullptr;                 // graph, for inline set_rows expansion

    // mark a tensor as a captured (read-back) graph output
    ggml_tensor * cap(const std::string & name, ggml_tensor * t) {
        if (!capture) return t;
        if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
        ggml_set_output(t);
        ggml_set_name(t, name.c_str());
        caps.push_back({name, t});
        return t;
    }
};

// RMSNorm with learnable weight (w may be null for the weightless emb_norm).
static ggml_tensor * rms_w(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    ggml_tensor * t = ggml_rms_norm(ctx, x, eps);
    return w ? ggml_mul(ctx, t, w) : t;
}

// Attention block: gater sigmoid, QK-RMSNorm + per-head temp, interleaved RoPE,
// GQA causal attention (scale 1/sqrt(head_dim)), per-head sigmoid gate, wo.
// When g.kv: write the new tokens' K/V into the per-layer cache at [pos0, pos0+n) and
// attend against the full cache [0, pos0+n). Otherwise self-attend over the n tokens.
static ggml_tensor * build_attention(gctx & g, ggml_tensor * cur, const zonos2_layer & ly, int il) {
    const zonos2_hparams & hp = g.m->hp;
    ggml_context * ctx = g.ctx;
    const int hd = (int) hp.head_dim, nh = (int) hp.n_head, nkv = (int) hp.n_head_kv, n = g.n;

    // per-head output gate from the (attn-norm) input
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ly.gater, cur)); // [nh, n]

    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, ly.wq, cur), hd, nh,  n);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, ly.wk, cur), hd, nkv, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, ly.wv, cur), hd, nkv, n);

    // QK-norm over head_dim (eps 1e-6); q additionally scaled by |temp| per head
    q = ggml_rms_norm(ctx, q, hp.qk_norm_eps);
    q = ggml_mul(ctx, q, ggml_reshape_3d(ctx, ly.attn_temp, 1, nh, 1)); // broadcast over hd, n
    k = ggml_rms_norm(ctx, k, hp.qk_norm_eps);

    // interleaved RoPE (is_neox=False => GGML_ROPE_TYPE_NORMAL)
    q = ggml_rope_ext(ctx, q, g.pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                      (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, g.pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                      (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor * kqv = nullptr;

    if (g.kc) {
        // append new K/V into the F16 cache at positions g.pos via in-place set_rows. The index
        // is data (g.pos content), so node properties stay constant across decode steps -> the
        // decode graph is static-shape and ggml-cuda can capture/replay it.
        ggml_tensor * kc = (*g.kc)[il];   // [hd, max_kv, nkv] F16
        ggml_tensor * vc = (*g.vc)[il];
        // set_rows takes an F32 source and converts into the F16 cache in place.
        ggml_tensor * kb = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, n, nkv] F32
        ggml_tensor * vb = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, kc, kb, g.pos));
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, vc, vb, g.pos));

        if (g.decode) {
            // fused flash attention over the cache window (one kernel; mask invalidates j>pos).
            // K/V are the F16 cache views directly; output is already [hd, nh, n].
            ggml_tensor * qf = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, n, nh]
            kqv = ggml_flash_attn_ext(ctx, qf, kc, vc, g.mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        }
    }
    if (!kqv) {
        // manual causal self-attention over the n tokens (prefill / validate / recompute)
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, n, nh]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, n, nkv]
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                        // [n, n, nh]
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd));
        kq = ggml_diag_mask_inf(ctx, kq, 0);
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n, hd, nkv]
        kqv = ggml_mul_mat(ctx, vp, kq);                                     // [hd, n, nh]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));            // [hd, nh, n]
    }


    // per-head sigmoid gate (broadcast over head_dim), then merge heads and project
    kqv = ggml_mul(ctx, kqv, ggml_reshape_3d(ctx, gate, 1, nh, n));
    ggml_tensor * o = ggml_reshape_2d(ctx, kqv, hd * nh, n);                 // [n_embd, n]
    return ggml_mul_mat(ctx, ly.wo, o);
}

// Dense SwiGLU FFN: down( up(x) * silu(gate(x)) ).  up=chunk0, gate=chunk1 (silu on gate).
static ggml_tensor * build_dense_ffn(gctx & g, ggml_tensor * cur, const zonos2_layer & ly) {
    ggml_context * ctx = g.ctx;
    ggml_tensor * up   = ggml_mul_mat(ctx, ly.ffn_up,   cur);
    ggml_tensor * gate = ggml_mul_mat(ctx, ly.ffn_gate, cur);
    ggml_tensor * y    = ggml_mul(ctx, up, ggml_silu(ctx, gate));
    return ggml_mul_mat(ctx, ly.ffn_down, y);
}

// MoE (sonic) block with stateful EDA router.
//   router: down_proj(+bias) -> [+ prev_state*eda_scale] -> save state -> RMSNorm
//           -> 3-layer erf-GELU MLP -> softmax -> top-k on (probs+bias), gather probs (no renorm)
//   experts: silu(gate.x)*(up.x) -> down, per-expert via mul_mat_id, weighted sum over top-k.
static ggml_tensor * build_moe(gctx & g, ggml_tensor * cur, const zonos2_layer & ly, int L) {
    const zonos2_hparams & hp = g.m->hp;
    ggml_context * ctx = g.ctx;
    const int ne = (int) hp.n_expert, k = ly.top_k, n = g.n, n_embd = (int) hp.n_embd;

    ggml_tensor * rh = ggml_add(ctx, ggml_mul_mat(ctx, ly.router_down, cur), ly.router_down_b); // [rd, n]
    if (ly.router_eda_scale && g.router_states) {
        rh = ggml_add(ctx, rh, ggml_mul(ctx, g.router_states, ly.router_eda_scale));
    }
    g.router_states = rh;                                    // pre-norm state for next MoE layer
    g.cap("router_states_" + std::to_string(L), rh);

    ggml_tensor * rn = rms_w(ctx, rh, ly.router_norm, hp.rms_eps);
    ggml_tensor * h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ly.router_mlp0, rn), ly.router_mlp0_b));
    h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ly.router_mlp2, h), ly.router_mlp2_b));
    ggml_tensor * rlogits = ggml_mul_mat(ctx, ly.router_mlp4, h);   // [ne, n]
    ggml_tensor * probs   = ggml_soft_max(ctx, rlogits);           // [ne, n]

    ggml_tensor * scores = ggml_add(ctx, probs, ly.router_bias);    // legacy: probs + bias
    ggml_tensor * sel    = ggml_top_k(ctx, scores, k);             // [k, n] i32
    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, ne, n), sel); // [1, k, n]

    ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, n_embd, 1, n);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, ly.ffn_up_exps,   cur3, sel); // [n_ff, k, n]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, ly.ffn_gate_exps, cur3, sel); // [n_ff, k, n]
    ggml_tensor * y    = ggml_mul(ctx, up, ggml_silu(ctx, gate));
    ggml_tensor * exps = ggml_mul_mat_id(ctx, ly.ffn_down_exps, y, sel);    // [n_embd, k, n]
    exps = ggml_mul(ctx, exps, weights);                                    // scale by route prob

    ggml_tensor * moe = ggml_view_2d(ctx, exps, n_embd, n, exps->nb[2], 0);
    for (int j = 1; j < k; ++j) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, exps, n_embd, n, exps->nb[2], (size_t) j * exps->nb[1]));
    }
    return ggml_cont(ctx, moe);
}

// Build the prefill graph. Returns the final node. n_layer_limit < 0 => all layers.
ggml_tensor * build_graph(gctx & g, int n_layer_limit) {
    const zonos2_model   & m  = *g.m;
    const zonos2_hparams & hp = m.hp;
    ggml_context * ctx = g.ctx;
    const int W = (int) hp.n_codebooks + 1;

    // --- multi-embedding: sum of per-column get_rows (F32) ---
    g.ids.resize(W);
    ggml_tensor * emb = nullptr;
    for (int k = 0; k < W; ++k) {
        ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        ggml_set_input(id);
        ggml_set_name(id, ("ids." + std::to_string(k)).c_str());
        g.ids[k] = id;
        ggml_tensor * tbl = (k < (int) hp.n_codebooks) ? m.audio_embd[k] : m.text_embd;
        ggml_tensor * e = ggml_get_rows(ctx, tbl, id); // [n_embd, n]
        emb = emb ? ggml_add(ctx, emb, e) : e;
    }
    g.cap("emb_sum", emb);

    // --- speaker: overwrite column spk_pos with proj(lda(spk)) before emb_norm ---
    // s_lda = W_lda·spk + b_lda  (spk_dim -> lda_dim); s_proj = W_proj·s_lda + b_proj
    // (lda_dim -> n_embd); then REPLACE (not add) the embedding at spk_pos.
    if (g.spk_pos >= 0) {
        g.spk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.spk_dim);
        ggml_set_input(g.spk);
        ggml_set_name(g.spk, "spk");
        ggml_tensor * sl = ggml_add(ctx, ggml_mul_mat(ctx, m.spk_lda_w,  g.spk), m.spk_lda_b);
        ggml_tensor * sp = ggml_add(ctx, ggml_mul_mat(ctx, m.spk_proj_w, sl),    m.spk_proj_b);
        g.cap("spk_proj", sp); // [n_embd]
        emb = ggml_set_1d(ctx, emb, ggml_reshape_1d(ctx, sp, hp.n_embd),
                          (size_t) g.spk_pos * hp.n_embd * ggml_element_size(emb));
        g.cap("emb_after_spk", emb);
    }

    // --- emb_norm: weightless RMSNorm; seeds the residual stream ---
    ggml_tensor * x = g.cap("emb_norm", ggml_rms_norm(ctx, emb, hp.rms_eps));

    const int nl = (n_layer_limit < 0) ? (int) hp.n_layer
                                       : std::min(n_layer_limit, (int) hp.n_layer);
    if (nl > 0) {
        g.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        ggml_set_input(g.pos);
        ggml_set_name(g.pos, "pos");
    }
    if (g.decode) {
        g.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.max_kv, 1); // flash_attn_ext requires F16 mask
        ggml_set_input(g.mask);
        ggml_set_name(g.mask, "mask");
    }

    // --- pre-norm residual transformer ---
    ggml_tensor * res = nullptr;
    for (int L = 0; L < nl; ++L) {
        const zonos2_layer & ly = m.layers[L];
        const std::string s = std::to_string(L);

        res = res ? ggml_add(ctx, x, res) : x;
        ggml_tensor * cur = g.cap("attn_in_" + s, rms_w(ctx, res, ly.attn_norm, hp.rms_eps));
        ggml_tensor * attn = g.cap("attn_out_" + s, build_attention(g, cur, ly, L));
        res = g.cap("layer_res_" + s, ggml_add(ctx, attn, res));

        cur = g.cap("ffn_in_" + s, rms_w(ctx, res, ly.ffn_norm, hp.rms_eps));
        ggml_tensor * ffn = ly.is_moe ? build_moe(g, cur, ly, L) : build_dense_ffn(g, cur, ly);
        x = g.cap("ffn_out_" + s, ffn);
    }

    if (n_layer_limit < 0) {
        res = res ? ggml_add(ctx, x, res) : x;
        ggml_tensor * h = g.cap("out_norm", rms_w(ctx, res, m.output_norm, hp.rms_eps));

        // output head: [audio_vocab*n_codebooks, n] -> reshape [audio_vocab, n_codebooks, n] -> softcap
        ggml_tensor * mout = g.cap("mout", ggml_mul_mat(ctx, m.output, h)); // [9234, n]
        ggml_tensor * logits = ggml_reshape_3d(ctx, mout, hp.audio_vocab, hp.n_codebooks, g.n);
        const float cap = hp.logit_softcap;
        if (cap > 0.0f) {
            logits = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, logits, 1.0f / cap)), cap);
        }
        x = g.cap("logits", logits); // [audio_vocab, n_codebooks, n]
    }
    return x;
}

void save_tensor(const std::string & dir, const std::string & name, ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "validate: skip %s (type %s)\n", name.c_str(), ggml_type_name(t->type));
        return;
    }
    const int nd = ggml_n_dims(t);
    std::vector<int64_t> shape(nd > 0 ? nd : 1, 1);
    for (int i = 0; i < nd; ++i) shape[i] = t->ne[nd - 1 - i]; // reverse ne -> numpy C-order
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    npy::save_f32(dir + "/" + name + ".npy", buf.data(), shape);
}

// Build + compute one prefill graph over `ids` (row-major [n_tokens, W=n_codebooks+1] floats).
// Shared by zonos2_validate (capture=true, dumps caps) and zonos2_logits (capture=false, bulk
// read). On success returns build_graph's output tensor; the caller must read it back before
// freeing *ctx_out / *galloc_out. Returns nullptr on failure (ctx/galloc still owned by caller).
static ggml_tensor * prefill_run(const zonos2_model & m, const float * ids, int n_tokens,
                                 bool capture, int n_layer_limit, const float * spk, int spk_pos,
                                 gctx & g, ggml_context *& ctx, ggml_gallocr_t & galloc) {
    struct ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
    ctx = ggml_init(ip);

    g.m = &m; g.ctx = ctx; g.n = n_tokens; g.capture = capture;
    g.spk_pos = spk ? spk_pos : -1;
    ggml_tensor * out = build_graph(g, n_layer_limit);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, out);
    for (auto & c : g.caps) ggml_build_forward_expand(gf, c.second);

    galloc = ggml_gallocr_new(m.buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "prefill: graph alloc failed\n");
        return nullptr;
    }

    // set inputs: column k of the [n_tokens, W] id matrix
    const int W = (int) m.hp.n_codebooks + 1;
    std::vector<int32_t> col(n_tokens);
    for (int k = 0; k < W; ++k) {
        for (int t = 0; t < n_tokens; ++t) col[t] = (int32_t) lroundf(ids[(size_t) t * W + k]);
        ggml_backend_tensor_set(g.ids[k], col.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    }
    if (g.pos) {
        for (int t = 0; t < n_tokens; ++t) col[t] = t; // contiguous 0-based positions
        ggml_backend_tensor_set(g.pos, col.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    }
    if (g.spk) {
        ggml_backend_tensor_set(g.spk, spk, 0, (size_t) m.hp.spk_dim * sizeof(float));
    }

    fprintf(stderr, "prefill: computing graph (%d nodes) ...\n", ggml_graph_n_nodes(gf));
    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "prefill: compute failed\n");
        return nullptr;
    }
    return out;
}

} // namespace

bool zonos2_validate(const zonos2_model & m, const float * ids, int n_tokens,
                     const char * out_dir, int n_layer_limit,
                     const float * spk, int spk_pos) {
    gctx g;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * out = prefill_run(m, ids, n_tokens, /*capture=*/true, n_layer_limit, spk, spk_pos, g, ctx, galloc);
    const bool ok = out != nullptr;
    if (ok) {
        for (auto & c : g.caps) save_tensor(out_dir, c.first, c.second);
        fprintf(stderr, "validate: wrote %zu tensors to %s\n", g.caps.size(), out_dir);
    }
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
    return ok;
}

bool zonos2_logits(const zonos2_model & m, const float * ids, int n_tokens,
                   std::vector<float> & out_logits, const float * spk, int spk_pos) {
    gctx g;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * logits = prefill_run(m, ids, n_tokens, /*capture=*/false, /*n_layer_limit=*/-1,
                                       spk, spk_pos, g, ctx, galloc);
    const bool ok = logits != nullptr;
    if (ok) {
        out_logits.resize((size_t) m.hp.audio_vocab * m.hp.n_codebooks * n_tokens);
        ggml_backend_tensor_get(logits, out_logits.data(), 0, out_logits.size() * sizeof(float));
    }
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
    return ok;
}

// O(n^2) reference path: rebuild the full prefill graph each step (no cache).
static int generate_recompute(const zonos2_model & m, const float * prompt_ids, int n0,
                              int max_frames, const zonos2_sampling & sp,
                              std::vector<int32_t> & out_codes, int & eos_frame,
                              const float * spk, int spk_pos) {
    const zonos2_hparams & hp = m.hp;
    const int W = (int) hp.n_codebooks + 1;
    const int ncb = (int) hp.n_codebooks;
    const int av = (int) hp.audio_vocab;

    std::vector<int32_t> seq((size_t) n0 * W);
    for (int i = 0; i < n0 * W; ++i) seq[i] = (int32_t) lroundf(prompt_ids[i]);

    zonos2_sampler smp(sp, ncb, av);
    eos_frame = -1;
    int countdown = -1;
    int n_frames = 0;

    ggml_gallocr_t galloc = ggml_gallocr_new(m.buft);

    for (int step = 0; step < max_frames; ++step) {
        const int n = (int) (seq.size() / W);

        struct ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
        ggml_context * ctx = ggml_init(ip);
        gctx g; g.m = &m; g.ctx = ctx; g.n = n; g.capture = false;
        g.spk_pos = spk ? spk_pos : -1;
        ggml_tensor * logits = build_graph(g, -1); // [audio_vocab, n_codebooks, n]
        ggml_set_output(logits);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_alloc_graph(galloc, gf);

        std::vector<int32_t> col(n);
        for (int k = 0; k < W; ++k) {
            for (int t = 0; t < n; ++t) col[t] = seq[(size_t) t * W + k];
            ggml_backend_tensor_set(g.ids[k], col.data(), 0, (size_t) n * sizeof(int32_t));
        }
        for (int t = 0; t < n; ++t) col[t] = t;
        ggml_backend_tensor_set(g.pos, col.data(), 0, (size_t) n * sizeof(int32_t));
        if (g.spk) ggml_backend_tensor_set(g.spk, spk, 0, (size_t) hp.spk_dim * sizeof(float));

        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "generate: compute failed at step %d\n", step);
            ggml_free(ctx); break;
        }

        // read the last frame's logits: contiguous block [audio_vocab, n_codebooks]
        std::vector<float> lo((size_t) av * ncb);
        const size_t frame_bytes = (size_t) av * ncb * sizeof(float);
        ggml_backend_tensor_get(logits, lo.data(), (size_t) (n - 1) * frame_bytes, frame_bytes);
        ggml_free(ctx);

        std::vector<int> frame(ncb);
        for (int cb = 0; cb < ncb; ++cb) frame[cb] = smp.sample(&lo[(size_t) cb * av], cb);
        smp.accept(frame);

        for (int cb = 0; cb < ncb; ++cb) out_codes.push_back(frame[cb]);
        ++n_frames;

        // feed back as next input frame (audio codes + text pad); delay handled by shear_up later
        for (int cb = 0; cb < ncb; ++cb) seq.push_back(frame[cb]);
        seq.push_back((int32_t) hp.text_vocab);

        // EOS: first eoa in any codebook -> align frame, then count down n_codebooks+1 steps
        int max_eoa = -1;
        for (int cb = 0; cb < ncb; ++cb) if (frame[cb] == (int) hp.eoa_id) max_eoa = cb;
        if (eos_frame < 0 && max_eoa >= 0) {
            eos_frame = std::max(0, step - max_eoa);
            countdown = ncb + 1;
        }
        if (countdown > 0 && --countdown == 0) { ++step; break; }

        if ((step % 32) == 0) fprintf(stderr, "generate: step %d (seq=%d)\n", step, n);
    }

    ggml_gallocr_free(galloc);
    return n_frames;
}

// ---------------------------------------------------------------------------
// KV cache + real-time generation
// ---------------------------------------------------------------------------

bool zonos2_context_init(zonos2_context & c, const zonos2_model & m, int max_seq) {
    c.model = &m; c.max_seq = max_seq; c.n_past = 0;
    const int hd = (int) m.hp.head_dim, nkv = (int) m.hp.n_head_kv, nl = (int) m.hp.n_layer;

    struct ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (2 * nl + 8), nullptr, /*no_alloc=*/ true };
    c.ctx_kv = ggml_init(ip);
    c.k_cache.resize(nl);
    c.v_cache.resize(nl);
    for (int i = 0; i < nl; ++i) {
        // F16 layout [head_dim, max_seq, n_head_kv]: set_rows writes a position (ne1) and the
        // tensor is directly usable as flash-attn K/V (contiguous in head_dim, no conversion).
        c.k_cache[i] = ggml_new_tensor_3d(c.ctx_kv, GGML_TYPE_F16, hd, max_seq, nkv);
        c.v_cache[i] = ggml_new_tensor_3d(c.ctx_kv, GGML_TYPE_F16, hd, max_seq, nkv);
        ggml_set_name(c.k_cache[i], ("k_cache." + std::to_string(i)).c_str());
        ggml_set_name(c.v_cache[i], ("v_cache." + std::to_string(i)).c_str());
    }
    c.buf_kv = ggml_backend_alloc_ctx_tensors(c.ctx_kv, m.backend);
    if (!c.buf_kv) { fprintf(stderr, "zonos2: KV cache alloc failed (%d seq)\n", max_seq); return false; }
    ggml_backend_buffer_clear(c.buf_kv, 0); // zero so masked-out (unwritten) positions never NaN
    c.galloc = ggml_gallocr_new(m.buft);
    return c.galloc != nullptr;
}

void zonos2_context_free(zonos2_context & c) {
    if (c.galloc) ggml_gallocr_free(c.galloc);
    if (c.buf_kv) ggml_backend_buffer_free(c.buf_kv);
    if (c.ctx_kv) ggml_free(c.ctx_kv);
    c = zonos2_context();
}

// O(n) real-time path: one prefill, then single-token decodes against a persistent KV cache.
// The decode graph is built + allocated ONCE and reused every step (only input data + cache
// content change), so its node properties stay constant and ggml-cuda captures/replays it as a
// CUDA graph — collapsing ~1731 kernel launches/step into one.
static int generate_kv(const zonos2_model & m, const float * prompt_ids, int n0,
                       int max_frames, const zonos2_sampling & sp,
                       std::vector<int32_t> & out_codes, int & eos_frame,
                       const float * spk, int spk_pos) {
    const zonos2_hparams & hp = m.hp;
    const int W = (int) hp.n_codebooks + 1, ncb = (int) hp.n_codebooks, av = (int) hp.audio_vocab;
    const int max_seq = ((n0 + max_frames + 8 + 255) / 256) * 256; // multiple of FATTN_KQ_STRIDE (flash vec kernel)

    zonos2_context c;
    if (!zonos2_context_init(c, m, max_seq)) return 0;

    zonos2_sampler smp(sp, ncb, av);
    eos_frame = -1; int countdown = -1; int n_frames = 0;
    std::vector<float> lo((size_t) av * ncb);
    const size_t frame_bytes = (size_t) av * ncb * sizeof(float);

    // ---- prefill (one-off graph): populate the cache, get frame-0 logits ----
    {
        struct ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
        ggml_context * ctx = ggml_init(ip);
        gctx g; g.m = &m; g.ctx = ctx; g.n = n0; g.capture = false;
        g.spk_pos = spk ? spk_pos : -1;
        g.kc = &c.k_cache; g.vc = &c.v_cache; g.max_kv = max_seq; g.decode = false;
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false); g.gf = gf;
        ggml_tensor * logits = build_graph(g, -1); ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        bool ok = ggml_gallocr_alloc_graph(c.galloc, gf);
        if (ok) {
            std::vector<int32_t> col(n0);
            for (int k = 0; k < W; ++k) {
                for (int t = 0; t < n0; ++t) col[t] = (int32_t) lroundf(prompt_ids[(size_t) t * W + k]);
                ggml_backend_tensor_set(g.ids[k], col.data(), 0, (size_t) n0 * sizeof(int32_t));
            }
            for (int t = 0; t < n0; ++t) col[t] = t;
            ggml_backend_tensor_set(g.pos, col.data(), 0, (size_t) n0 * sizeof(int32_t));
            if (g.spk) ggml_backend_tensor_set(g.spk, spk, 0, (size_t) hp.spk_dim * sizeof(float));
            ok = ggml_backend_graph_compute(m.backend, gf) == GGML_STATUS_SUCCESS;
            if (ok) ggml_backend_tensor_get(logits, lo.data(), (size_t) (n0 - 1) * frame_bytes, frame_bytes);
        }
        ggml_free(ctx);
        if (!ok) { zonos2_context_free(c); return 0; }
    }
    c.n_past = n0;

    // ---- persistent decode graph: built + allocated once, reused every step ----
    struct ggml_init_params ipd = { (size_t) 64 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx_dec = ggml_init(ipd);
    gctx gd; gd.m = &m; gd.ctx = ctx_dec; gd.n = 1; gd.capture = false;
    gd.kc = &c.k_cache; gd.vc = &c.v_cache; gd.max_kv = max_seq; gd.decode = true;
    ggml_cgraph * gfd = ggml_new_graph_custom(ctx_dec, 16384, false); gd.gf = gfd;
    ggml_tensor * dlogits = build_graph(gd, -1); ggml_set_output(dlogits);
    ggml_build_forward_expand(gfd, dlogits);
    ggml_gallocr_t galloc_dec = ggml_gallocr_new(m.buft);
    if (!ggml_gallocr_alloc_graph(galloc_dec, gfd)) {
        fprintf(stderr, "generate: decode graph alloc failed\n");
        ggml_gallocr_free(galloc_dec); ggml_free(ctx_dec); zonos2_context_free(c); return 0;
    }

    std::vector<int32_t> idcol(1);
    std::vector<ggml_fp16_t> maskbuf(max_seq);
    const ggml_fp16_t MASK0 = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t MASKINF = ggml_fp32_to_fp16(-INFINITY);

    int decode_steps = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < max_frames; ++step) {
        std::vector<int> frame(ncb);
        for (int cb = 0; cb < ncb; ++cb) frame[cb] = smp.sample(&lo[(size_t) cb * av], cb);
        smp.accept(frame);
        for (int cb = 0; cb < ncb; ++cb) out_codes.push_back(frame[cb]);
        ++n_frames;

        int max_eoa = -1;
        for (int cb = 0; cb < ncb; ++cb) if (frame[cb] == (int) hp.eoa_id) max_eoa = cb;
        if (eos_frame < 0 && max_eoa >= 0) { eos_frame = std::max(0, step - max_eoa); countdown = ncb + 1; }
        const bool finished = (countdown > 0 && --countdown == 0);
        if (finished || step + 1 >= max_frames) break;

        // decode the sampled frame at position c.n_past: set_rows writes it, mask exposes [0, n_past]
        const int pos = c.n_past;
        for (int k = 0; k < W; ++k) {
            idcol[0] = (k < ncb) ? frame[k] : (int) hp.text_vocab;
            ggml_backend_tensor_set(gd.ids[k], idcol.data(), 0, sizeof(int32_t));
        }
        idcol[0] = pos;
        ggml_backend_tensor_set(gd.pos, idcol.data(), 0, sizeof(int32_t));
        for (int j = 0; j < max_seq; ++j) maskbuf[j] = (j <= pos) ? MASK0 : MASKINF;
        ggml_backend_tensor_set(gd.mask, maskbuf.data(), 0, (size_t) max_seq * sizeof(ggml_fp16_t));

        if (ggml_backend_graph_compute(m.backend, gfd) != GGML_STATUS_SUCCESS) break;
        ggml_backend_tensor_get(dlogits, lo.data(), 0, frame_bytes);
        c.n_past += 1;
        ++decode_steps;
    }
    {
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double frame_s = 512.0 / 44100.0; // one codebook frame at 44.1 kHz
        fprintf(stderr, "generate(kv): %d decode steps in %.3fs = %.1f frames/s (real-time = %.1f), RTF=%.2f\n",
                decode_steps, dt, decode_steps / dt, 1.0 / frame_s, (dt / decode_steps) / frame_s);
    }

    ggml_gallocr_free(galloc_dec);
    ggml_free(ctx_dec);
    zonos2_context_free(c);
    return n_frames;
}

int zonos2_generate(const zonos2_model & m, const float * prompt_ids, int n0,
                    int max_frames, const zonos2_sampling & sp,
                    std::vector<int32_t> & out_codes, int & eos_frame, bool use_kv,
                    const float * spk, int spk_pos) {
    return use_kv
        ? generate_kv       (m, prompt_ids, n0, max_frames, sp, out_codes, eos_frame, spk, spk_pos)
        : generate_recompute(m, prompt_ids, n0, max_frames, sp, out_codes, eos_frame, spk, spk_pos);
}
