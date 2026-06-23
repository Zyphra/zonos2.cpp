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
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct gctx {
    const zonos2_model * m = nullptr;
    ggml_context * ctx = nullptr;
    int n = 0;
    std::vector<ggml_tensor *> ids;
    // Two position inputs (length g.n). For a single sequence they coincide (0..n-1); for a
    // batched decode they differ: pos_rope carries each column's logical position (for RoPE),
    // pos_cache the absolute cache row (slot*slot_cap + logical) where set_rows writes its K/V.
    ggml_tensor * pos_rope  = nullptr;
    ggml_tensor * pos_cache = nullptr;
    ggml_tensor * router_states = nullptr; // EDA state threaded across MoE layers

    // speaker conditioning: overwrite the embedding at column spk_pos with the
    // projected speaker vector (before emb_norm). spk_pos < 0 => no speaker.
    ggml_tensor * spk = nullptr; // input speaker embedding [spk_dim]
    int spk_pos = -1;
    bool capture = true;                   // false during generation (lean graph)
    bool cap_moe = false;                   // capture per-MoE-layer inputs for imatrix collection
    std::vector<std::pair<std::string, ggml_tensor *>> caps;

    // force a read-back output regardless of `capture` (used for imatrix moe inputs).
    ggml_tensor * out_named(const std::string & name, ggml_tensor * t) {
        if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
        ggml_set_output(t);
        ggml_set_name(t, name.c_str());
        caps.push_back({name, t});
        return t;
    }

    // KV cache. The cache is a unified per-layer tensor [head_dim, n_slots*slot_cap, n_head_kv];
    // slot s owns the contiguous row band [s*slot_cap, (s+1)*slot_cap). For the single-sequence
    // paths n_slots==1 and slot_cap==max_kv, recovering the original [hd, max_kv, nkv] layout.
    std::vector<ggml_tensor *> * kc = nullptr;  // per-layer K cache
    std::vector<ggml_tensor *> * vc = nullptr;  // per-layer V cache (same layout)
    bool decode = false;                        // true: attend over each slot's cache band via mask
    int  n_slots  = 1;                          // batch slots = ne3 of the decode attention
    int  slot_cap = 0;                          // per-slot cache rows (multiple of FATTN stride 256)
    ggml_tensor * mask = nullptr;               // [slot_cap, 1, 1, n_slots] additive mask (decode only)
    ggml_tensor * mask_causal = nullptr;        // [n, n] F16 causal additive mask (prefill / non-decode)
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

// Fill an [n_kv, n_q] F16 causal additive mask: 0 where key kv <= query q, -inf above the
// diagonal. Element (kv, q) lives at index q*n + kv. Positions are the contiguous 0..n-1 used
// by every prefill / non-decode graph, so the same lower-triangular mask serves all of them.
static void set_causal_mask(ggml_tensor * t, int n) {
    std::vector<ggml_fp16_t> mk((size_t) n * n);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n; ++q)
        for (int kv = 0; kv < n; ++kv)
            mk[(size_t) q * n + kv] = (kv <= q) ? zero : ninf;
    ggml_backend_tensor_set(t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
}

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

    // interleaved RoPE (is_neox=False => GGML_ROPE_TYPE_NORMAL); logical positions
    q = ggml_rope_ext(ctx, q, g.pos_rope, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                      (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, g.pos_rope, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                      (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor * kqv = nullptr;

    if (g.kc) {
        // append new K/V into the F16 cache at rows g.pos_cache via in-place set_rows. The index
        // is data (g.pos_cache content), so node properties stay constant across decode steps ->
        // the decode graph is static-shape and ggml-cuda can capture/replay it. Each of the n
        // columns writes a unique absolute row (slot*slot_cap + logical pos).
        ggml_tensor * kc = (*g.kc)[il];   // [hd, n_slots*slot_cap, nkv] F16
        ggml_tensor * vc = (*g.vc)[il];
        // set_rows takes an F32 source and converts into the F16 cache in place.
        ggml_tensor * kb = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, n, nkv] F32
        ggml_tensor * vb = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, kc, kb, g.pos_cache));
        ggml_build_forward_expand(g.gf, ggml_set_rows(ctx, vc, vb, g.pos_cache));

        if (g.decode) {
            // Per-slot batched flash attention. View the unified cache as 4D
            // [hd, slot_cap, nkv, n_slots] — a no-copy reinterpret since row = slot*slot_cap + pos
            // — and reshape the single per-slot query to [hd, 1, nh, n_slots]. The ne33 per-slot
            // mask (g.mask = [slot_cap,1,1,n_slots]) confines slot s to its own band [0, pos_s], so
            // one fused kernel handles all slots with no cross-slot work. At n_slots==1 this is the
            // original [hd,max_kv,nkv] cache / [hd,1,nh,1] query — bit-identical to single-seq.
            ggml_tensor * k4 = ggml_view_4d(ctx, kc, hd, g.slot_cap, nkv, g.n_slots,
                                            kc->nb[1], kc->nb[2], (size_t) g.slot_cap * kc->nb[1], 0);
            ggml_tensor * v4 = ggml_view_4d(ctx, vc, hd, g.slot_cap, nkv, g.n_slots,
                                            vc->nb[1], vc->nb[2], (size_t) g.slot_cap * vc->nb[1], 0);
            ggml_tensor * qf = ggml_reshape_4d(ctx, q, hd, 1, nh, g.n_slots); // [hd, 1, nh, n_slots]
            kqv = ggml_flash_attn_ext(ctx, qf, k4, v4, g.mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            kqv = ggml_reshape_3d(ctx, kqv, hd, nh, g.n_slots);               // [hd, nh, n]
        }
    }
    if (!kqv) {
        // manual causal self-attention over the n tokens (prefill / validate / recompute).
        // ggml_diag_mask_inf has no Metal kernel, so causality comes from an additive F16
        // [n_kv, n_q] mask folded into soft_max_ext (which also applies the 1/sqrt(hd) scale).
        // Kept as mul_mat+soft_max rather than flash_attn_ext: for these short prefills Metal's
        // mul_mm+soft_max is far cheaper than the F32 flash kernel + its pad/blk preprocessing,
        // which dominated time-to-first-audio. soft_max_ext broadcasts the mask over the nh heads.
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, n, nh]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, n, nkv]
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                        // [n, n, nh]
        kq = ggml_soft_max_ext(ctx, kq, g.mask_causal, 1.0f / sqrtf((float) hd), 0.0f);
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
    if (g.cap_moe) {
        g.out_named("moe_in_"  + std::to_string(L), cur); // [n_embd, n]  gate/up input
        g.out_named("moe_y_"   + std::to_string(L), y);   // [n_ff, k, n] down input
        g.out_named("moe_sel_" + std::to_string(L), sel); // [k, n] i32   routing
    }
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
        // Replace column spk_pos of emb with sp, built as a concat of contiguous column
        // views rather than ggml_set_1d. GGML_OP_SET's non-inplace path on the Metal backend
        // corrupts the columns it should be copying through unchanged (only the written slice
        // is correct), which silently destroyed the speaker conditioning and made the model
        // emit EOS on the first frame. concat is correct on every backend; emb is contiguous
        // [n_embd, n], so each column slice below is itself a contiguous block.
        const int ne = (int) hp.n_embd, n = g.n, p = g.spk_pos;
        ggml_tensor * out = ggml_reshape_2d(ctx, sp, ne, 1);            // speaker column [n_embd, 1]
        if (p > 0) {
            ggml_tensor * head = ggml_view_2d(ctx, emb, ne, p, emb->nb[1], 0);
            out = ggml_concat(ctx, head, out, 1);                      // [n_embd, p+1]
        }
        if (p < n - 1) {
            ggml_tensor * tail = ggml_view_2d(ctx, emb, ne, n - 1 - p, emb->nb[1],
                                              (size_t) (p + 1) * emb->nb[1]);
            out = ggml_concat(ctx, out, tail, 1);                      // [n_embd, n]
        }
        emb = ggml_cont(ctx, out);
        g.cap("emb_after_spk", emb);
    }

    // --- emb_norm: weightless RMSNorm; seeds the residual stream ---
    ggml_tensor * x = g.cap("emb_norm", ggml_rms_norm(ctx, emb, hp.rms_eps));

    const int nl = (n_layer_limit < 0) ? (int) hp.n_layer
                                       : std::min(n_layer_limit, (int) hp.n_layer);
    if (nl > 0) {
        g.pos_rope = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
        ggml_set_input(g.pos_rope);
        ggml_set_name(g.pos_rope, "pos_rope");
        if (g.kc) { // absolute cache write rows; only the KV-cache paths consume this
            g.pos_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, g.n);
            ggml_set_input(g.pos_cache);
            ggml_set_name(g.pos_cache, "pos_cache");
        }
        if (!g.decode) { // additive causal mask for the prefill soft_max_ext attention branch
            g.mask_causal = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.n, g.n);
            ggml_set_input(g.mask_causal);
            ggml_set_name(g.mask_causal, "mask_causal");
        }
    }
    if (g.decode) {
        // per-slot causal mask on ne3; ne2 must stay 1 for the FATTN kernel selector. F16 required.
        g.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, g.slot_cap, 1, 1, g.n_slots);
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
    if (g.pos_rope) {
        for (int t = 0; t < n_tokens; ++t) col[t] = t; // contiguous 0-based positions
        ggml_backend_tensor_set(g.pos_rope, col.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    }
    if (g.mask_causal) set_causal_mask(g.mask_causal, n_tokens);
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

bool zonos2_moe_capture(const zonos2_model & m, const float * ids, int n_tokens,
                        std::vector<zonos2_moe_act> & out, const float * spk, int spk_pos) {
    gctx g;
    g.cap_moe = true;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * res = prefill_run(m, ids, n_tokens, /*capture=*/false, /*n_layer_limit=*/-1,
                                    spk, spk_pos, g, ctx, galloc);
    const bool ok = res != nullptr;
    if (ok) {
        std::map<int, zonos2_moe_act> by_layer;            // keyed by layer index, ordered
        for (auto & c : g.caps) {
            const std::string & nm = c.first;
            const int L = atoi(nm.c_str() + nm.rfind('_') + 1);
            ggml_tensor * t = c.second;
            zonos2_moe_act & a = by_layer[L];
            a.layer = L;
            if (nm.rfind("moe_in_", 0) == 0) {
                a.n_embd = (int) t->ne[0]; a.n = (int) t->ne[1];
                a.moe_in.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.moe_in.data(), 0, ggml_nbytes(t));
            } else if (nm.rfind("moe_y_", 0) == 0) {
                a.n_ff = (int) t->ne[0]; a.k = (int) t->ne[1]; a.n = (int) t->ne[2];
                a.moe_y.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.moe_y.data(), 0, ggml_nbytes(t));
            } else if (nm.rfind("moe_sel_", 0) == 0) {
                a.k = (int) t->ne[0]; a.n = (int) t->ne[1];
                a.sel.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, a.sel.data(), 0, ggml_nbytes(t));
            }
        }
        for (auto & kv : by_layer) out.push_back(std::move(kv.second));
    }
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
    return ok;
}

// O(n^2) reference path: rebuild the full prefill graph each step (no cache).
static int generate_recompute(const zonos2_model & m, const float * prompt_ids, int n0,
                              int max_frames, const zonos2_sampling & sp,
                              std::vector<int32_t> & out_codes, int & eos_frame,
                              const float * spk, int spk_pos, const zonos2_frame_cb & on_frame) {
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
        ggml_backend_tensor_set(g.pos_rope, col.data(), 0, (size_t) n * sizeof(int32_t));
        if (g.mask_causal) set_causal_mask(g.mask_causal, n);
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

        if (on_frame) {
            const std::vector<int32_t> fc(frame.begin(), frame.end());
            if (!on_frame(n_frames - 1, fc.data(), ncb)) break;   // client aborted
        }

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
// Batched / continuous-batching runtime (KV cache + real-time generation)
// ---------------------------------------------------------------------------

static int round_up_256(int x) { return ((x + 255) / 256) * 256; }

// Build one persistent decode graph of the given column width, viewing KV-cache bands [0,width).
// All ladder widths share bc.k_cache/v_cache; slot index == column == cache band, so the formulas
// in build_graph (cache row = slot*slot_cap + pos) are width-independent.
static bool build_decode_graph(zonos2_batch_ctx & bc, const zonos2_model & m, int width,
                               zonos2_batch_ctx::dec_graph & dg) {
    dg.width = width;
    struct ggml_init_params ipd = { (size_t) 24 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
    dg.ctx = ggml_init(ipd);
    gctx gd; gd.m = &m; gd.ctx = dg.ctx; gd.n = width; gd.capture = false;
    gd.kc = &bc.k_cache; gd.vc = &bc.v_cache; gd.decode = true;
    gd.n_slots = width; gd.slot_cap = bc.slot_cap;
    dg.gf = ggml_new_graph_custom(dg.ctx, 16384, false); gd.gf = dg.gf;
    ggml_tensor * logits = build_graph(gd, -1); ggml_set_output(logits);
    ggml_build_forward_expand(dg.gf, logits);
    dg.galloc = ggml_gallocr_new(m.buft);
    if (!dg.galloc || !ggml_gallocr_alloc_graph(dg.galloc, dg.gf)) {
        fprintf(stderr, "zonos2: decode graph alloc failed (width=%d)\n", width);
        return false;
    }
    dg.ids       = gd.ids;
    dg.pos_rope  = gd.pos_rope;
    dg.pos_cache = gd.pos_cache;
    dg.mask      = gd.mask;
    dg.logits    = logits;
    return true;
}

bool zonos2_batch_init(zonos2_batch_ctx & bc, const zonos2_model & m, int n_slots, int slot_cap_frames) {
    const zonos2_hparams & hp = m.hp;
    bc = zonos2_batch_ctx();
    bc.model    = &m;
    bc.n_slots  = n_slots;
    bc.slot_cap = round_up_256(slot_cap_frames);
    bc.W   = (int) hp.n_codebooks + 1;
    bc.ncb = (int) hp.n_codebooks;
    bc.av  = (int) hp.audio_vocab;

    const int hd = (int) hp.head_dim, nkv = (int) hp.n_head_kv, nl = (int) hp.n_layer;
    const int64_t total = (int64_t) n_slots * bc.slot_cap;

    // unified KV cache [hd, n_slots*slot_cap, nkv] per layer; slot s owns rows [s*slot_cap, ...)
    struct ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (2 * nl + 8), nullptr, /*no_alloc=*/ true };
    bc.ctx_kv = ggml_init(ip);
    bc.k_cache.resize(nl);
    bc.v_cache.resize(nl);
    for (int i = 0; i < nl; ++i) {
        bc.k_cache[i] = ggml_new_tensor_3d(bc.ctx_kv, GGML_TYPE_F16, hd, total, nkv);
        bc.v_cache[i] = ggml_new_tensor_3d(bc.ctx_kv, GGML_TYPE_F16, hd, total, nkv);
        ggml_set_name(bc.k_cache[i], ("k_cache." + std::to_string(i)).c_str());
        ggml_set_name(bc.v_cache[i], ("v_cache." + std::to_string(i)).c_str());
    }
    bc.buf_kv = ggml_backend_alloc_ctx_tensors(bc.ctx_kv, m.backend);
    if (!bc.buf_kv) {
        fprintf(stderr, "zonos2: batch KV alloc failed (%d slots x %d cap)\n", n_slots, bc.slot_cap);
        zonos2_batch_free(bc); return false;
    }
    ggml_backend_buffer_clear(bc.buf_kv, 0); // zero so idle / unwritten positions never NaN

    // Decode-graph ladder over widths {1,2,4,...,n_slots}: each step runs the smallest width that
    // covers the active slots, so a solo request pays 1-column compute instead of n_slots-column.
    std::vector<int> widths;
    for (int w = 1; w < n_slots; w *= 2) widths.push_back(w);
    widths.push_back(n_slots);
    bc.dec_ladder.resize(widths.size());
    for (size_t i = 0; i < widths.size(); ++i) {
        if (!build_decode_graph(bc, m, widths[i], bc.dec_ladder[i])) {
            zonos2_batch_free(bc); return false;
        }
    }

    bc.galloc_prefill = ggml_gallocr_new(m.buft);

    bc.h_ids.assign((size_t) bc.W * n_slots, 0);
    bc.h_pos_rope.assign(n_slots, 0);
    bc.h_pos_cache.assign(n_slots, 0);
    bc.h_mask.assign((size_t) bc.slot_cap * n_slots, 0);
    bc.h_logits.assign((size_t) n_slots * bc.av * bc.ncb, 0.0f);
    return bc.galloc_prefill != nullptr;
}

void zonos2_batch_free(zonos2_batch_ctx & bc) {
    for (auto & dg : bc.dec_ladder) {
        if (dg.galloc) ggml_gallocr_free(dg.galloc);
        if (dg.ctx)    ggml_free(dg.ctx);
    }
    if (bc.galloc_prefill) ggml_gallocr_free(bc.galloc_prefill);
    if (bc.buf_kv)         ggml_backend_buffer_free(bc.buf_kv);
    if (bc.ctx_kv)         ggml_free(bc.ctx_kv);
    bc = zonos2_batch_ctx();
}

bool zonos2_batch_slot_prefill(zonos2_batch_ctx & bc, int slot, const float * prompt_ids, int n0,
                               const float * spk, int spk_pos, float * out_logits) {
    const zonos2_model & m = *bc.model;
    const int W = bc.W, S = bc.slot_cap, av = bc.av, ncb = bc.ncb;
    if (n0 <= 0 || n0 > S) {
        fprintf(stderr, "zonos2: prefill n0=%d exceeds slot_cap=%d\n", n0, S);
        return false;
    }
    // single-sequence prefill graph writing K/V into this slot's band [slot*S, slot*S+n0).
    struct ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    gctx g; g.m = &m; g.ctx = ctx; g.n = n0; g.capture = false;
    g.spk_pos = spk ? spk_pos : -1;
    g.kc = &bc.k_cache; g.vc = &bc.v_cache; g.decode = false; g.n_slots = 1; g.slot_cap = S;
    // Optional profiling: ZONOS2_PREFILL_NLAYERS=N runs only the first N transformer layers
    // (logits are then garbage) so first-audio latency can be attributed across the layer stack.
    const char * nl_env = getenv("ZONOS2_PREFILL_NLAYERS");
    const int prefill_nl = nl_env ? atoi(nl_env) : -1;
    const bool prof = getenv("ZONOS2_PROFILE") != nullptr;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = clk::now();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false); g.gf = gf;
    ggml_tensor * logits = build_graph(g, prefill_nl); ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    bool ok = ggml_gallocr_alloc_graph(bc.galloc_prefill, gf);
    auto t1 = clk::now();
    if (ok) {
        std::vector<int32_t> col(n0);
        for (int k = 0; k < W; ++k) {
            for (int t = 0; t < n0; ++t) col[t] = (int32_t) lroundf(prompt_ids[(size_t) t * W + k]);
            ggml_backend_tensor_set(g.ids[k], col.data(), 0, (size_t) n0 * sizeof(int32_t));
        }
        for (int t = 0; t < n0; ++t) col[t] = t;                  // logical rope positions
        ggml_backend_tensor_set(g.pos_rope, col.data(), 0, (size_t) n0 * sizeof(int32_t));
        for (int t = 0; t < n0; ++t) col[t] = slot * S + t;       // absolute cache rows in this band
        ggml_backend_tensor_set(g.pos_cache, col.data(), 0, (size_t) n0 * sizeof(int32_t));
        if (g.mask_causal) set_causal_mask(g.mask_causal, n0);
        if (g.spk) ggml_backend_tensor_set(g.spk, spk, 0, (size_t) m.hp.spk_dim * sizeof(float));
        auto t2 = clk::now();
        ok = ggml_backend_graph_compute(m.backend, gf) == GGML_STATUS_SUCCESS;
        auto t3 = clk::now();
        if (prof) fprintf(stderr,
            "prefill[prof]: n0=%d nlayers=%d nodes=%d  build+alloc=%.1fms set=%.1fms compute=%.1fms\n",
            n0, prefill_nl, ggml_graph_n_nodes(gf), ms(t0, t1), ms(t1, t2), ms(t2, t3));
        if (ok) ggml_backend_tensor_get(logits, out_logits, (size_t) (n0 - 1) * av * ncb * sizeof(float),
                                        (size_t) av * ncb * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

void zonos2_batch_step(zonos2_batch_ctx & bc, const std::vector<zonos2_slot *> & active) {
    const zonos2_model & m = *bc.model;
    const int N = bc.n_slots, W = bc.W, S = bc.slot_cap, av = bc.av, ncb = bc.ncb;
    const uint16_t M0   = ggml_fp32_to_fp16(0.0f);
    const uint16_t MINF = ggml_fp32_to_fp16(-INFINITY);

    static const bool prof_step = getenv("ZONOS2_PROFILE_STEP") != nullptr;
    using sclk = std::chrono::steady_clock;
    sclk::time_point t_start, ta, tb;
    if (prof_step) t_start = sclk::now();

    // Pick the narrowest ladder graph covering the active slots. Slots are allocated low-index
    // first, so the needed column width is (highest active slot index + 1); a width-Wd graph views
    // KV-cache bands [0,Wd) and is bit-identical to the full-width graph for those columns.
    int eff = 0;
    for (zonos2_slot * sp : active)
        if (sp && sp->active && !sp->done) eff = std::max(eff, sp->index + 1);
    if (eff <= 0) return;                                  // nothing active
    const zonos2_batch_ctx::dec_graph * dg = &bc.dec_ladder.back();
    for (const auto & cand : bc.dec_ladder) if (cand.width >= eff) { dg = &cand; break; }
    const int Wd = dg->width;

    // default every (covered) column to idle: pad ids, position 0, mask exposing only band row 0.
    for (int c = 0; c < Wd; ++c) {
        for (int k = 0; k < W; ++k)
            bc.h_ids[(size_t) k * N + c] = (k < ncb) ? (int32_t) m.hp.audio_pad_id : (int32_t) m.hp.text_vocab;
        bc.h_pos_rope[c]  = 0;
        bc.h_pos_cache[c] = (int32_t) ((size_t) c * S);
        uint16_t * mc = &bc.h_mask[(size_t) c * S];
        mc[0] = M0;
        for (int p = 1; p < S; ++p) mc[p] = MINF;
    }
    // active columns: real next frame, logical position n_past, causal mask over [0, n_past].
    for (zonos2_slot * sp : active) {
        if (!sp || !sp->active || sp->done) continue;
        const int c = sp->index;
        for (int k = 0; k < W; ++k) bc.h_ids[(size_t) k * N + c] = sp->next_ids[k];
        bc.h_pos_rope[c]  = sp->n_past;
        bc.h_pos_cache[c] = (int32_t) ((size_t) c * S + sp->n_past);
        uint16_t * mc = &bc.h_mask[(size_t) c * S];
        for (int p = 0; p < S; ++p) mc[p] = (p <= sp->n_past) ? M0 : MINF;
    }

    if (prof_step) ta = sclk::now();

    // h_ids has stride N between codebooks; the first Wd columns of each are contiguous.
    for (int k = 0; k < W; ++k)
        ggml_backend_tensor_set(dg->ids[k], &bc.h_ids[(size_t) k * N], 0, (size_t) Wd * sizeof(int32_t));
    ggml_backend_tensor_set(dg->pos_rope,  bc.h_pos_rope.data(),  0, (size_t) Wd * sizeof(int32_t));
    ggml_backend_tensor_set(dg->pos_cache, bc.h_pos_cache.data(), 0, (size_t) Wd * sizeof(int32_t));
    ggml_backend_tensor_set(dg->mask,      bc.h_mask.data(),      0, (size_t) S * Wd * sizeof(uint16_t));
    if (prof_step) tb = sclk::now();

    if (prof_step) {
        auto sms = [](sclk::time_point a, sclk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        ggml_backend_graph_compute_async(m.backend, dg->gf);
        auto tc = sclk::now();
        ggml_backend_synchronize(m.backend);
        auto td = sclk::now();
        ggml_backend_tensor_get(dg->logits, bc.h_logits.data(), 0,
                                (size_t) Wd * av * ncb * sizeof(float));
        auto te = sclk::now();
        fprintf(stderr,
            "step[prof]: active=%d width=%d nodes=%d  fill=%.2fms set=%.2fms encode=%.2fms gpu=%.2fms get=%.2fms\n",
            eff, Wd, ggml_graph_n_nodes(dg->gf), sms(t_start, ta), sms(ta, tb), sms(tb, tc), sms(tc, td), sms(td, te));
    } else {
        ggml_backend_graph_compute(m.backend, dg->gf);
        ggml_backend_tensor_get(dg->logits, bc.h_logits.data(), 0,
                                (size_t) Wd * av * ncb * sizeof(float));
    }

    for (zonos2_slot * sp : active)
        if (sp && sp->active && !sp->done) sp->n_past += 1;
}

const float * zonos2_batch_slot_logits(const zonos2_batch_ctx & bc, int index) {
    return bc.h_logits.data() + (size_t) index * bc.av * bc.ncb;
}

bool zonos2_slot_sample(const zonos2_model & m, zonos2_slot & s, zonos2_sampler & smp,
                        const float * logits, std::vector<int32_t> & out_codes,
                        int max_frames, const zonos2_frame_cb & on_frame) {
    const int ncb = (int) m.hp.n_codebooks, av = (int) m.hp.audio_vocab, W = ncb + 1;
    std::vector<int> frame(ncb);
    for (int cb = 0; cb < ncb; ++cb) frame[cb] = smp.sample(&logits[(size_t) cb * av], cb);
    smp.accept(frame);
    for (int cb = 0; cb < ncb; ++cb) out_codes.push_back(frame[cb]);

    const int frame_idx = s.step;
    bool keep = true;
    if (on_frame) {
        const std::vector<int32_t> fc(frame.begin(), frame.end());
        keep = on_frame(frame_idx, fc.data(), ncb);
    }

    int max_eoa = -1;
    for (int cb = 0; cb < ncb; ++cb) if (frame[cb] == (int) m.hp.eoa_id) max_eoa = cb;
    if (s.eos_frame < 0 && max_eoa >= 0) { s.eos_frame = std::max(0, frame_idx - max_eoa); s.countdown = ncb + 1; }
    const bool finished = (s.countdown > 0 && --s.countdown == 0);

    s.next_ids.resize(W);
    for (int cb = 0; cb < ncb; ++cb) s.next_ids[cb] = frame[cb];
    s.next_ids[ncb] = (int32_t) m.hp.text_vocab;
    s.step += 1;
    if (!keep || finished || s.step >= max_frames) s.done = true;
    return keep;
}

// O(n) real-time path: one prefill, then single-token decodes against a persistent KV cache.
// Implemented as the batch runtime at n_slots=1 — the unified cache collapses to [hd, slot_cap,
// nkv] and the decode query is [hd,1,nh,1], so it routes through the same flash-attn vec kernel and
// CUDA-graph replay as a dedicated single-sequence path (verified bit-exact). This keeps the CLI
// and the server's continuous batcher on one code path.
static int generate_kv(const zonos2_model & m, const float * prompt_ids, int n0,
                       int max_frames, const zonos2_sampling & sp,
                       std::vector<int32_t> & out_codes, int & eos_frame,
                       const float * spk, int spk_pos, const zonos2_frame_cb & on_frame) {
    const int ncb = (int) m.hp.n_codebooks, av = (int) m.hp.audio_vocab;
    eos_frame = -1;

    zonos2_batch_ctx bc;
    if (!zonos2_batch_init(bc, m, /*n_slots=*/1, /*slot_cap_frames=*/n0 + max_frames + 8)) return 0;

    zonos2_sampler smp(sp, ncb, av);
    zonos2_slot slot;
    slot.index = 0; slot.active = true; slot.n_past = n0;

    std::vector<float> logits((size_t) av * ncb);
    if (!zonos2_batch_slot_prefill(bc, 0, prompt_ids, n0, spk, spk_pos, logits.data())) {
        zonos2_batch_free(bc); return 0;
    }

    const std::vector<zonos2_slot *> active = { &slot };
    int n_frames = 0, decode_steps = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        zonos2_slot_sample(m, slot, smp, logits.data(), out_codes, max_frames, on_frame);
        ++n_frames;
        if (slot.done) break;
        zonos2_batch_step(bc, active);
        memcpy(logits.data(), zonos2_batch_slot_logits(bc, 0), (size_t) av * ncb * sizeof(float));
        ++decode_steps;
    }
    eos_frame = slot.eos_frame;

    if (decode_steps > 0) {
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double frame_s = 512.0 / 44100.0; // one codebook frame at 44.1 kHz
        fprintf(stderr, "generate(kv): %d decode steps in %.3fs = %.1f frames/s (real-time = %.1f), RTF=%.2f\n",
                decode_steps, dt, decode_steps / dt, 1.0 / frame_s, (dt / decode_steps) / frame_s);
    }

    zonos2_batch_free(bc);
    return n_frames;
}

int zonos2_generate(const zonos2_model & m, const float * prompt_ids, int n0,
                    int max_frames, const zonos2_sampling & sp,
                    std::vector<int32_t> & out_codes, int & eos_frame, bool use_kv,
                    const float * spk, int spk_pos, std::vector<int32_t> * out_full_ids,
                    const zonos2_frame_cb & on_frame) {
    const size_t codes0 = out_codes.size();
    const int nf = use_kv
        ? generate_kv       (m, prompt_ids, n0, max_frames, sp, out_codes, eos_frame, spk, spk_pos, on_frame)
        : generate_recompute(m, prompt_ids, n0, max_frames, sp, out_codes, eos_frame, spk, spk_pos, on_frame);

    if (out_full_ids) {
        // Rebuild the exact input sequence the model saw: prompt rows, then each generated
        // frame's codes plus the text-pad column. Identical for both decode paths (delay shear
        // is applied inside the graph, so these pre-shear ids match the prompt convention).
        const int W = (int) m.hp.n_codebooks + 1, ncb = (int) m.hp.n_codebooks;
        out_full_ids->clear();
        out_full_ids->reserve((size_t) (n0 + nf) * W);
        for (int i = 0; i < n0 * W; ++i) out_full_ids->push_back((int32_t) lroundf(prompt_ids[i]));
        for (int f = 0; f < nf; ++f) {
            const int32_t * fr = &out_codes[codes0 + (size_t) f * ncb];
            for (int cb = 0; cb < ncb; ++cb) out_full_ids->push_back(fr[cb]);
            out_full_ids->push_back((int32_t) m.hp.text_vocab);
        }
    }
    return nf;
}
