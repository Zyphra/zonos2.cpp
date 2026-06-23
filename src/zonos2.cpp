#include "zonos2.h"

#include "compat.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

bool zonos2_model_load(zonos2_model & m, const char * path, bool use_gpu) {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ &ctx_meta };
    struct gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) {
        fprintf(stderr, "zonos2: failed to open gguf '%s'\n", path);
        return false;
    }

    bool ok = true;
    auto & hp = m.hp;

#define U32(key, dst) do { int64_t _id = gguf_find_key(gguf, key); \
    if (_id < 0) { fprintf(stderr, "zonos2: missing kv '%s'\n", key); ok = false; } \
    else dst = gguf_get_val_u32(gguf, _id); } while (0)
#define F32(key, dst) do { int64_t _id = gguf_find_key(gguf, key); \
    if (_id < 0) { fprintf(stderr, "zonos2: missing kv '%s'\n", key); ok = false; } \
    else dst = gguf_get_val_f32(gguf, _id); } while (0)

    U32("zonos2.block_count",                       hp.n_layer);
    U32("zonos2.embedding_length",                  hp.n_embd);
    U32("zonos2.head_dim",                          hp.head_dim);
    U32("zonos2.attention.head_count",              hp.n_head);
    U32("zonos2.attention.head_count_kv",           hp.n_head_kv);
    U32("zonos2.feed_forward_length",               hp.n_ff);
    U32("zonos2.expert_count",                      hp.n_expert);
    U32("zonos2.context_length",                    hp.n_ctx_train);
    F32("zonos2.rope.freq_base",                    hp.rope_freq_base);
    F32("zonos2.attention.layer_norm_rms_epsilon",  hp.rms_eps);
    F32("zonos2.attention.qk_norm_epsilon",         hp.qk_norm_eps);
    F32("zonos2.logit_softcap",                     hp.logit_softcap);
    U32("zonos2.n_codebooks",                       hp.n_codebooks);
    U32("zonos2.codebook_size",                     hp.codebook_size);
    U32("zonos2.audio_vocab",                       hp.audio_vocab);
    U32("zonos2.eoa_id",                            hp.eoa_id);
    U32("zonos2.audio_pad_id",                      hp.audio_pad_id);
    U32("zonos2.text_vocab",                        hp.text_vocab);
    U32("zonos2.speaker.embedding_dim",             hp.spk_dim);
    U32("zonos2.speaker.lda_dim",                   hp.spk_lda_dim);
#undef U32
#undef F32

    {
        int64_t id = gguf_find_key(gguf, "zonos2.expert_used_count_per_layer");
        if (id < 0 || gguf_get_arr_type(gguf, id) != GGUF_TYPE_INT32) {
            fprintf(stderr, "zonos2: missing/!int32 expert_used_count_per_layer\n");
            ok = false;
        } else {
            const size_t n = gguf_get_arr_n(gguf, id);
            const int32_t * d = (const int32_t *) gguf_get_arr_data(gguf, id);
            hp.expert_used.assign(d, d + n);
            if (n != hp.n_layer) {
                fprintf(stderr, "zonos2: expert_used n=%zu != n_layer=%u\n", n, hp.n_layer);
                ok = false;
            }
        }
    }

    if (!ok) { gguf_free(gguf); ggml_free(ctx_meta); return false; }

    // --- backend ---
    ggml_backend_dev_t dev = nullptr;
    if (use_gpu) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev)    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev)    { fprintf(stderr, "zonos2: no backend device\n"); gguf_free(gguf); ggml_free(ctx_meta); return false; }
    // Metal: encode the decode graph across 4 command buffers (default is 1). Single-token decode
    // is a deep chain of ~1675 tiny kernels; with one command buffer the ~3 ms CPU encode runs
    // almost serially before the GPU, but spreading it over 4 buffers lets the GPU start streaming
    // through them as they're enqueued — ~10% faster decode (measured) at no quality cost. prefill
    // (big GEMMs) is unaffected. Overridable: respect an explicit GGML_METAL_NCB from the caller.
    if (use_gpu) setenv("GGML_METAL_NCB", "4", /*overwrite=*/0);
    m.backend = ggml_backend_dev_init(dev, nullptr);
    m.buft    = ggml_backend_dev_buffer_type(dev);
    m.is_gpu  = (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU);
    fprintf(stderr, "zonos2: backend = %s | %s\n",
            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    // --- allocate all weight tensors (declared in ctx_meta) on the backend ---
    m.ctx_w = ctx_meta;
    m.buf_w = ggml_backend_alloc_ctx_tensors(ctx_meta, m.backend);
    if (!m.buf_w) {
        fprintf(stderr, "zonos2: failed to allocate weight buffer\n");
        gguf_free(gguf); return false;
    }

    // --- stream tensor data from the file into the backend ---
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "zonos2: cannot reopen '%s'\n", path); gguf_free(gguf); return false; }
    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<char> tmp;
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        const char * name = ggml_get_name(t);
        const int64_t tid = gguf_find_tensor(gguf, name);
        if (tid < 0) { fprintf(stderr, "zonos2: '%s' missing in gguf index\n", name); ok = false; break; }
        const size_t off = data_off + gguf_get_tensor_offset(gguf, tid);
        const size_t nb  = ggml_nbytes(t);
        tmp.resize(nb);
        if (fseeko(f, off, SEEK_SET) != 0 || fread(tmp.data(), 1, nb, f) != nb) {
            fprintf(stderr, "zonos2: read failed for '%s'\n", name); ok = false; break;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        m.tensors[name] = t;
        total += nb;
    }
    fclose(f);
    gguf_free(gguf);
    if (!ok) return false;
    fprintf(stderr, "zonos2: loaded %zu tensors, %.2f GiB\n",
            m.tensors.size(), total / (1024.0 * 1024.0 * 1024.0));

    // --- map tensors to struct fields ---
    auto T = [&](const std::string & n) -> ggml_tensor * {
        auto it = m.tensors.find(n);
        return it == m.tensors.end() ? nullptr : it->second;
    };
    auto REQ = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = T(n);
        if (!t) { fprintf(stderr, "zonos2: required tensor '%s' not found\n", n.c_str()); ok = false; }
        return t;
    };

    m.audio_embd.resize(hp.n_codebooks);
    for (uint32_t i = 0; i < hp.n_codebooks; ++i) {
        m.audio_embd[i] = REQ("audio_embd." + std::to_string(i) + ".weight");
    }
    m.text_embd   = REQ("text_embd.weight");
    m.spk_lda_w   = REQ("spk_lda.weight");
    m.spk_lda_b   = REQ("spk_lda.bias");
    m.spk_proj_w  = REQ("spk_proj.weight");
    m.spk_proj_b  = REQ("spk_proj.bias");
    m.output_norm = REQ("output_norm.weight");
    m.output      = REQ("output.weight");

    m.layers.resize(hp.n_layer);
    for (uint32_t L = 0; L < hp.n_layer; ++L) {
        zonos2_layer & ly = m.layers[L];
        const std::string b = "blk." + std::to_string(L) + ".";
        ly.attn_norm = REQ(b + "attn_norm.weight");
        ly.ffn_norm  = REQ(b + "ffn_norm.weight");
        ly.wq        = REQ(b + "attn_q.weight");
        ly.wk        = REQ(b + "attn_k.weight");
        ly.wv        = REQ(b + "attn_v.weight");
        ly.wo        = REQ(b + "attn_output.weight");
        ly.attn_temp = REQ(b + "attn_temp");
        ly.gater     = REQ(b + "attn_gate.weight");

        ly.top_k  = hp.expert_used[L];
        ly.is_moe = ly.top_k > 0;
        if (!ly.is_moe) {
            ly.ffn_up   = REQ(b + "ffn_up.weight");
            ly.ffn_gate = REQ(b + "ffn_gate.weight");
            ly.ffn_down = REQ(b + "ffn_down.weight");
        } else {
            ly.ffn_gate_exps = REQ(b + "ffn_gate_exps.weight");
            ly.ffn_up_exps   = REQ(b + "ffn_up_exps.weight");
            ly.ffn_down_exps = REQ(b + "ffn_down_exps.weight");
            ly.router_down   = REQ(b + "router_down.weight");
            ly.router_down_b = REQ(b + "router_down.bias");
            ly.router_mlp0   = REQ(b + "router_mlp0.weight");
            ly.router_mlp0_b = REQ(b + "router_mlp0.bias");
            ly.router_mlp2   = REQ(b + "router_mlp2.weight");
            ly.router_mlp2_b = REQ(b + "router_mlp2.bias");
            ly.router_mlp4   = REQ(b + "router_mlp4.weight");
            ly.router_norm   = REQ(b + "router_norm.weight");
            ly.router_bias   = REQ(b + "router_bias");
            ly.router_eda_scale = T(b + "router_eda_scale"); // optional (first MoE layer has none)
        }
    }

    return ok;
}

void zonos2_model_free(zonos2_model & m) {
    if (m.buf_w)   ggml_backend_buffer_free(m.buf_w);
    if (m.ctx_w)   ggml_free(m.ctx_w);
    if (m.backend) ggml_backend_free(m.backend);
    m = zonos2_model();
}

int zonos2_router_dim(const zonos2_model & m) {
    for (const auto & ly : m.layers) {
        if (ly.is_moe && ly.router_down) return (int) ly.router_down->ne[1];
    }
    return 0;
}
