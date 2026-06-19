// gguf -> gguf requantizer for zonos2 backbone files.
//
// Takes the lossless F16 backbone GGUF (e.g. the one published on HF) and writes a
// quantized copy, mirroring the policy in models/convert-zonos2-to-gguf.py:
//   - 1-D tensors (norms / biases / temp / eda-scale) are kept as-is (F32).
//   - the quant-sensitive set (output head, token+audio embeddings, MoE routers)
//     is kept one tier ABOVE the bulk quant (F16 when bulk is Q8_0/legacy,
//     else Q8_0).
//   - every other 2-D/3-D matrix takes the requested bulk quant, falling back to
//     F16 if its row length isn't a multiple of the quant block size.
// Quantization is ggml_quantize_chunk, so K-quants (Q4_K, Q5_K, Q6_K, ...) work
// here even though the pure-Python converter cannot emit them.
//
// Usage:  quantize-cli <in-f16.gguf> <out.gguf> <type> [--experts-only]
//   type: q8_0 q4_0 q4_1 q5_0 q5_1 q2_k q3_k q4_k q5_k q6_k iq4_nl iq4_xs
//   --experts-only: low-bit <type> on the MoE expert stacks only; spine stays q8_0. Per the
//     KLD analysis the spine (attention/dense-FFN/router) flips routing under sub-q8 bits,
//     while the experts (the bulk of the weights) tolerate it -- best size/quality tradeoff.
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct qtype { const char * name; ggml_type type; uint32_t ftype; };

// ftype is the informational general.file_type KV (llama ftype space); the loader
// reads per-tensor types, so it is cosmetic but we set a sensible value anyway.
static const qtype QTYPES[] = {
    { "q4_0", GGML_TYPE_Q4_0, 2 },  { "q4_1", GGML_TYPE_Q4_1, 3 },
    { "q5_0", GGML_TYPE_Q5_0, 8 },  { "q5_1", GGML_TYPE_Q5_1, 9 },
    { "q8_0", GGML_TYPE_Q8_0, 7 },
    { "q2_k", GGML_TYPE_Q2_K, 10 }, { "q3_k", GGML_TYPE_Q3_K, 12 },
    { "q4_k", GGML_TYPE_Q4_K, 15 }, { "q5_k", GGML_TYPE_Q5_K, 17 },
    { "q6_k", GGML_TYPE_Q6_K, 18 },
    { "iq4_nl", GGML_TYPE_IQ4_NL, 25 }, { "iq4_xs", GGML_TYPE_IQ4_XS, 23 },
};

// Tensors kept near-lossless even in a quantized file (see convert-zonos2-to-gguf.py):
// the output head, the token/audio embedding tables, and the per-layer MoE routers.
static bool is_high_precision(const std::string & n) {
    if (n == "output.weight" || n == "text_embd.weight") return true;
    if (n.rfind("audio_embd.", 0) == 0) return true;
    if (n.rfind("blk.", 0) == 0 && (n.find(".router_down.") != std::string::npos || n.find(".router_mlp") != std::string::npos)) return true;
    return false;
}

// One tier above the bulk quant: F16 over Q8_0/legacy, else Q8_0 (or F16 if not
// 32-aligned). Matches the converter's "bump the sensitive set up one level".
static ggml_type higher_tier(ggml_type bulk, int64_t ne0) {
    switch (bulk) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
            return GGML_TYPE_F16;
        default:
            return (ne0 % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    }
}

// The MoE expert FFN stacks (ffn_{gate,up,down}_exps): the bulk of the parameters and,
// per KLD analysis, the only weights that tolerate sub-q8 bits without flipping routing.
static bool is_expert(const std::string & n) {
    return n.find("_exps") != std::string::npos;
}

// experts_only: apply the (low-bit) bulk quant ONLY to the expert stacks; keep the whole
// spine (attention, dense FFN, routers, head, embeddings) at q8_0 so router inputs stay
// clean and top-k expert selection does not flip. Sensitive set stays F16 as usual.
static ggml_type pick_qtype(const ggml_tensor * t, ggml_type bulk, bool experts_only) {
    if (ggml_n_dims(t) == 1)        return t->type;                  // 1-D: keep (F32)
    const std::string n = ggml_get_name(t);
    if (experts_only && !is_expert(n)) {
        if (is_high_precision(n)) return GGML_TYPE_F16;
        return (t->ne[0] % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; // spine stays q8_0
    }
    if (is_high_precision(n)) return higher_tier(bulk, t->ne[0]);
    if (t->ne[0] % ggml_blck_size(bulk) != 0) return GGML_TYPE_F16;  // not block-aligned
    return bulk;
}

int main(int argc, char ** argv) {
    bool experts_only = false;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--experts-only")) experts_only = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 3) {
        fprintf(stderr, "usage: %s <in-f16.gguf> <out.gguf> <type> [--experts-only]\n"
                        "  --experts-only: apply <type> only to MoE expert stacks; spine stays q8_0\n"
                        "  type:", argv[0]);
        for (const auto & q : QTYPES) fprintf(stderr, " %s", q.name);
        fprintf(stderr, "\n");
        return 1;
    }
    const char * in_path = pos[0], * out_path = pos[1], * type_s = pos[2];

    ggml_type bulk = GGML_TYPE_COUNT; uint32_t ftype = 1;
    for (const auto & q : QTYPES) if (!strcmp(type_s, q.name)) { bulk = q.type; ftype = q.ftype; }
    if (bulk == GGML_TYPE_COUNT) { fprintf(stderr, "quantize: unknown type '%s'\n", type_s); return 1; }
    if (ggml_quantize_requires_imatrix(bulk)) {
        fprintf(stderr, "quantize: %s needs an importance matrix, which this tool does not support\n", type_s);
        return 1;
    }

    ggml_context * ctx_in = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ false, /*ctx=*/ &ctx_in };
    gguf_context * gin = gguf_init_from_file(in_path, gp);
    if (!gin) { fprintf(stderr, "quantize: cannot open '%s'\n", in_path); return 1; }

    // Iterate the gguf tensor list (the authoritative on-file tensors), not the
    // ggml context walk -- with no_alloc=false the context also holds the backing
    // data object, which is not a real tensor.
    const int64_t n_tensors = gguf_get_n_tensors(gin);

    // Pass 1: decide target types and size the output context.
    size_t need = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in, gguf_get_tensor_name(gin, i));
        const ggml_type tt = pick_qtype(t, bulk, experts_only);
        need += ggml_row_size(tt, t->ne[0]) * (ggml_nelements(t) / t->ne[0]);
    }
    need += (size_t) (n_tensors + 1) * ggml_tensor_overhead() + (1u << 20);

    ggml_context * ctx_out = ggml_init({ need, nullptr, /*no_alloc=*/ false });
    gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gin);                              // copy all metadata KVs
    gguf_set_val_u32(gout, "general.file_type", ftype);

    // Pass 2: requantize tensor by tensor.
    std::vector<float> f32;
    size_t in_bytes = 0, out_bytes = 0;
    int n_quant = 0, n_kept = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in, gguf_get_tensor_name(gin, i));
        const ggml_type tt = pick_qtype(t, bulk, experts_only);
        const int64_t   ne0 = t->ne[0], n = ggml_nelements(t), nrows = n / ne0;

        ggml_tensor * d = ggml_new_tensor(ctx_out, tt, ggml_n_dims(t), t->ne);
        ggml_set_name(d, ggml_get_name(t));

        if (tt == t->type) {                            // kept (F32 1-D, or F16 sensitive)
            memcpy(d->data, t->data, ggml_nbytes(t));
            n_kept++;
        } else {                                        // F16/F32 source -> quantized
            f32.resize(n);
            if (t->type == GGML_TYPE_F16)      ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f32.data(), n);
            else if (t->type == GGML_TYPE_F32) memcpy(f32.data(), t->data, n * sizeof(float));
            else { fprintf(stderr, "quantize: unexpected source type for %s\n", ggml_get_name(t)); return 1; }
            ggml_quantize_chunk(tt, f32.data(), d->data, 0, nrows, ne0, nullptr);
            n_quant++;
        }
        gguf_add_tensor(gout, d);
        in_bytes  += ggml_nbytes(t);
        out_bytes += ggml_nbytes(d);
    }

    if (!gguf_write_to_file(gout, out_path, false)) { fprintf(stderr, "quantize: write failed\n"); return 1; }

    fprintf(stderr, "quantize: %s -> %s [%s%s]: %d tensors (%d quantized, %d kept), %.2f GB -> %.2f GB\n",
            in_path, out_path, type_s, experts_only ? ", experts-only" : "",
            (int) n_tensors, n_quant, n_kept, in_bytes / 1e9, out_bytes / 1e9);

    gguf_free(gout); ggml_free(ctx_out); gguf_free(gin); ggml_free(ctx_in);
    return 0;
}
