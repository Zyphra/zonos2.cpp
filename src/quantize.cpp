// gguf -> gguf requantizer for zonos2 backbone files.
//
// Takes the lossless F16 backbone GGUF (e.g. the one published on HF) and writes a
// quantized copy, mirroring the policy in models/convert-zonos2-to-gguf.py:
//   - 1-D tensors (norms / biases / temp / eda-scale) are kept as-is (F32).
//   - the quant-sensitive set (output head, token+audio embeddings, MoE routers,
//     speaker projection) is kept one tier ABOVE the bulk quant (F16 when bulk is
//     Q8_0/legacy, else Q8_0).
//   - every other 2-D/3-D matrix takes the requested bulk quant, falling back to
//     F16 if its row length isn't a multiple of the quant block size.
// Quantization is ggml_quantize_chunk, so K-quants (Q4_K, Q5_K, Q6_K, ...) work
// here even though the pure-Python converter cannot emit them.
//
// Usage:  quantize-cli <in-f16.gguf> <out.gguf> <type> [--experts-only] [--down-tier-up]
//   type: q8_0 q4_0 q4_1 q5_0 q5_1 q2_k q3_k q4_k q5_k q6_k iq4_nl iq4_xs
//   --experts-only: low-bit <type> on the MoE expert stacks only; spine stays q8_0. Per the
//     KLD analysis the spine (attention/dense-FFN/router) flips routing under sub-q8 bits,
//     while the experts (the bulk of the weights) tolerate it -- best size/quality tradeoff.
//   --down-tier-up: put the expert down projection one K-tier above gate/up (q2_k down→q3_k
//     cut KLD ~19x for +0.24 GB). Off by default so plain --experts-only is uniform-bit.
#include "ggml.h"
#include "gguf.h"
#include "imatrix.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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
    { "iq3_xxs", GGML_TYPE_IQ3_XXS, 23 }, { "iq3_s", GGML_TYPE_IQ3_S, 26 },
};

// Tensors kept near-lossless even in a quantized file (see convert-zonos2-to-gguf.py):
// the output head, the token/audio embedding tables, and the per-layer MoE routers.
static bool is_high_precision(const std::string & n) {
    if (n == "output.weight" || n == "text_embd.weight") return true;
    // Speaker projection: its output REPLACES the embedding at spk_pos (not a residual add),
    // so it seeds the whole residual stream with no downstream averaging. The two-stage
    // lda->proj is a low-dim speaker code where direction is what SpkSim measures; even Q8_0's
    // blockwise absmax tanks SpkSim (~66->48) flat across all bulk tiers. ~8 MB to keep at F16.
    if (n == "spk_lda.weight" || n == "spk_proj.weight") return true;
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

// The expert down projection is the most quant-sensitive of the three stacks, so in
// --experts-only mode it takes one K-quant tier above gate/up (mirrors llama.cpp's _M mixes).
// No-op for non-K bulk types.
static ggml_type next_ktier(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K: return GGML_TYPE_Q3_K;
        case GGML_TYPE_Q3_K: return GGML_TYPE_Q4_K;
        case GGML_TYPE_Q4_K: return GGML_TYPE_Q5_K;
        case GGML_TYPE_Q5_K: return GGML_TYPE_Q6_K;
        default:             return t;
    }
}

// experts_only: apply the (low-bit) bulk quant ONLY to the expert stacks; keep the whole
// spine (attention, dense FFN, routers, head, embeddings) at q8_0 so router inputs stay
// clean and top-k expert selection does not flip. Sensitive set stays F16 as usual.
// down_tier_up (opt-in): the expert down projection — the most quant-sensitive of the three
// stacks — takes one K-quant tier above gate/up. Rescues 2-bit experts (q2_k down→q3_k cut
// KLD ~19x for +0.24 GB); marginal above q4_k. Left off by default so plain --experts-only
// stays a uniform-bit recipe.
// Quantize `src` ([ne0, ne1, ne2]) to `dst`. With a matching per-expert imatrix, each expert
// slice (ne1 rows of ne0) is quantized with its own importance vector; a degenerate (all-zero)
// slice falls back to uniform RTN so the K-quant solvers don't divide by zero. ggml_quantize_chunk
// indexes by `start` (= e*ne1*ne0), offsetting both src and dst internally.
static void quantize_with_imatrix(ggml_type tt, const float * src, void * dst,
                                  int64_t ne0, int64_t ne1, int64_t ne2, const imatrix::entry * im) {
    if (im && (int64_t) im->n_in == ne0 && (int64_t) im->n_expert == ne2) {
        for (int64_t e = 0; e < ne2; ++e) {
            const float * imv = &im->data[(size_t) e * ne0];
            double s = 0.0; for (int64_t c = 0; c < ne0; ++c) s += imv[c];
            ggml_quantize_chunk(tt, src, dst, e * ne1 * ne0, ne1, ne0, s > 0.0 ? imv : nullptr);
        }
    } else {
        ggml_quantize_chunk(tt, src, dst, 0, ne1 * ne2, ne0, nullptr);
    }
}

static ggml_type pick_qtype(const ggml_tensor * t, ggml_type bulk, bool experts_only, bool down_tier_up, bool spine_f16) {
    if (ggml_n_dims(t) == 1)        return t->type;                  // 1-D: keep (F32)
    const std::string n = ggml_get_name(t);
    if (experts_only && !is_expert(n)) {
        if (is_high_precision(n)) return GGML_TYPE_F16;
        if (spine_f16) return GGML_TYPE_F16;                          // spine fully F16 (A/B: q8 spine vs lossless)
        return (t->ne[0] % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; // spine stays q8_0
    }
    if (down_tier_up && is_expert(n) && n.find("ffn_down_exps") != std::string::npos) {
        const ggml_type up = next_ktier(bulk);               // down projection one K-tier up
        if (up != bulk && t->ne[0] % ggml_blck_size(up) == 0) return up;
    }
    if (is_high_precision(n)) return higher_tier(bulk, t->ne[0]);
    if (t->ne[0] % ggml_blck_size(bulk) != 0) return GGML_TYPE_F16;  // not block-aligned
    return bulk;
}

int main(int argc, char ** argv) {
    bool experts_only = false, down_tier_up = false, spine_f16 = false;
    const char * imat_path = nullptr;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--experts-only")) experts_only = true;
        else if (!strcmp(argv[i], "--down-tier-up")) down_tier_up = true;
        else if (!strcmp(argv[i], "--spine-f16"))    spine_f16 = true;
        else if (!strcmp(argv[i], "--imatrix") && i + 1 < argc) imat_path = argv[++i];
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 3) {
        fprintf(stderr, "usage: %s <in-f16.gguf> <out.gguf> <type> [--experts-only] [--down-tier-up] [--imatrix f.bin]\n"
                        "  --experts-only: apply <type> only to MoE expert stacks; spine stays q8_0\n"
                        "  --down-tier-up: put the expert down projection one K-tier up (rescues q2_k)\n"
                        "  --imatrix f.bin: per-expert importance matrix from `zonos2-perplexity --imatrix-out`\n"
                        "  type:", argv[0]);
        for (const auto & q : QTYPES) fprintf(stderr, " %s", q.name);
        fprintf(stderr, "\n");
        return 1;
    }
    const char * in_path = pos[0], * out_path = pos[1], * type_s = pos[2];

    std::map<std::string, imatrix::entry> imat;
    if (imat_path && !imatrix::load(imat_path, imat)) return 1;
    if (imat_path) fprintf(stderr, "quantize: loaded imatrix %s (%zu tensors)\n", imat_path, imat.size());

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
        const ggml_type tt = pick_qtype(t, bulk, experts_only, down_tier_up, spine_f16);
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
    int n_quant = 0, n_kept = 0, n_imat = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in, gguf_get_tensor_name(gin, i));
        const ggml_type tt = pick_qtype(t, bulk, experts_only, down_tier_up, spine_f16);
        const int64_t   ne0 = t->ne[0], n = ggml_nelements(t);

        ggml_tensor * d = ggml_new_tensor(ctx_out, tt, ggml_n_dims(t), t->ne);
        ggml_set_name(d, ggml_get_name(t));

        // Progress: print BEFORE the work so the slow IQ4/K-quant expert tensors are visible
        // while they run (stderr is unbuffered). '=' means kept as-is, '->' means requantized.
        fprintf(stderr, "[%4lld/%lld] %-32s %s %s %s\n", (long long) (i + 1), (long long) n_tensors,
                ggml_get_name(t), ggml_type_name(t->type), tt == t->type ? "==" : "->", ggml_type_name(tt));

        if (tt == t->type) {                            // kept (F32 1-D, or F16 sensitive)
            memcpy(d->data, t->data, ggml_nbytes(t));
            n_kept++;
        } else {                                        // F16/F32 source -> quantized
            f32.resize(n);
            if (t->type == GGML_TYPE_F16)      ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f32.data(), n);
            else if (t->type == GGML_TYPE_F32) memcpy(f32.data(), t->data, n * sizeof(float));
            else { fprintf(stderr, "quantize: unexpected source type for %s\n", ggml_get_name(t)); return 1; }
            auto it = imat.find(ggml_get_name(t));
            const imatrix::entry * im = (it != imat.end()) ? &it->second : nullptr;
            if (im) n_imat++;
            quantize_with_imatrix(tt, f32.data(), d->data, ne0, t->ne[1], t->ne[2], im);
            n_quant++;
        }
        gguf_add_tensor(gout, d);
        in_bytes  += ggml_nbytes(t);
        out_bytes += ggml_nbytes(d);
    }

    if (!gguf_write_to_file(gout, out_path, false)) { fprintf(stderr, "quantize: write failed\n"); return 1; }

    fprintf(stderr, "quantize: %s -> %s [%s%s%s]: %d tensors (%d quantized%s, %d kept), %.2f GB -> %.2f GB\n",
            in_path, out_path, type_s,
            experts_only ? (down_tier_up ? ", experts-only, down-tier-up" : ", experts-only") : "",
            imat_path ? ", imatrix" : "",
            (int) n_tensors, n_quant,
            n_imat ? (std::string(", ") + std::to_string(n_imat) + " w/ imatrix").c_str() : "",
            n_kept, in_bytes / 1e9, out_bytes / 1e9);

    gguf_free(gout); ggml_free(ctx_out); gguf_free(gin); ggml_free(ctx_in);
    return 0;
}
