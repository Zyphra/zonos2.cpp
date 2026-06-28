// gguf -> gguf MoE expert pruner for zonos2 backbone files.
//
// Drops the least-important experts from each MoE layer and rewrites a smaller GGUF. Expert
// importance is the MSAN score (mean ‖down-input‖² over routed tokens) collected by
// `zonos2-perplexity --prune-stats`; per the one-shot MoE-pruning literature (arXiv:2606.15716)
// activation-norm scores rank experts far better than routing frequency alone.
//
// For each MoE layer we keep the top experts by MSAN and gather the corresponding slices of:
//   - ffn_{gate,up,down}_exps.weight  (3-D, expert = ne2)
//   - router_mlp4.weight              (2-D [router_dim, n_expert], expert = ne1)
//   - router_bias                     (1-D [n_expert], expert = ne0)
// The router softmax then runs over the surviving logits, renormalizing the gate weights for
// free (see build_moe, which now reads the expert count per layer from the stack itself).
// Slicing is a pure per-expert memcpy gather — experts are whole contiguous blocks along their
// axis — so it works on an F16 *or* an already-quantized backbone, and composes with
// quantize-cli in either order.
//
// Usage:  prune-cli <in.gguf> <out.gguf> --stats <stats.bin> (--keep N | --drop-below-hits H)
//   --keep N           keep the N highest-MSAN experts in every MoE layer (uniform budget)
//   --drop-below-hits H drop experts routed fewer than H times over the corpus (conservative)
// A layer never drops below its top-k (top-k selection needs at least k experts); when both the
// budget and the floor apply, the floor wins. expert_used_count_per_layer is lowered to match
// if a layer ends up with fewer experts than its old top-k.
#include "ggml.h"
#include "gguf.h"
#include "prune-stats.h"
#include "prune-policy.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// blk.<L>. prefix -> L, else -1.
static int layer_of(const std::string & n) {
    if (n.rfind("blk.", 0) != 0) return -1;
    const size_t p = 4, e = n.find('.', p);
    if (e == std::string::npos) return -1;
    return atoi(n.substr(p, e - p).c_str());
}

// Which tensor axis indexes the experts (and thus gets sliced), or -1 if this tensor is not
// per-expert and should be copied verbatim.
static int expert_axis(const std::string & n) {
    if (n.find("_exps") != std::string::npos)             return 2; // ffn_{gate,up,down}_exps
    if (n.find(".router_mlp4.weight") != std::string::npos) return 1; // [router_dim, n_expert]
    if (n.find(".router_bias") != std::string::npos)      return 0; // [n_expert]
    return -1;
}

int main(int argc, char ** argv) {
    std::string stats_path;
    int keep_n = -1, drop_below_hits = -1;
    double mass_eps = -1.0;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--stats")           && i + 1 < argc) stats_path      = argv[++i];
        else if (!strcmp(argv[i], "--keep")            && i + 1 < argc) keep_n          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mass-eps")        && i + 1 < argc) mass_eps        = atof(argv[++i]);
        else if (!strcmp(argv[i], "--drop-below-hits") && i + 1 < argc) drop_below_hits = atoi(argv[++i]);
        else pos.push_back(argv[i]);
    }
    if (pos.size() != 2 || stats_path.empty() || (keep_n < 0 && drop_below_hits < 0 && mass_eps < 0.0)) {
        fprintf(stderr,
            "usage: %s <in.gguf> <out.gguf> --stats <stats.bin> (--keep N | --mass-eps E | --drop-below-hits H)\n"
            "  --stats f.bin       per-(layer,expert) MSAN from `zonos2-perplexity --prune-stats`\n"
            "  --keep N            keep the N highest-MSAN experts per MoE layer (uniform)\n"
            "  --mass-eps E        keep experts with MSAN share >= E per layer (non-uniform: prunes\n"
            "                      peaky deep layers hard, protects flat early layers)\n"
            "  --drop-below-hits H drop experts routed fewer than H times (conservative)\n"
            "  (a layer never drops below its top-k)\n", argv[0]);
        return 1;
    }
    const char * in_path = pos[0], * out_path = pos[1];

    std::map<int, prune_stats::layer> stats;
    if (!prune_stats::load(stats_path, stats)) return 1;

    // Decide the keep-set (ascending original expert ids) for each MoE layer — shared with the
    // runtime --prune-mask path so a mask sweep and this prune produce identical models.
    const std::map<int, std::vector<int>> keep =
        prune_policy::select(stats, { keep_n, drop_below_hits, mass_eps });
    if (keep.empty()) {
        fprintf(stderr, "prune: budget drops no experts; nothing to do\n");
        return 1;
    }

    ggml_context * ctx_in = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ false, /*ctx=*/ &ctx_in };
    gguf_context * gin = gguf_init_from_file(in_path, gp);
    if (!gin) { fprintf(stderr, "prune: cannot open '%s'\n", in_path); return 1; }
    const int64_t n_tensors = gguf_get_n_tensors(gin);

    // Pass 1: size the output pool (kept tensors unchanged, pruned tensors shrink to keep slabs).
    size_t need = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in, gguf_get_tensor_name(gin, i));
        const std::string n = ggml_get_name(t);
        const int L = layer_of(n), ax = expert_axis(n);
        auto it = (L >= 0) ? keep.find(L) : keep.end();
        if (ax >= 0 && it != keep.end()) need += it->second.size() * t->nb[ax];
        else                             need += ggml_nbytes(t);
    }
    need += (size_t) (n_tensors + 1) * ggml_tensor_overhead() + (1u << 20);

    ggml_context * ctx_out = ggml_init({ need, nullptr, /*no_alloc=*/ false });
    gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gin);                                          // copy all metadata KVs

    // Pass 2: copy or gather tensor by tensor.
    size_t in_bytes = 0, out_bytes = 0;
    int max_experts = 0, n_pruned = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in, gguf_get_tensor_name(gin, i));
        const std::string n = ggml_get_name(t);
        const int L = layer_of(n), ax = expert_axis(n);
        auto it = (L >= 0) ? keep.find(L) : keep.end();

        ggml_tensor * d;
        if (ax >= 0 && it != keep.end()) {                          // gather the kept experts
            const std::vector<int> & ks = it->second;
            if ((int64_t) stats[L].cnt.size() != t->ne[ax]) {
                fprintf(stderr, "prune: %s expert axis %lld != stats n_expert %zu\n",
                        n.c_str(), (long long) t->ne[ax], stats[L].cnt.size());
                return 1;
            }
            int64_t ne[GGML_MAX_DIMS];
            for (int k = 0; k < GGML_MAX_DIMS; ++k) ne[k] = t->ne[k];
            ne[ax] = (int64_t) ks.size();
            d = ggml_new_tensor(ctx_out, t->type, ggml_n_dims(t), ne);
            ggml_set_name(d, n.c_str());
            const size_t stride = t->nb[ax];                        // one expert = one contiguous slab
            for (size_t e = 0; e < ks.size(); ++e)
                memcpy((char *) d->data + e * stride, (char *) t->data + (size_t) ks[e] * stride, stride);
            if (n.find("ffn_gate_exps") != std::string::npos) { ++n_pruned;
                fprintf(stderr, "  blk.%d  %lld -> %zu experts\n", L, (long long) t->ne[ax], ks.size()); }
        } else {                                                    // verbatim
            d = ggml_new_tensor(ctx_out, t->type, ggml_n_dims(t), t->ne);
            ggml_set_name(d, n.c_str());
            memcpy(d->data, t->data, ggml_nbytes(t));
        }
        if (n.find("_exps") != std::string::npos) max_experts = std::max(max_experts, (int) d->ne[2]);
        gguf_add_tensor(gout, d);
        in_bytes  += ggml_nbytes(t);
        out_bytes += ggml_nbytes(d);
    }

    // Update expert metadata: lower per-layer top-k if a layer now has fewer experts than its
    // old top-k, and set expert_count to the new global maximum.
    const int64_t euid = gguf_find_key(gin, "zonos2.expert_used_count_per_layer");
    if (euid >= 0 && gguf_get_arr_type(gin, euid) == GGUF_TYPE_INT32) {
        const size_t nl = gguf_get_arr_n(gin, euid);
        const int32_t * old = (const int32_t *) gguf_get_arr_data(gin, euid);
        std::vector<int32_t> eu(old, old + nl);
        for (const auto & kv : keep)
            if (kv.first >= 0 && (size_t) kv.first < nl)
                eu[kv.first] = std::min(eu[kv.first], (int32_t) kv.second.size());
        gguf_set_arr_data(gout, "zonos2.expert_used_count_per_layer", GGUF_TYPE_INT32, eu.data(), eu.size());
    }
    if (max_experts > 0) gguf_set_val_u32(gout, "zonos2.expert_count", (uint32_t) max_experts);

    if (!gguf_write_to_file(gout, out_path, false)) { fprintf(stderr, "prune: write failed\n"); return 1; }

    fprintf(stderr, "prune: %s -> %s : pruned %d MoE layers, max %d experts/layer, %.2f GB -> %.2f GB\n",
            in_path, out_path, n_pruned, max_experts, in_bytes / 1e9, out_bytes / 1e9);

    gguf_free(gout); ggml_free(ctx_out); gguf_free(gin); ggml_free(ctx_in);
    return 0;
}
