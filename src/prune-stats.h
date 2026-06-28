// prune-stats.h — per-(layer,expert) importance for MoE expert pruning.
//
// One record per MoE layer: the routed hit count and the mean-squared-activation-norm
// (MSAN) of every expert. MSAN is the (1,0,2) score from the unified one-shot expert-
// pruning formulation (arXiv:2606.15716) — mean over routed tokens of ‖f‖₂², where f is the
// expert's down-projection input silu(gate(x))·up(x). The literature finds activation-norm
// scores (MAN/MSAN) clearly beat routing frequency alone for deciding which experts to drop.
//
// Written by `zonos2-perplexity --prune-stats`, read by `prune-cli`. Header-only with no ggml
// dependency so prune-cli (ggml-only, like quantize-cli) can include it too.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace prune_stats {

struct layer {
    int32_t              top_k = 0;   // experts selected per token (from the model hparams)
    std::vector<int64_t> cnt;         // [n_expert] routed (token,expert) hit count
    std::vector<float>   msan;        // [n_expert] mean ‖down-input‖² over routed tokens
};

static const char MAGIC[8] = { 'Z', '2', 'P', 'R', 'U', 'N', 'E', '1' };

inline bool save(const std::string & path, const std::map<int, layer> & m) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "prune-stats: cannot write %s\n", path.c_str()); return false; }
    fwrite(MAGIC, 1, 8, f);
    const uint32_t n = (uint32_t) m.size();
    fwrite(&n, sizeof n, 1, f);
    for (const auto & kv : m) {
        const int32_t  L  = kv.first;
        const uint32_t ne = (uint32_t) kv.second.cnt.size();
        fwrite(&L,  sizeof L,  1, f);
        fwrite(&ne, sizeof ne, 1, f);
        fwrite(&kv.second.top_k, sizeof(int32_t), 1, f);
        fwrite(kv.second.cnt.data(),  sizeof(int64_t), ne, f);
        fwrite(kv.second.msan.data(), sizeof(float),   ne, f);
    }
    fclose(f);
    return true;
}

inline bool load(const std::string & path, std::map<int, layer> & m) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "prune-stats: cannot read %s\n", path.c_str()); return false; }
    char magic[8];
    uint32_t n = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8) != 0) {
        fprintf(stderr, "prune-stats: %s bad magic\n", path.c_str()); fclose(f); return false;
    }
    if (fread(&n, sizeof n, 1, f) != 1) { fclose(f); return false; }
    for (uint32_t i = 0; i < n; ++i) {
        int32_t  L  = 0;
        uint32_t ne = 0;
        layer    e;
        if (fread(&L,  sizeof L,  1, f) != 1) { fclose(f); return false; }
        if (fread(&ne, sizeof ne, 1, f) != 1) { fclose(f); return false; }
        if (fread(&e.top_k, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
        e.cnt.resize(ne);
        e.msan.resize(ne);
        if (fread(e.cnt.data(),  sizeof(int64_t), ne, f) != ne ||
            fread(e.msan.data(), sizeof(float),   ne, f) != ne) {
            fprintf(stderr, "prune-stats: %s truncated layer %d\n", path.c_str(), L);
            fclose(f); return false;
        }
        m.emplace(L, std::move(e));
    }
    fclose(f);
    return true;
}

} // namespace prune_stats
