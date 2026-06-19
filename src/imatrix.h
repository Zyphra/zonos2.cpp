// imatrix.h — per-expert importance matrix for K-quantization.
//
// One entry per quantized weight tensor: a [n_expert, n_in] grid of mean activation-squared
// (the column importance the K-quant quantizers weight their error by). For non-MoE tensors
// n_expert is 1. Written by `zonos2-perplexity --imatrix-out`, read by `quantize-cli --imatrix`.
// Header-only with no ggml dependency so quantize-cli (ggml-only) can include it too.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace imatrix {

struct entry {
    uint32_t           n_expert = 0;
    uint32_t           n_in     = 0;
    std::vector<float> data;          // size n_expert*n_in, importance[e*n_in + col]
};

static const char MAGIC[8] = { 'Z', '2', 'I', 'M', 'A', 'T', '0', '1' };

inline bool save(const std::string & path, const std::map<std::string, entry> & m) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "imatrix: cannot write %s\n", path.c_str()); return false; }
    fwrite(MAGIC, 1, 8, f);
    const uint32_t n = (uint32_t) m.size();
    fwrite(&n, sizeof n, 1, f);
    for (const auto & kv : m) {
        const uint32_t len = (uint32_t) kv.first.size();
        fwrite(&len, sizeof len, 1, f);
        fwrite(kv.first.data(), 1, len, f);
        fwrite(&kv.second.n_expert, sizeof(uint32_t), 1, f);
        fwrite(&kv.second.n_in,     sizeof(uint32_t), 1, f);
        fwrite(kv.second.data.data(), sizeof(float), kv.second.data.size(), f);
    }
    fclose(f);
    return true;
}

inline bool load(const std::string & path, std::map<std::string, entry> & m) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "imatrix: cannot read %s\n", path.c_str()); return false; }
    char magic[8];
    uint32_t n = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8) != 0) {
        fprintf(stderr, "imatrix: %s bad magic\n", path.c_str()); fclose(f); return false;
    }
    if (fread(&n, sizeof n, 1, f) != 1) { fclose(f); return false; }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof len, 1, f) != 1) { fclose(f); return false; }
        std::string name(len, '\0');
        if (fread(&name[0], 1, len, f) != len) { fclose(f); return false; }
        entry e;
        if (fread(&e.n_expert, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
        if (fread(&e.n_in,     sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
        e.data.resize((size_t) e.n_expert * e.n_in);
        if (fread(e.data.data(), sizeof(float), e.data.size(), f) != e.data.size()) {
            fprintf(stderr, "imatrix: %s truncated entry %s\n", path.c_str(), name.c_str());
            fclose(f); return false;
        }
        m.emplace(std::move(name), std::move(e));
    }
    fclose(f);
    return true;
}

} // namespace imatrix
