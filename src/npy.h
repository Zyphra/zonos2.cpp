// Minimal .npy (v1.0) reader/writer for float32 C-order arrays — validation only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npy {

inline bool save_f32(const std::string & path, const float * data, const std::vector<int64_t> & shape) {
    std::string shp;
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string(shape[i]);
        shp += ",";
        if (i + 1 < shape.size()) shp += " ";
    }
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shp + "), }";
    const size_t base = 10 + dict.size() + 1; // magic(6)+ver(2)+hlen(2) + trailing '\n'
    const size_t pad  = (64 - (base % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    const uint16_t hlen = (uint16_t) dict.size();

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    fwrite(magic, 1, 8, f);
    fwrite(&hlen, sizeof(uint16_t), 1, f);   // host is little-endian (x86)
    fwrite(dict.data(), 1, dict.size(), f);
    size_t n = 1;
    for (int64_t d : shape) n *= (size_t) d;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    return true;
}

// Reads a '<f4' C-order .npy into `data` (flattened) and returns its shape.
inline bool load_f32(const std::string & path, std::vector<float> & data, std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "npy: cannot open %s\n", path.c_str()); return false; }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || magic[0] != 0x93 || memcmp(magic + 1, "NUMPY", 5) != 0) {
        fprintf(stderr, "npy: bad magic %s\n", path.c_str()); fclose(f); return false;
    }
    uint16_t hlen = 0;
    fread(&hlen, sizeof(uint16_t), 1, f);
    std::string hdr(hlen, '\0');
    fread(&hdr[0], 1, hlen, f);
    if (hdr.find("'<f4'") == std::string::npos) {
        fprintf(stderr, "npy: %s is not <f4\n", path.c_str()); fclose(f); return false;
    }
    // parse shape tuple
    shape.clear();
    size_t sp = hdr.find("'shape':");
    size_t lp = hdr.find('(', sp);
    size_t rp = hdr.find(')', lp);
    const std::string nums = hdr.substr(lp + 1, rp - lp - 1);
    size_t i = 0;
    while (i < nums.size()) {
        while (i < nums.size() && !isdigit(nums[i])) ++i;
        if (i >= nums.size()) break;
        int64_t v = 0;
        while (i < nums.size() && isdigit(nums[i])) { v = v * 10 + (nums[i] - '0'); ++i; }
        shape.push_back(v);
    }
    size_t n = 1;
    for (int64_t d : shape) n *= (size_t) d;
    data.resize(n);
    const size_t got = fread(data.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "npy: short read %s (%zu/%zu)\n", path.c_str(), got, n); return false; }
    return true;
}

} // namespace npy
