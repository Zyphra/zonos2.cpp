// dac-cli — decode ZONOS2 audio codes [T,9] to a 44.1 kHz WAV via the ggml DAC decoder.
// Mirrors decode-codes.py (reads <codes>.eos.npy automatically if present).
//
//   dac-cli <dac.gguf> <codes.npy> <out.wav> [--eos N] [--gpu|--cpu]
#include "dac.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dac.gguf> <codes.npy> <out.wav> [--eos N] [--gpu|--cpu]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * codes_path = argv[2];
    const char * out_wav = argv[3];
    bool use_gpu = false;
    int eos_override = -2;  // -2 = unset
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--cpu") use_gpu = false;
        else if (a == "--eos" && i + 1 < argc) eos_override = atoi(argv[++i]);
    }

    // read raw codes [H,9] (stored f32) -> int32
    std::vector<float> cf; std::vector<int64_t> sh;
    if (!npy::load_f32(codes_path, cf, sh) || sh.size() != 2) { fprintf(stderr, "bad codes npy\n"); return 1; }
    const int H = (int) sh[0], W = (int) sh[1];
    std::vector<int32_t> codes(cf.size());
    for (size_t i = 0; i < cf.size(); ++i) codes[i] = (int32_t) lrintf(cf[i]);

    // eos: --eos overrides; else <codes>.eos.npy; else full length
    int eos = -1;
    if (eos_override != -2) {
        eos = eos_override;
    } else {
        std::vector<float> ev; std::vector<int64_t> es;
        std::string ep = std::string(codes_path) + ".eos.npy";
        if (npy::load_f32(ep.c_str(), ev, es) && !ev.empty()) eos = (int) lrintf(ev[0]);
    }

    dac_model m;
    if (!dac_load(m, gguf_path, use_gpu)) return 1;
    std::vector<float> audio;
    if (!dac_decode(m, codes.data(), H, W, eos, audio)) { dac_free(m); return 1; }
    dac_write_wav(out_wav, audio, m.sample_rate);
    dac_free(m);
    return 0;
}
