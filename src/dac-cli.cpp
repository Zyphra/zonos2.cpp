// dac-cli — decode ZONOS2 audio codes [T,9] to a 44.1 kHz WAV via the ggml DAC decoder.
// Mirrors decode-codes.py (reads <codes>.eos.npy automatically if present).
//
//   dac-cli <dac.gguf> <codes.npy> <out.wav> [--eos N] [--gpu|--cpu]
#include "dac.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Self-check: decode `codes` once in full, then again block-by-block via dac_decode_window
// with Lc=Rc=ctx context, and report the max abs sample difference over the overlapping
// range. A seam-free window size yields ~0 diff (bit-identical streaming).
static int seam_check(int argc, char ** argv) {
    const char * gguf_path = argv[2];
    const char * codes_path = argv[3];
    bool use_gpu = false;
    int block = 40, ctx = 32;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--cpu") use_gpu = false;
        else if (a == "--block" && i + 1 < argc) block = atoi(argv[++i]);
        else if (a == "--ctx"   && i + 1 < argc) ctx = atoi(argv[++i]);
    }

    std::vector<float> cf; std::vector<int64_t> sh;
    if (!npy::load_f32(codes_path, cf, sh) || sh.size() != 2) { fprintf(stderr, "bad codes npy\n"); return 1; }
    const int H = (int) sh[0], W = (int) sh[1];
    std::vector<int32_t> codes(cf.size());
    for (size_t i = 0; i < cf.size(); ++i) codes[i] = (int32_t) lrintf(cf[i]);

    dac_model m;
    if (!dac_load(m, gguf_path, use_gpu)) return 1;

    int spf = 1; for (int r : m.decoder_rates) spf *= r;
    const int shear = m.n_codebooks - 1;

    std::vector<float> full;
    if (!dac_decode(m, codes.data(), H, W, /*eos=*/-1, full)) { dac_free(m); return 1; }

    // mid-stream emittable range: frames whose right context (ctx) + shear lookahead exist
    const int safe = H - ctx - shear;
    std::vector<float> tiled;
    int emitted = 0;
    while (emitted < safe) {
        const int f_hi = std::min(emitted + block, safe);
        std::vector<float> w;
        if (!dac_decode_window(m, codes.data(), H, emitted, f_hi, ctx, ctx, w)) { dac_free(m); return 1; }
        tiled.insert(tiled.end(), w.begin(), w.end());
        emitted = f_hi;
    }
    dac_free(m);

    const size_t n = std::min(tiled.size(), full.size());
    double maxd = 0; size_t at = 0;
    for (size_t i = 0; i < n; ++i) { double d = std::fabs((double) tiled[i] - (double) full[i]); if (d > maxd) { maxd = d; at = i; } }
    fprintf(stderr, "seam: H=%d block=%d ctx=%d shear=%d -> compared %zu samples (%d frames), max abs diff=%.3e at %zu (frame %zu)\n",
            H, block, ctx, shear, n, (int) (n / spf), maxd, at, at / spf);
    fprintf(stderr, "seam: %s\n", maxd < 1e-6 ? "PASS (bit-identical)" : (maxd < 1e-3 ? "OK (sub-millisample)" : "FAIL (audible seam)"));
    return maxd < 1e-3 ? 0 : 2;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--seam") return seam_check(argc, argv);
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dac.gguf> <codes.npy> <out.wav> [--eos N] [--gpu|--cpu]\n"
                        "       %s --seam <dac.gguf> <codes.npy> [--block N] [--ctx N] [--gpu|--cpu]\n", argv[0], argv[0]);
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
