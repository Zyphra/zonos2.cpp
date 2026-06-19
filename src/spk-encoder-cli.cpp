// spk-encoder-cli — ECAPA-TDNN speaker encoder CLI. log-mel/24kHz audio -> 2048-d x-vector.
//
//   spk-encoder-cli <spk.gguf> --clone <audio>       <out.npy>   (any file; shells ffmpeg)
//   spk-encoder-cli <spk.gguf> --mel   <mel.npy>      <out.npy>   (log-mel [T,128])
//   spk-encoder-cli <spk.gguf> --wav   <wav24k.npy>   <out.npy>   (24kHz mono f32 npy)
//   spk-encoder-cli <spk.gguf> --raw   <wav24k.f32le> <out.npy>   (ffmpeg -ar 24000 -ac 1 -f f32le)
#include "compat.h"
#include "spk-encoder.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <spk.gguf> --clone <audio>       <out.npy>   (any file; shells ffmpeg)\n"
                        "       %s <spk.gguf> --mel   <mel.npy>      <out.npy>   (log-mel [T,128])\n"
                        "       %s <spk.gguf> --wav   <wav24k.npy>   <out.npy>   (24kHz mono f32 npy)\n"
                        "       %s <spk.gguf> --raw   <wav24k.f32le> <out.npy>   (ffmpeg -ar 24000 -ac 1 -f f32le)\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const std::string mode = argv[2];
    const char * in_path = argv[3];
    const char * out_path = argv[4];

    spk_model m;
    if (!spk_load(m, gguf_path)) return 1;

    std::vector<float> emb;
    if (mode == "--mel") {
        std::vector<float> md; std::vector<int64_t> sh;
        if (!npy::load_f32(in_path, md, sh) || sh.size() != 2) { fprintf(stderr, "bad mel npy\n"); spk_free(m); return 1; }
        const int64_t T = sh[0], MD = sh[1];   // [T, mel_dim]
        fprintf(stderr, "mel: [T=%lld, mel=%lld]\n", (long long) T, (long long) MD);
        emb = spk_embed_from_mel(m, md.data(), (int) T);
    } else if (mode == "--wav") {
        std::vector<float> wav; std::vector<int64_t> sh;
        if (!npy::load_f32(in_path, wav, sh)) { fprintf(stderr, "bad wav npy\n"); spk_free(m); return 1; }
        emb = spk_embed_from_pcm24k(m, wav.data(), (int) wav.size());
    } else if (mode == "--raw") {
        FILE * rf = fopen(in_path, "rb");   // headerless float32 LE mono @ sample_rate
        if (!rf) { fprintf(stderr, "cannot open raw %s\n", in_path); spk_free(m); return 1; }
        fseeko(rf, 0, SEEK_END); long sz = ftello(rf); fseeko(rf, 0, SEEK_SET);
        std::vector<float> wav(sz / sizeof(float));
        if (fread(wav.data(), 1, (size_t) sz, rf) != (size_t) sz) { fprintf(stderr, "raw read fail\n"); fclose(rf); spk_free(m); return 1; }
        fclose(rf);
        emb = spk_embed_from_pcm24k(m, wav.data(), (int) wav.size());
    } else if (mode == "--clone") {
        emb = spk_embed_from_file(m, in_path);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        spk_free(m); return 1;
    }

    if (emb.empty()) { fprintf(stderr, "spk: embedding failed\n"); spk_free(m); return 1; }
    double nrm = 0; for (float v : emb) nrm += (double) v * v;
    fprintf(stderr, "emb: [%d] norm=%.4f\n", (int) emb.size(), sqrt(nrm));
    npy::save_f32(out_path, emb.data(), { (int64_t) emb.size() });
    fprintf(stderr, "wrote %s\n", out_path);
    spk_free(m);
    return 0;
}
