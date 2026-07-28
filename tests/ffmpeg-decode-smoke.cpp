#include "spk-encoder.h"

#include <cmath>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ffmpeg-decode-smoke AUDIO\n");
        return 2;
    }

    spk_model model;
    model.sr = 24000;
    const std::vector<float> pcm = spk_decode_audio_file(model, argv[1]);
    if (pcm.size() != 12000) {
        fprintf(stderr, "expected 12000 decoded samples, got %zu\n", pcm.size());
        return 1;
    }
    for (const float sample : pcm) {
        if (!std::isfinite(sample)) {
            fprintf(stderr, "decoded output contains a non-finite sample\n");
            return 1;
        }
    }

    fprintf(stderr, "decoded %zu samples\n", pcm.size());
    return 0;
}
