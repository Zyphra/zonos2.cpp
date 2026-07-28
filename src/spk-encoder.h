// spk-encoder — ggml port of the ECAPA-TDNN speaker encoder
// (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B, a ~6M-param ECAPA-TDNN).
// log-mel [128,T] -> 2048-d x-vector. Optional in-graph mel frontend from a 24 kHz
// mono waveform (STFT via DFT matmul + slaney mel + log). CPU backend.
//
// Shared by spk-encoder-cli and zonos2-server (in-process voice cloning).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

struct spk_model {
    ggml_context * ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor *> t;
    // hparams
    int mel_dim = 128, enc_dim = 2048;
    int n_fft = 1024, hop = 256, win = 1024, sr = 24000;
    int block0_k = 5;
    int res2net_scale = 8;
    std::vector<int> res_dils = {2, 3, 4};  // SE-Res2Net block dilations
    float fmin = 0.0f, fmax = 12000.0f;

    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "spk: missing tensor '%s'\n", n.c_str()); return nullptr; }
        return it->second;
    }
};

// Load the speaker-encoder GGUF onto the CPU backend. Returns false on error.
bool spk_load(spk_model & m, const char * path);
void spk_free(spk_model & m);

// Embed from a precomputed log-mel spectrogram, row-major [T, mel_dim].
// Returns an enc_dim (2048) x-vector, or an empty vector on failure.
std::vector<float> spk_embed_from_mel(const spk_model & m, const float * mel, int T);

// Embed from a 24 kHz mono f32 waveform (builds the STFT/mel frontend in-graph).
std::vector<float> spk_embed_from_pcm24k(const spk_model & m, const float * wav, int n);

// Decode any audio file to sample_rate (24 kHz) mono f32 via ffmpeg. Returns empty on failure.
// Resolves ffmpeg from ZONOS2_FFMPEG, next to the executable, or PATH. Useful to keep the
// decoded waveform (e.g. for a preview) alongside the embedding from spk_embed_from_pcm24k.
std::vector<float> spk_decode_audio_file(const spk_model & m, const char * path);

// Embed from any audio file: spk_decode_audio_file() then spk_embed_from_pcm24k().
std::vector<float> spk_embed_from_file(const spk_model & m, const char * path);
