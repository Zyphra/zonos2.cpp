// DAC 44kHz decoder (codes -> waveform) — standalone ggml port of the Descript Audio Codec
// decode path. Shared by dac-cli and zonos2-cli (--dac) so the full TTS pipeline is one binary.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct dac_model {
    ggml_context * ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    bool is_gpu = false;
    std::map<std::string, ggml_tensor *> t;
    // hparams
    int n_codebooks = 9, codebook_size = 1024, latent_dim = 1024, decoder_dim = 1536;
    int sample_rate = 44100, hop_length = 512, audio_pad_id = 1025;
    std::vector<int> decoder_rates = {8, 8, 4, 2};

    ggml_tensor * get(const std::string & n) const;
};

// Load the DAC GGUF and allocate weights on GPU (if use_gpu and available) or CPU.
bool dac_load(dac_model & m, const char * path, bool use_gpu);
void dac_free(dac_model & m);

// Decode RAW (delayed) codes [H, W=n_codebooks], row-major, into a mono f32 waveform.
// Mirrors tokenizer/vocoder.py: shear_up(pad) -> truncate at eos (eos<0 = full)
// -> clamp<=1023 -> quantizer.from_codes -> decoder -> tanh.
bool dac_decode(const dac_model & m, const int32_t * codes, int H, int W, int eos,
                std::vector<float> & audio);

// Streaming-friendly windowed decode. Decode output frames [f_lo, f_hi) from the raw
// codes [H, n_codebooks], building the latent over the extended range
// [f_lo-Lc, f_hi+Rc) so conv boundaries get enough context, then cropping `audio` to the
// central samples for [f_lo, f_hi). With Lc/Rc >= the decoder's receptive field, the
// returned samples are bit-identical to the corresponding slice of a single dac_decode().
// The caller must have generated codes up to (f_hi+Rc-1)+(n_codebooks-1) (the shear
// lookahead) for the window to be seam-free; frames past H are treated as pad.
bool dac_decode_window(const dac_model & m, const int32_t * codes, int H,
                       int f_lo, int f_hi, int Lc, int Rc, std::vector<float> & audio);

// Encode a mono f32 waveform as a 16-bit PCM mono WAV byte buffer (RIFF/WAVE).
std::vector<uint8_t> dac_wav_bytes(const std::vector<float> & audio, int sample_rate);

// Write a 16-bit PCM mono WAV file.
bool dac_write_wav(const char * path, const std::vector<float> & audio, int sample_rate);
