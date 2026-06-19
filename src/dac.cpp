// DAC 44kHz decoder — ggml port of the Descript Audio Codec decode path (codes -> waveform).
// The last piece of the ZONOS2 pipeline that was still Python. Mirrors tokenizer/vocoder.py:
//   raw codes [H,9] -> shear_up(pad 1025) -> truncate at eos -> clamp<=1023
//   -> quantizer.from_codes (9 folded lookup tables) -> decoder -> tanh.
#include "dac.h"

#include "compat.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

ggml_tensor * dac_model::get(const std::string & n) const {
    auto it = t.find(n);
    if (it == t.end()) { fprintf(stderr, "dac: missing tensor '%s'\n", n.c_str()); return nullptr; }
    return it->second;
}

bool dac_load(dac_model & m, const char * path, bool use_gpu) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ &ctx_meta };
    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) { fprintf(stderr, "dac: cannot open gguf '%s'\n", path); return false; }

    auto u32 = [&](const char * k, int dflt) {
        int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? dflt : (int) gguf_get_val_u32(gguf, id);
    };
    m.n_codebooks   = u32("dac.n_codebooks", 9);
    m.codebook_size = u32("dac.codebook_size", 1024);
    m.latent_dim    = u32("dac.latent_dim", 1024);
    m.decoder_dim   = u32("dac.decoder_dim", 1536);
    m.sample_rate   = u32("dac.sample_rate", 44100);
    m.hop_length    = u32("dac.hop_length", 512);
    m.audio_pad_id  = u32("dac.audio_pad_id", 1025);
    {
        int64_t rr = gguf_find_key(gguf, "dac.decoder_rates");
        if (rr >= 0) {
            const int32_t * r = (const int32_t *) gguf_get_arr_data(gguf, rr);
            const int n = (int) gguf_get_arr_n(gguf, rr);
            m.decoder_rates.assign(r, r + n);
        }
    }

    ggml_backend_dev_t dev = nullptr;
    if (use_gpu) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev)    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    m.backend = ggml_backend_dev_init(dev, nullptr);
    m.buft    = ggml_backend_dev_buffer_type(dev);
    m.is_gpu  = (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU);
    m.ctx_w   = ctx_meta;
    m.buf_w   = ggml_backend_alloc_ctx_tensors(ctx_meta, m.backend);

    FILE * f = fopen(path, "rb");
    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<char> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        const char * name = ggml_get_name(t);
        const int64_t tid = gguf_find_tensor(gguf, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, tid);
        const size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        fseeko(f, off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "dac: read fail %s\n", name); fclose(f); return false; }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        m.t[name] = t;
    }
    fclose(f);
    fprintf(stderr, "dac: loaded %zu tensors on %s (latent=%d dec_dim=%d hop=%d sr=%d)\n",
            m.t.size(), m.is_gpu ? "GPU" : "CPU", m.latent_dim, m.decoder_dim, m.hop_length, m.sample_rate);
    gguf_free(gguf);
    return true;
}

void dac_free(dac_model & m) {
    if (m.buf_w)   ggml_backend_buffer_free(m.buf_w);
    if (m.ctx_w)   ggml_free(m.ctx_w);
    if (m.backend) ggml_backend_free(m.backend);
    m.buf_w = nullptr; m.ctx_w = nullptr; m.backend = nullptr; m.t.clear();
}

// ---- graph helpers; activations are TIME-major [T, C] (ne0=time), as ggml convs want ----

// Conv1d via im2col(F32) + mul_mat, full precision. w ne=[K,IC,OC], bias ne=[1,OC] (or null).
// x:[L,IC] -> [OL,OC].
static ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * bias,
                            ggml_tensor * x, int s, int p, int d) {
    ggml_tensor * ic = ggml_im2col(ctx, w, x, s, 0, p, 0, d, 0, false, GGML_TYPE_F32);  // [K*IC, OL, 1]
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, ic, ic->ne[0], ic->ne[1] * ic->ne[2]),   // src0 = im2col [K*IC, OL]
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));      // src1 = w      [K*IC, OC]
    if (bias) r = ggml_add(ctx, r, bias);   // [OL,OC] + [1,OC] broadcast over time
    return r;   // [OL, OC]
}

// ConvTranspose1d upsample by `stride` (kernel K=2*stride), then crop stride/2 each side to
// match PyTorch padding=ceil(stride/2) (exact for even strides). w ne=[K,OC,IC], bias [1,OC].
static ggml_tensor * convt1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * bias,
                             ggml_tensor * x, int stride) {
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);  // [(L+1)*stride, OC]
    const int crop = stride / 2;
    const int64_t outL = x->ne[0] * stride;
    ggml_tensor * v = ggml_view_2d(ctx, y, outL, y->ne[1], y->nb[1], (size_t) crop * y->nb[0]);
    v = ggml_cont(ctx, v);                  // [L*stride, OC]
    if (bias) v = ggml_add(ctx, v, bias);
    return v;
}

// snake(x) = x + (1/(alpha+1e-9)) * sin(alpha*x)^2 ; alpha,inv_alpha ne=[1,C] (per-channel).
static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * alpha, ggml_tensor * inv_alpha, ggml_tensor * x) {
    ggml_tensor * ax = ggml_mul(ctx, x, alpha);                  // [T,C] * [1,C] broadcast
    ggml_tensor * s2 = ggml_sqr(ctx, ggml_sin(ctx, ax));
    return ggml_add(ctx, x, ggml_mul(ctx, s2, inv_alpha));
}

static ggml_tensor * snake_named(ggml_context * ctx, const dac_model & m, ggml_tensor * x, const std::string & p) {
    return snake(ctx, m.get(p + ".alpha"), m.get(p + ".inv_alpha"), x);
}

// ResidualUnit: x + conv2(snake2(conv1(snake1(x)))); conv1 k=7 dilation d (pad 3d), conv2 k=1.
static ggml_tensor * res_unit(ggml_context * ctx, const dac_model & m, ggml_tensor * x,
                              const std::string & p, int dil) {
    ggml_tensor * y = snake_named(ctx, m, x, p + ".snake1");
    y = conv1d(ctx, m.get(p + ".conv1.weight"), m.get(p + ".conv1.bias"), y, 1, 3 * dil, dil);
    y = snake_named(ctx, m, y, p + ".snake2");
    y = conv1d(ctx, m.get(p + ".conv2.weight"), m.get(p + ".conv2.bias"), y, 1, 0, 1);
    return ggml_add(ctx, x, y);
}

static ggml_tensor * dec_block(ggml_context * ctx, const dac_model & m, ggml_tensor * x, int b, int stride) {
    const std::string p = "dec.b" + std::to_string(b);
    ggml_tensor * y = snake_named(ctx, m, x, p + ".snake");
    y = convt1d(ctx, m.get(p + ".convt.weight"), m.get(p + ".convt.bias"), y, stride);
    y = res_unit(ctx, m, y, p + ".res0", 1);
    y = res_unit(ctx, m, y, p + ".res1", 3);
    y = res_unit(ctx, m, y, p + ".res2", 9);
    return y;
}

// Total upsampling factor (samples emitted per latent frame) = product of decoder rates.
static int dac_samples_per_frame(const dac_model & m) {
    int spf = 1;
    for (int r : m.decoder_rates) spf *= r;
    return spf;
}

// shear_up + clamp for latent frames [t0,t1): idx[j][local] = codes[(t0+local)+j][j]
// (pad 1025 -> clamp 1023 when (t0+local)+j >= H). codes is row-major [H, W].
static void dac_build_idx(const dac_model & m, const int32_t * codes, int H, int W,
                          int t0, int t1, std::vector<std::vector<int32_t>> & idx) {
    const int L = t1 - t0;
    const int PAD = m.audio_pad_id;
    idx.assign(m.n_codebooks, std::vector<int32_t>(L));
    for (int j = 0; j < m.n_codebooks; ++j)
        for (int local = 0; local < L; ++local) {
            const int t = t0 + local;
            int v = (t + j < H) ? codes[(size_t) (t + j) * W + j] : PAD;   // shear_up
            if (v > 1023) v = 1023;                                        // clamp (incl. pad -> 1023)
            if (v < 0) v = 0;
            idx[j][local] = v;
        }
}

// Run the decoder over pre-sheared, clamped per-codebook indices (each length L);
// returns L * dac_samples_per_frame(m) mono f32 samples.
static bool dac_run_decoder(const dac_model & m, const std::vector<std::vector<int32_t>> & idx,
                            int L, std::vector<float> & audio) {
    ggml_context * ctx = ggml_init({ (size_t) 64 * 1024 * 1024, nullptr, /*no_alloc=*/ true });
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // from_codes: sum_i get_rows(table_i, codes_i) + total bias -> [latent, L] -> [L, latent]
    std::vector<ggml_tensor *> code_in(m.n_codebooks);
    ggml_tensor * z = nullptr;
    for (int i = 0; i < m.n_codebooks; ++i) {
        code_in[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
        ggml_set_input(code_in[i]);
        ggml_tensor * zi = ggml_get_rows(ctx, m.get("quant." + std::to_string(i) + ".table"), code_in[i]);  // [latent,L]
        z = z ? ggml_add(ctx, z, zi) : zi;
    }
    z = ggml_add(ctx, z, m.get("quant.bias"));               // + [latent] broadcast over time
    z = ggml_cont(ctx, ggml_transpose(ctx, z));              // [L, latent]

    // decoder
    ggml_tensor * h = conv1d(ctx, m.get("dec.conv_in.weight"), m.get("dec.conv_in.bias"), z, 1, 3, 1);  // [L,1536]
    for (int b = 0; b < (int) m.decoder_rates.size(); ++b)
        h = dec_block(ctx, m, h, b, m.decoder_rates[b]);
    h = snake_named(ctx, m, h, "dec.snake_out");
    h = conv1d(ctx, m.get("dec.conv_out.weight"), m.get("dec.conv_out.bias"), h, 1, 3, 1);  // [samples, 1]
    h = ggml_tanh(ctx, h);
    ggml_tensor * out = ggml_reshape_1d(ctx, h, h->ne[0]);   // [samples]
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(m.buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "dac: alloc failed\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return false; }
    for (int i = 0; i < m.n_codebooks; ++i)
        ggml_backend_tensor_set(code_in[i], idx[i].data(), 0, (size_t) L * sizeof(int32_t));

    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "dac: compute failed\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return false;
    }

    const int64_t ns = out->ne[0];
    audio.resize(ns);
    ggml_backend_tensor_get(out, audio.data(), 0, ns * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}

bool dac_decode(const dac_model & m, const int32_t * codes, int H, int W, int eos,
                std::vector<float> & audio) {
    if (W != m.n_codebooks) { fprintf(stderr, "dac: codes have %d codebooks, expected %d\n", W, m.n_codebooks); return false; }
    const int E = (eos >= 0) ? (eos < H ? eos : H) : H;   // rows after shear_up + truncate
    if (E <= 0) { fprintf(stderr, "dac: no frames to decode (eos=%d, H=%d)\n", eos, H); return false; }

    std::vector<std::vector<int32_t>> idx;
    dac_build_idx(m, codes, H, W, 0, E, idx);
    if (!dac_run_decoder(m, idx, E, audio)) return false;

    const int64_t ns = (int64_t) audio.size();
    double rms = 0, peak = 0;
    for (float v : audio) { rms += (double) v * v; peak = std::max(peak, (double) fabsf(v)); }
    fprintf(stderr, "dac: H=%d eos=%d -> %lld samples = %.2fs, rms=%.4f peak=%.4f\n",
            H, eos, (long long) ns, (double) ns / m.sample_rate, sqrt(rms / (double) ns), peak);
    return true;
}

bool dac_decode_window(const dac_model & m, const int32_t * codes, int H,
                       int f_lo, int f_hi, int Lc, int Rc, std::vector<float> & audio) {
    audio.clear();
    if (f_hi <= f_lo) return true;
    const int W = m.n_codebooks;
    const int spf = dac_samples_per_frame(m);
    const int a = std::max(0, f_lo - Lc);   // latent start (left context, clamped at 0)
    const int b = f_hi + Rc;                // latent end (right context); frames >= H are pad

    std::vector<std::vector<int32_t>> idx;
    dac_build_idx(m, codes, H, W, a, b, idx);
    std::vector<float> full;
    if (!dac_run_decoder(m, idx, b - a, full)) return false;

    const size_t s0 = (size_t) (f_lo - a) * spf;
    const size_t s1 = (size_t) (f_hi - a) * spf;
    if (s1 > full.size()) { fprintf(stderr, "dac: window crop out of range\n"); return false; }
    audio.assign(full.begin() + s0, full.begin() + s1);
    return true;
}

std::vector<uint8_t> dac_wav_bytes(const std::vector<float> & audio, int sample_rate) {
    const int64_t ns = (int64_t) audio.size();
    const uint32_t sr = (uint32_t) sample_rate, byte_rate = sr * 2, data_bytes = (uint32_t) (ns * 2);
    const uint32_t riff = 36 + data_bytes;
    std::vector<uint8_t> out;
    out.reserve(44 + (size_t) ns * 2);
    auto bytes = [&](const void * p, size_t n) { const uint8_t * b = (const uint8_t *) p; out.insert(out.end(), b, b + n); };
    auto w32 = [&](uint32_t v) { bytes(&v, 4); };
    auto w16 = [&](uint16_t v) { bytes(&v, 2); };
    bytes("RIFF", 4); w32(riff); bytes("WAVE", 4);
    bytes("fmt ", 4); w32(16); w16(1); w16(1); w32(sr); w32(byte_rate); w16(2); w16(16);
    bytes("data", 4); w32(data_bytes);
    for (int64_t i = 0; i < ns; ++i) {
        float v = audio[i]; if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t) lrintf(v * 32767.0f);
        w16((uint16_t) s);
    }
    return out;
}

bool dac_write_wav(const char * path, const std::vector<float> & audio, int sample_rate) {
    FILE * wf = fopen(path, "wb");
    if (!wf) { fprintf(stderr, "dac: cannot write %s\n", path); return false; }
    const std::vector<uint8_t> buf = dac_wav_bytes(audio, sample_rate);
    fwrite(buf.data(), 1, buf.size(), wf);
    fclose(wf);
    fprintf(stderr, "dac: wrote %s\n", path);
    return true;
}
