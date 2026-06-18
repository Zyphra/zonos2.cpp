// spk-encoder-cli — standalone ggml port of the ECAPA-TDNN speaker encoder
// (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B, actually a ~6M-param ECAPA-TDNN).
// log-mel [128,T] -> 2048-d x-vector. Optional in-graph mel frontend from a 24 kHz
// mono waveform (STFT via DFT matmul + slaney mel + log). CPU backend.
//
//   spk-encoder-cli <spk.gguf> --mel <mel.npy> <out_emb.npy>
//   spk-encoder-cli <spk.gguf> --wav <wav24k.npy> <out_emb.npy>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "npy.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static bool spk_load(spk_model & m, const char * path) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ &ctx_meta };
    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) { fprintf(stderr, "spk: cannot open gguf '%s'\n", path); return false; }

    auto u32 = [&](const char * k, int dflt) {
        int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? dflt : (int) gguf_get_val_u32(gguf, id);
    };
    auto f32 = [&](const char * k, float dflt) {
        int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? dflt : gguf_get_val_f32(gguf, id);
    };
    m.mel_dim = u32("ecapa-tdnn.mel_dim", 128);
    m.enc_dim = u32("ecapa-tdnn.enc_dim", 2048);
    m.n_fft   = u32("ecapa-tdnn.n_fft", 1024);
    m.hop     = u32("ecapa-tdnn.hop_length", 256);
    m.win     = u32("ecapa-tdnn.win_length", 1024);
    m.sr      = u32("ecapa-tdnn.sample_rate", 24000);
    m.fmin    = f32("ecapa-tdnn.mel_fmin", 0.0f);
    m.fmax    = f32("ecapa-tdnn.mel_fmax", 12000.0f);
    {   // kernel size of block0 and the 3 res2net dilations, from the kv arrays
        int64_t kk = gguf_find_key(gguf, "ecapa-tdnn.enc_kernel_sizes");
        int64_t dd = gguf_find_key(gguf, "ecapa-tdnn.enc_dilations");
        if (kk >= 0) m.block0_k = ((const int32_t *) gguf_get_arr_data(gguf, kk))[0];
        if (dd >= 0) {
            const int32_t * d = (const int32_t *) gguf_get_arr_data(gguf, dd);
            m.res_dils = { d[1], d[2], d[3] };
        }
    }

    // backend = CPU
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    m.backend = ggml_backend_dev_init(dev, nullptr);
    m.buft    = ggml_backend_dev_buffer_type(dev);
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
        fseeko(f, (off_t) off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "spk: read fail %s\n", name); fclose(f); return false; }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        m.t[name] = t;
    }
    fclose(f);
    fprintf(stderr, "spk: loaded %zu tensors (enc_dim=%d, block0_k=%d, res_dils=%d,%d,%d)\n",
            m.t.size(), m.enc_dim, m.block0_k, m.res_dils[0], m.res_dils[1], m.res_dils[2]);
    gguf_free(gguf);
    return true;
}

// ---- graph helpers (activations are channel-major [C, T]; ne0=C, ne1=T) ----

// reflect 'same' padding along time (ne1) by p columns, mirroring around the edges.
static ggml_tensor * reflect_pad(ggml_context * ctx, ggml_tensor * x, int p) {
    if (p <= 0) return x;
    const int64_t C = x->ne[0], T = x->ne[1];
    auto col = [&](int64_t i) { return ggml_view_2d(ctx, x, C, 1, x->nb[1], (size_t) i * x->nb[1]); };
    ggml_tensor * acc = nullptr;
    for (int i = p; i >= 1; --i) acc = acc ? ggml_concat(ctx, acc, col(i), 1) : col(i);   // left mirror
    acc = ggml_concat(ctx, acc, x, 1);
    for (int i = 1; i <= p; ++i) acc = ggml_concat(ctx, acc, col(T - 1 - i), 1);            // right mirror
    return acc;  // [C, T+2p]
}

// Conv1d as a sum of K dilated 1x1 matmuls over channels. weight ne=[IC,OC,K], bias [OC].
static ggml_tensor * conv1d(ggml_context * ctx, const spk_model & m, ggml_tensor * x,
                            const std::string & name, int dil, int k) {
    ggml_tensor * w = m.get(name + ".weight");
    ggml_tensor * b = m.get(name + ".bias");
    const int64_t IC = w->ne[0], OC = w->ne[1];
    const int64_t T = x->ne[1], C = x->ne[0];
    const int p = dil * (k - 1) / 2;
    ggml_tensor * xp = reflect_pad(ctx, x, p);   // [C, T+2p]
    ggml_tensor * out = nullptr;
    for (int j = 0; j < k; ++j) {
        ggml_tensor * wj = ggml_view_2d(ctx, w, IC, OC, w->nb[1], (size_t) j * w->nb[2]);     // [IC,OC]
        ggml_tensor * xj = ggml_view_2d(ctx, xp, C, T, xp->nb[1], (size_t) j * dil * xp->nb[1]); // [C,T] window
        ggml_tensor * term = ggml_mul_mat(ctx, wj, xj);   // [OC,T]
        out = out ? ggml_add(ctx, out, term) : term;
    }
    return ggml_add(ctx, out, b);   // bias broadcast over time
}

static ggml_tensor * tdnn(ggml_context * ctx, const spk_model & m, ggml_tensor * x,
                          const std::string & name, int dil, int k) {
    return ggml_relu(ctx, conv1d(ctx, m, x, name, dil, k));
}

// mean over time (ne1) -> [C,1]
static ggml_tensor * time_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t C = x->ne[0];
    ggml_tensor * mt = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)));  // [1,C]
    return ggml_reshape_2d(ctx, mt, C, 1);
}

static ggml_tensor * res2net(ggml_context * ctx, const spk_model & m, ggml_tensor * x,
                             const std::string & prefix, int dil) {
    const int scale = m.res2net_scale;
    const int64_t C = x->ne[0], T = x->ne[1], cs = C / scale;
    ggml_tensor * prev = nullptr;
    std::vector<ggml_tensor *> outs;
    for (int i = 0; i < scale; ++i) {
        ggml_tensor * ci = ggml_cont(ctx, ggml_view_2d(ctx, x, cs, T, x->nb[1], (size_t) i * cs * x->nb[0]));
        ggml_tensor * op;
        if (i == 0)      op = ci;
        else if (i == 1) op = tdnn(ctx, m, ci, prefix + ".blocks.0.conv", dil, 3);
        else             op = tdnn(ctx, m, ggml_add(ctx, ci, prev), prefix + ".blocks." + std::to_string(i - 1) + ".conv", dil, 3);
        prev = op;
        outs.push_back(op);
    }
    ggml_tensor * acc = outs[0];
    for (size_t i = 1; i < outs.size(); ++i) acc = ggml_concat(ctx, acc, outs[i], 0);
    return acc;  // [C,T]
}

static ggml_tensor * se_block(ggml_context * ctx, const spk_model & m, ggml_tensor * x, const std::string & prefix) {
    ggml_tensor * s = time_mean(ctx, x);                                   // [C,1]
    s = ggml_relu(ctx, conv1d(ctx, m, s, prefix + ".conv1", 1, 1));        // [se,1]
    s = ggml_sigmoid(ctx, conv1d(ctx, m, s, prefix + ".conv2", 1, 1));     // [C,1]
    return ggml_mul(ctx, x, s);                                            // gate broadcast over time
}

static ggml_tensor * se_res2net(ggml_context * ctx, const spk_model & m, ggml_tensor * x, int i, int dil) {
    const std::string p = "blocks." + std::to_string(i);
    ggml_tensor * h = tdnn(ctx, m, x, p + ".tdnn1.conv", 1, 1);
    h = res2net(ctx, m, h, p + ".res2net_block", dil);
    h = tdnn(ctx, m, h, p + ".tdnn2.conv", 1, 1);
    h = se_block(ctx, m, h, p + ".se_block");
    return ggml_add(ctx, h, x);
}

// attentive statistics pooling: [C,T] -> [2C,1]
static ggml_tensor * asp(ggml_context * ctx, const spk_model & m, ggml_tensor * h) {
    const int64_t C = h->ne[0], T = h->ne[1];
    ggml_tensor * ht = ggml_cont(ctx, ggml_transpose(ctx, h));            // [T,C]
    ggml_tensor * mean_g = time_mean(ctx, h);                            // [C,1]
    ggml_tensor * dg = ggml_sub(ctx, h, mean_g);                         // [C,T]
    ggml_tensor * var_g = time_mean(ctx, ggml_sqr(ctx, dg));            // [C,1]
    ggml_tensor * std_g = ggml_sqrt(ctx, ggml_clamp(ctx, var_g, 1e-12f, FLT_MAX));
    ggml_tensor * ctxcat = ggml_concat(ctx, ggml_concat(ctx, h, ggml_repeat(ctx, mean_g, h), 0),
                                       ggml_repeat(ctx, std_g, h), 0);    // [3C,T]
    ggml_tensor * a = tdnn(ctx, m, ctxcat, "asp.tdnn.conv", 1, 1);        // [att,T]
    a = ggml_tanh(ctx, a);
    a = conv1d(ctx, m, a, "asp.conv", 1, 1);                             // [C,T]
    ggml_tensor * aw = ggml_soft_max(ctx, ggml_cont(ctx, ggml_transpose(ctx, a)));  // softmax over time -> [T,C]
    ggml_tensor * mean1c = ggml_sum_rows(ctx, ggml_mul(ctx, aw, ht));    // [1,C]
    ggml_tensor * d = ggml_sub(ctx, ht, mean1c);                        // [T,C] - [1,C] broadcast over time
    ggml_tensor * var1c = ggml_sum_rows(ctx, ggml_mul(ctx, aw, ggml_sqr(ctx, d)));  // [1,C]
    ggml_tensor * mean = ggml_reshape_2d(ctx, mean1c, C, 1);            // [C,1]
    ggml_tensor * sd = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_reshape_2d(ctx, var1c, C, 1), 1e-12f, FLT_MAX));
    return ggml_concat(ctx, mean, sd, 0);   // [2C,1]
}

static ggml_tensor * build_ecapa(ggml_context * ctx, const spk_model & m, ggml_tensor * mel) {
    ggml_tensor * h = tdnn(ctx, m, mel, "blocks.0.conv", 1, m.block0_k);  // [512,T]
    std::vector<ggml_tensor *> feats;
    for (int i = 1; i <= 3; ++i) { h = se_res2net(ctx, m, h, i, m.res_dils[i - 1]); feats.push_back(h); }
    ggml_tensor * cat = ggml_concat(ctx, ggml_concat(ctx, feats[0], feats[1], 0), feats[2], 0);  // [1536,T]
    h = tdnn(ctx, m, cat, "mfa.conv", 1, 1);                              // [1536,T]
    ggml_tensor * pooled = asp(ctx, m, h);                                // [3072,1]
    ggml_tensor * emb = conv1d(ctx, m, pooled, "fc", 1, 1);               // [2048,1]
    return ggml_reshape_1d(ctx, emb, m.enc_dim);
}

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

    ggml_context * ctx = ggml_init({ (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true });
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // ---- build the mel input tensor (and, for --wav, the mel frontend) ----
    std::vector<float> host_in;       // data to upload into the first input tensor
    ggml_tensor * input = nullptr;    // tensor to receive host_in
    std::vector<float> cosb, sinb;    // for --wav DFT basis
    ggml_tensor * cosB = nullptr, * sinB = nullptr;
    ggml_tensor * mel = nullptr;

    if (mode == "--mel") {
        std::vector<float> md; std::vector<int64_t> sh;
        if (!npy::load_f32(in_path, md, sh) || sh.size() != 2) { fprintf(stderr, "bad mel npy\n"); return 1; }
        const int64_t T = sh[0], MD = sh[1];   // [T, mel_dim]
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MD, T);  // ne=[mel,T]; row-major [T,mel] maps directly
        ggml_set_input(input);
        host_in = md;
        mel = input;
        fprintf(stderr, "mel: [T=%lld, mel=%lld]\n", (long long) T, (long long) MD);
    } else if (mode == "--wav" || mode == "--raw" || mode == "--clone") {
        std::vector<float> wav;
        if (mode == "--wav") {                 // [N] f32 npy of 24 kHz mono samples
            std::vector<int64_t> sh;
            if (!npy::load_f32(in_path, wav, sh)) { fprintf(stderr, "bad wav npy\n"); return 1; }
        } else if (mode == "--raw") {          // headerless float32 LE mono @ sample_rate
            FILE * rf = fopen(in_path, "rb");   // e.g. ffmpeg -i v.mp3 -ac 1 -ar 24000 -f f32le -
            if (!rf) { fprintf(stderr, "cannot open raw %s\n", in_path); return 1; }
            fseeko(rf, 0, SEEK_END); long sz = ftello(rf); fseeko(rf, 0, SEEK_SET);
            wav.resize(sz / sizeof(float));
            if (fread(wav.data(), 1, (size_t) sz, rf) != (size_t) sz) { fprintf(stderr, "raw read fail\n"); fclose(rf); return 1; }
            fclose(rf);
        } else {                               // --clone: decode any audio via ffmpeg -> 24 kHz mono f32
            std::string cmd = "ffmpeg -v error -i '" + std::string(in_path) +
                              "' -ac 1 -ar " + std::to_string(m.sr) + " -f f32le -";
            FILE * pp = popen(cmd.c_str(), "r");
            if (!pp) { fprintf(stderr, "cannot run ffmpeg\n"); return 1; }
            float buf[8192]; size_t n;
            while ((n = fread(buf, sizeof(float), 8192, pp)) > 0) wav.insert(wav.end(), buf, buf + n);
            int rc = pclose(pp);
            if (rc != 0 || wav.empty()) { fprintf(stderr, "ffmpeg failed (rc=%d, %zu samples) for %s\n", rc, wav.size(), in_path); return 1; }
        }
        const int N = (int) wav.size();
        const int nfft = m.n_fft, hop = m.hop, win = m.win, nfreq = nfft / 2 + 1;
        const int pad = (nfft - hop) / 2;                       // 384, reflect-pad the waveform
        std::vector<float> wp(N + 2 * pad);
        for (int i = 0; i < pad; ++i) { wp[i] = wav[pad - i]; wp[N + pad + i] = wav[N - 2 - i]; }
        for (int i = 0; i < N; ++i) wp[pad + i] = wav[i];
        const int nframes = 1 + (int) (wp.size() - nfft) / hop;
        // analysis window (from gguf), fold into the framed segments
        ggml_tensor * wt = m.get("mel_window");
        std::vector<float> window(win);
        ggml_backend_tensor_get(wt, window.data(), 0, win * sizeof(float));
        host_in.assign((size_t) win * nframes, 0.0f);          // framed: ne=[win, nframes]
        for (int fr = 0; fr < nframes; ++fr)
            for (int n = 0; n < win; ++n)
                host_in[(size_t) fr * win + n] = wp[fr * hop + n] * window[n];
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nframes);
        ggml_set_input(input);
        // DFT basis cos/sin: ne=[win, nfreq], element (n,freq) = cos/sin(2pi*freq*n/nfft)
        cosb.assign((size_t) win * nfreq, 0.0f);
        sinb.assign((size_t) win * nfreq, 0.0f);
        for (int fq = 0; fq < nfreq; ++fq)
            for (int n = 0; n < win; ++n) {
                double a = 2.0 * M_PI * fq * n / nfft;
                cosb[(size_t) fq * win + n] = (float) cos(a);
                sinb[(size_t) fq * win + n] = (float) sin(a);
            }
        cosB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nfreq); ggml_set_input(cosB);
        sinB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nfreq); ggml_set_input(sinB);
        ggml_tensor * re = ggml_mul_mat(ctx, cosB, input);     // [nfreq, nframes]
        ggml_tensor * im = ggml_mul_mat(ctx, sinB, input);
        ggml_tensor * mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));  // magnitude
        ggml_tensor * fb = m.get("mel_fb");                    // ne=[nfreq, n_mels]
        ggml_tensor * me = ggml_mul_mat(ctx, fb, mag);         // [n_mels, nframes]
        mel = ggml_log(ctx, ggml_clamp(ctx, me, 1e-5f, FLT_MAX));
        fprintf(stderr, "wav: N=%d -> %d frames, nfreq=%d\n", N, nframes, nfreq);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }

    ggml_tensor * emb = build_ecapa(ctx, m, mel);
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_gallocr_t galloc = ggml_gallocr_new(m.buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(input, host_in.data(), 0, host_in.size() * sizeof(float));
    if (cosB) ggml_backend_tensor_set(cosB, cosb.data(), 0, cosb.size() * sizeof(float));
    if (sinB) ggml_backend_tensor_set(sinB, sinb.data(), 0, sinb.size() * sizeof(float));

    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    std::vector<float> out(m.enc_dim);
    ggml_backend_tensor_get(emb, out.data(), 0, m.enc_dim * sizeof(float));
    double nrm = 0; for (float v : out) nrm += (double) v * v;
    fprintf(stderr, "emb: [%d] norm=%.4f\n", m.enc_dim, sqrt(nrm));
    npy::save_f32(out_path, out.data(), { (int64_t) m.enc_dim });
    fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
