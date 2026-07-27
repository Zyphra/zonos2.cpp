// spk-encoder — ggml port of the ECAPA-TDNN speaker encoder
// (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B, actually a ~6M-param ECAPA-TDNN).
// log-mel [128,T] -> 2048-d x-vector. Optional in-graph mel frontend from a 24 kHz
// mono waveform (STFT via DFT matmul + slaney mel + log). CPU backend.
#ifdef _WIN32
// STARTUPINFOEX and PROC_THREAD_ATTRIBUTE_HANDLE_LIST are available on Vista+.
// Define the floor before any project or system header can include windows.h.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#endif

#include "compat.h"
#include "spk-encoder.h"
#include "exe-path.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

bool spk_load(spk_model & m, const char * path) {
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
        fseeko(f, off, SEEK_SET);
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

void spk_free(spk_model & m) {
    if (m.buf_w)   ggml_backend_buffer_free(m.buf_w);
    if (m.ctx_w)   ggml_free(m.ctx_w);
    if (m.backend) ggml_backend_free(m.backend);
    m.buf_w = nullptr; m.ctx_w = nullptr; m.backend = nullptr; m.t.clear();
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

// ---- compute drivers ----

namespace {
struct spk_input { ggml_tensor * t; const float * data; size_t n; };

// Build + compute the graph ending at `emb`, upload `inputs`, and read back the enc_dim vector.
std::vector<float> spk_run(const spk_model & m, ggml_context * ctx, ggml_tensor * emb,
                           const std::vector<spk_input> & inputs) {
    ggml_set_output(emb);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, emb);

    ggml_gallocr_t galloc = ggml_gallocr_new(m.buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "spk: alloc failed\n"); ggml_gallocr_free(galloc); return {}; }
    for (const auto & in : inputs) ggml_backend_tensor_set(in.t, in.data, 0, in.n * sizeof(float));
    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "spk: compute failed\n"); ggml_gallocr_free(galloc); return {};
    }
    std::vector<float> out(m.enc_dim);
    ggml_backend_tensor_get(emb, out.data(), 0, (size_t) m.enc_dim * sizeof(float));
    ggml_gallocr_free(galloc);
    return out;
}
} // namespace

std::vector<float> spk_embed_from_mel(const spk_model & m, const float * mel_data, int T) {
    if (T <= 0) { fprintf(stderr, "spk: empty mel\n"); return {}; }
    ggml_context * ctx = ggml_init({ (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true });
    // ne=[mel,T]; a row-major [T,mel] host buffer maps directly.
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.mel_dim, T);
    ggml_set_input(input);
    ggml_tensor * emb = build_ecapa(ctx, m, input);
    std::vector<float> out = spk_run(m, ctx, emb, {{ input, mel_data, (size_t) m.mel_dim * T }});
    ggml_free(ctx);
    return out;
}

std::vector<float> spk_embed_from_pcm24k(const spk_model & m, const float * wav, int N) {
    if (N <= 0) { fprintf(stderr, "spk: empty waveform\n"); return {}; }
    const int nfft = m.n_fft, hop = m.hop, win = m.win, nfreq = nfft / 2 + 1;
    const int pad = (nfft - hop) / 2;                       // reflect-pad the waveform
    std::vector<float> wp((size_t) N + 2 * pad);
    for (int i = 0; i < pad; ++i) { wp[i] = wav[pad - i]; wp[N + pad + i] = wav[N - 2 - i]; }
    for (int i = 0; i < N; ++i) wp[pad + i] = wav[i];
    const int nframes = 1 + (int) (wp.size() - nfft) / hop;
    if (nframes <= 0) { fprintf(stderr, "spk: waveform too short (%d samples)\n", N); return {}; }

    ggml_context * ctx = ggml_init({ (size_t) 256 * 1024 * 1024, nullptr, /*no_alloc=*/ true });

    // analysis window (from gguf), folded into the framed segments
    ggml_tensor * wt = m.get("mel_window");
    std::vector<float> window(win);
    ggml_backend_tensor_get(wt, window.data(), 0, (size_t) win * sizeof(float));
    std::vector<float> host_in((size_t) win * nframes, 0.0f);          // framed: ne=[win, nframes]
    for (int fr = 0; fr < nframes; ++fr)
        for (int n = 0; n < win; ++n)
            host_in[(size_t) fr * win + n] = wp[(size_t) fr * hop + n] * window[n];
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nframes);
    ggml_set_input(input);

    // DFT basis cos/sin: ne=[win, nfreq], element (n,freq) = cos/sin(2pi*freq*n/nfft)
    std::vector<float> cosb((size_t) win * nfreq, 0.0f), sinb((size_t) win * nfreq, 0.0f);
    for (int fq = 0; fq < nfreq; ++fq)
        for (int n = 0; n < win; ++n) {
            double a = 2.0 * M_PI * fq * n / nfft;
            cosb[(size_t) fq * win + n] = (float) cos(a);
            sinb[(size_t) fq * win + n] = (float) sin(a);
        }
    ggml_tensor * cosB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nfreq); ggml_set_input(cosB);
    ggml_tensor * sinB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, win, nfreq); ggml_set_input(sinB);
    ggml_tensor * re = ggml_mul_mat(ctx, cosB, input);     // [nfreq, nframes]
    ggml_tensor * im = ggml_mul_mat(ctx, sinB, input);
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));  // magnitude
    ggml_tensor * fb = m.get("mel_fb");                    // ne=[nfreq, n_mels]
    ggml_tensor * me = ggml_mul_mat(ctx, fb, mag);         // [n_mels, nframes]
    ggml_tensor * mel = ggml_log(ctx, ggml_clamp(ctx, me, 1e-5f, FLT_MAX));

    ggml_tensor * emb = build_ecapa(ctx, m, mel);
    std::vector<float> out = spk_run(m, ctx, emb, {
        { input, host_in.data(), host_in.size() },
        { cosB,  cosb.data(),    cosb.size()    },
        { sinB,  sinb.data(),    sinb.size()    },
    });
    ggml_free(ctx);
    return out;
}

static std::string shell_quote_path(const char * path) {
    std::string q = "'";
    for (const char * p = path; *p; ++p) q += (*p == '\'') ? "'\\''" : std::string(1, *p);
    return q + "'";
}

// Locate the ffmpeg used to decode reference audio for cloning. The launcher and the
// desktop app download an ffmpeg on demand and point us at it via ZONOS2_FFMPEG; we
// also look next to the executable (dist layout / macOS bundle Resources) before
// falling back to a bare "ffmpeg" on PATH.
static std::filesystem::path resolve_ffmpeg() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    if (const wchar_t * e = _wgetenv(L"ZONOS2_FFMPEG"); e && *e) return fs::path(e);
    const fs::path name = L"ffmpeg.exe";
#else
    if (const char * e = getenv("ZONOS2_FFMPEG"); e && *e) return fs::path(e);
    const fs::path name = "ffmpeg";
#endif
    std::error_code ec;
    const fs::path dir = zonos2_exe_dir();
    for (const fs::path c : { dir / name, dir / "bin" / name, dir / ".." / "Resources" / name })
        if (fs::exists(c, ec)) return c.lexically_normal();
    return name;   // rely on PATH
}

#ifdef _WIN32
namespace {

// A small owning wrapper keeps every process/pipe handle on one cleanup path.
class win_handle {
public:
    win_handle() = default;
    explicit win_handle(HANDLE handle) : handle_(handle) {}
    ~win_handle() { reset(); }

    win_handle(const win_handle &) = delete;
    win_handle & operator=(const win_handle &) = delete;

    HANDLE get() const { return handle_; }
    bool valid() const { return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE; }

    void reset(HANDLE handle = INVALID_HANDLE_VALUE) {
        if (valid()) CloseHandle(handle_);
        handle_ = handle;
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

static void log_windows_error(const char * what, DWORD error) {
    fprintf(stderr, "spk: %s (Windows error %lu)\n", what, (unsigned long) error);
}

// CreateProcessW does not search PATH when lpApplicationName is supplied as a bare
// filename. Resolve that case ourselves, while keeping explicit paths explicit.
static std::filesystem::path resolve_windows_executable(const std::filesystem::path & requested) {
    namespace fs = std::filesystem;
    if (requested.has_parent_path()) {
        std::error_code ec;
        const fs::path absolute = fs::absolute(requested, ec);
        return ec ? requested : absolute.lexically_normal();
    }

    std::vector<wchar_t> buffer(32768);
    const DWORD n = SearchPathW(
        nullptr,
        requested.c_str(),
        nullptr,
        (DWORD) buffer.size(),
        buffer.data(),
        nullptr);
    if (n == 0 || n >= buffer.size()) return {};
    return fs::path(buffer.data(), buffer.data() + n);
}

// Quote one argv element according to the parsing rules used by the Microsoft C
// runtime. This is argument quoting only; no command shell is involved.
static std::wstring quote_windows_argument(const std::wstring & argument) {
    const bool needs_quotes = argument.empty() || std::any_of(
        argument.begin(),
        argument.end(),
        [](wchar_t c) { return std::iswspace(c) || c == L'"'; });
    if (!needs_quotes) return argument;

    std::wstring quoted(1, L'"');
    size_t backslashes = 0;
    for (const wchar_t c : argument) {
        if (c == L'\\') {
            ++backslashes;
        } else if (c == L'"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(L'"');
            backslashes = 0;
        } else {
            quoted.append(backslashes, L'\\');
            backslashes = 0;
            quoted.push_back(c);
        }
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

static win_handle inheritable_stderr(SECURITY_ATTRIBUTES & security) {
    HANDLE duplicate = INVALID_HANDLE_VALUE;
    const HANDLE current = GetStdHandle(STD_ERROR_HANDLE);
    if (current != nullptr && current != INVALID_HANDLE_VALUE &&
        DuplicateHandle(
            GetCurrentProcess(),
            current,
            GetCurrentProcess(),
            &duplicate,
            0,
            TRUE,
            DUPLICATE_SAME_ACCESS)) {
        return win_handle(duplicate);
    }

    return win_handle(CreateFileW(
        L"NUL",
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr));
}

static std::vector<float> decode_audio_windows(const spk_model & m, const char * path) {
    namespace fs = std::filesystem;

    const fs::path ffmpeg = resolve_windows_executable(resolve_ffmpeg());
    if (ffmpeg.empty()) {
        fprintf(stderr, "spk: cannot find ffmpeg.exe\n");
        return {};
    }

    const fs::path input = fs::u8path(path);
    std::wstring command =
        quote_windows_argument(ffmpeg.native()) +
        L" -nostdin -v error -i " +
        quote_windows_argument(input.native()) +
        L" -ac 1 -ar " +
        std::to_wstring(m.sr) +
        L" -f f32le -";

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE stdout_read_raw = INVALID_HANDLE_VALUE;
    HANDLE stdout_write_raw = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&stdout_read_raw, &stdout_write_raw, &security, 0)) {
        log_windows_error("cannot create ffmpeg output pipe", GetLastError());
        return {};
    }
    win_handle stdout_read(stdout_read_raw);
    win_handle stdout_write(stdout_write_raw);
    if (!SetHandleInformation(stdout_read.get(), HANDLE_FLAG_INHERIT, 0)) {
        log_windows_error("cannot protect ffmpeg output pipe", GetLastError());
        return {};
    }

    win_handle stdin_null(CreateFileW(
        L"NUL",
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr));
    win_handle stderr_handle = inheritable_stderr(security);
    if (!stdin_null.valid() || !stderr_handle.valid()) {
        log_windows_error("cannot prepare ffmpeg standard handles", GetLastError());
        return {};
    }

    std::array<HANDLE, 3> inherited_handles = {
        stdin_null.get(),
        stdout_write.get(),
        stderr_handle.get(),
    };
    SIZE_T attribute_bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
    if (attribute_bytes == 0) {
        log_windows_error("cannot size ffmpeg process attributes", GetLastError());
        return {};
    }

    std::vector<unsigned char> attribute_storage(attribute_bytes);
    auto * attributes =
        reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
        log_windows_error("cannot initialize ffmpeg process attributes", GetLastError());
        return {};
    }

    const bool handles_updated = UpdateProcThreadAttribute(
        attributes,
        0,
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        inherited_handles.data(),
        sizeof(inherited_handles),
        nullptr,
        nullptr) != FALSE;
    if (!handles_updated) {
        const DWORD error = GetLastError();
        DeleteProcThreadAttributeList(attributes);
        log_windows_error("cannot restrict ffmpeg inherited handles", error);
        return {};
    }

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdin_null.get();
    startup.StartupInfo.hStdOutput = stdout_write.get();
    startup.StartupInfo.hStdError = stderr_handle.get();
    startup.lpAttributeList = attributes;

    PROCESS_INFORMATION process_info{};
    const BOOL created = CreateProcessW(
        ffmpeg.c_str(),
        command.data(),
        nullptr,
        nullptr,
        TRUE,
        EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup.StartupInfo,
        &process_info);
    const DWORD create_error = created ? ERROR_SUCCESS : GetLastError();
    DeleteProcThreadAttributeList(attributes);
    if (!created) {
        log_windows_error("cannot start ffmpeg", create_error);
        return {};
    }

    win_handle process(process_info.hProcess);
    win_handle thread(process_info.hThread);
    // Only the child may retain these ends while the parent drains stdout.
    stdout_write.reset();
    stdin_null.reset();
    stderr_handle.reset();

    std::vector<unsigned char> raw;
    std::array<unsigned char, 32768> buffer{};
    DWORD read_error = ERROR_SUCCESS;
    for (;;) {
        DWORD bytes_read = 0;
        if (!ReadFile(
                stdout_read.get(),
                buffer.data(),
                (DWORD) buffer.size(),
                &bytes_read,
                nullptr)) {
            const DWORD error = GetLastError();
            if (error != ERROR_BROKEN_PIPE) read_error = error;
            break;
        }
        if (bytes_read == 0) break;
        raw.insert(raw.end(), buffer.begin(), buffer.begin() + bytes_read);
    }

    // If the pipe itself failed, close our read end before waiting so a child
    // that is still writing receives a broken pipe instead of blocking forever.
    if (read_error != ERROR_SUCCESS) stdout_read.reset();
    const DWORD wait_result = WaitForSingleObject(process.get(), INFINITE);
    DWORD exit_code = 1;
    const bool got_exit_code = GetExitCodeProcess(process.get(), &exit_code) != FALSE;
    if (read_error != ERROR_SUCCESS) {
        log_windows_error("cannot read ffmpeg output", read_error);
        return {};
    }
    if (wait_result != WAIT_OBJECT_0) {
        log_windows_error("cannot wait for ffmpeg", GetLastError());
        return {};
    }
    if (!got_exit_code) {
        log_windows_error("cannot read ffmpeg exit code", GetLastError());
        return {};
    }
    if (exit_code != 0 || raw.empty() || raw.size() % sizeof(float) != 0) {
        fprintf(
            stderr,
            "spk: ffmpeg failed (rc=%lu, %zu bytes) for %s\n",
            (unsigned long) exit_code,
            raw.size(),
            path);
        return {};
    }

    std::vector<float> wav(raw.size() / sizeof(float));
    memcpy(wav.data(), raw.data(), raw.size());
    return wav;
}

} // namespace
#endif

std::vector<float> spk_decode_audio_file(const spk_model & m, const char * path) {
    // decode any audio via ffmpeg -> 24 kHz mono f32
#ifdef _WIN32
    // Do not use _popen here: it delegates to cmd.exe, whose /c quote stripping
    // breaks an absolute executable path followed by another quoted input path.
    return decode_audio_windows(m, path);
#else
    const std::string ffmpeg = resolve_ffmpeg().string();
    std::string cmd = shell_quote_path(ffmpeg.c_str()) + " -nostdin -v error -i " +
                      shell_quote_path(path) + " -ac 1 -ar " + std::to_string(m.sr) +
                      " -f f32le -";
    FILE * pp = popen(cmd.c_str(), "r");
    if (!pp) { fprintf(stderr, "spk: cannot run ffmpeg\n"); return {}; }
    std::vector<float> wav;
    float buf[8192]; size_t n;
    while ((n = fread(buf, sizeof(float), 8192, pp)) > 0) wav.insert(wav.end(), buf, buf + n);
    int rc = pclose(pp);
    if (rc != 0 || wav.empty()) {
        fprintf(stderr, "spk: ffmpeg failed (rc=%d, %zu samples) for %s\n", rc, wav.size(), path);
        return {};
    }
    return wav;
#endif
}

std::vector<float> spk_embed_from_file(const spk_model & m, const char * path) {
    std::vector<float> wav = spk_decode_audio_file(m, path);
    if (wav.empty()) return {};
    return spk_embed_from_pcm24k(m, wav.data(), (int) wav.size());
}
