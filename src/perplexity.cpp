// zonos2-perplexity — teacher-forced perplexity and KL-divergence against a full-precision
// reference, the ZONOS2 analogue of llama.cpp's tools/perplexity. Each (frame position,
// codebook) pair is one prediction event with its own distribution over audio_vocab; the logits
// at position t score frame t+1's codes (skipping the delay-pattern audio_pad_id labels).
//
// Two-pass KL workflow (one model in memory at a time; reference computed once, reused for any
// number of quants):
//   zonos2-perplexity <ref.gguf>   --kl-divergence-base <base.bin> <ids.npy> [more.npy ...]
//   zonos2-perplexity <quant.gguf> --kl-divergence      <base.bin>
// Plain perplexity:
//   zonos2-perplexity <model.gguf> --perplexity <ids.npy> [more.npy ...]
#include "zonos2.h"
#include "npy.h"
#include "imatrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// base-file magic + format version; bump version on any layout change.
const char     KLD_MAGIC[8] = { 'Z', '2', 'K', 'L', 'D', 'I', 'V', '1' };
const uint32_t KLD_VERSION  = 1;

struct sequence {
    std::vector<float> ids;  // flattened row-major [n, W] as floats (npy convention)
    int                n = 0;
};

void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s <model.gguf> --perplexity <ids.npy> [more.npy ...] [--speaker s.npy] [--cpu|--gpu]\n"
        "       %s <ref.gguf>   --kl-divergence-base <base.bin> <ids.npy> [more.npy ...] [--speaker s.npy] [--cpu|--gpu]\n"
        "       %s <quant.gguf> --kl-divergence <base.bin> [--cpu|--gpu]\n"
        "       %s <f16.gguf>   --imatrix-out <imatrix.bin> <ids.npy> [more.npy ...] [--imatrix-min-hits N] [--cpu|--gpu]\n"
        "  (ids.npy: row-major [n, n_codebooks+1] input_ids, e.g. from `zonos2-cli --build-prompt`)\n",
        a0, a0, a0, a0);
}

// load a 2-D [n, W] input_ids npy; W must equal n_codebooks+1.
bool load_ids(const std::string & path, int W, sequence & s) {
    std::vector<int64_t> shape;
    if (!npy::load_f32(path, s.ids, shape) || shape.size() != 2) {
        fprintf(stderr, "perplexity: %s: need a 2-D [n, %d] npy\n", path.c_str(), W);
        return false;
    }
    if ((int) shape[1] != W) {
        fprintf(stderr, "perplexity: %s: width %lld != n_codebooks+1 (%d)\n",
                path.c_str(), (long long) shape[1], W);
        return false;
    }
    s.n = (int) shape[0];
    if (s.n < 2) fprintf(stderr, "perplexity: %s: %d rows, no scorable events\n", path.c_str(), s.n);
    return true;
}

// numerically-stable log-softmax over `n` logits into `logp`; reports argmax.
void log_softmax(const float * logits, int n, std::vector<float> & logp, int & argmax) {
    float mx = logits[0]; argmax = 0;
    for (int i = 1; i < n; ++i) if (logits[i] > mx) { mx = logits[i]; argmax = i; }
    double se = 0.0;
    for (int i = 0; i < n; ++i) se += std::exp((double) (logits[i] - mx));
    const float lse = mx + (float) std::log(se);
    logp.resize(n);
    for (int i = 0; i < n; ++i) logp[i] = logits[i] - lse;
}

// sorted-vector quantile (q in [0,1]); v must be pre-sorted ascending.
double quantile(const std::vector<double> & v, double q) {
    if (v.empty()) return 0.0;
    const size_t i = (size_t) std::lround(q * (double) (v.size() - 1));
    return v[i];
}

template <class T> bool rd(FILE * f, T & x)      { return fread(&x, sizeof(T), 1, f) == 1; }
template <class T> void wr(FILE * f, const T & x) { fwrite(&x, sizeof(T), 1, f); }

// ---------------------------------------------------------------------------
// --perplexity : teacher-forced PPL of one model over the corpus
// ---------------------------------------------------------------------------
int run_perplexity(const zonos2_model & m, const std::vector<sequence> & seqs,
                   const float * spk, int spk_pos) {
    const auto & hp = m.hp;
    const int ncb = (int) hp.n_codebooks, av = (int) hp.audio_vocab, W = ncb + 1;
    const int pad = (int) hp.audio_pad_id;

    std::vector<double>  nll(ncb, 0.0);
    std::vector<int64_t> cnt(ncb, 0);
    int64_t skipped = 0;
    std::vector<float> logits, logp;

    for (const auto & s : seqs) {
        if (!zonos2_logits(m, s.ids.data(), s.n, logits, spk, spk_pos)) return 1;
        for (int t = 0; t + 1 < s.n; ++t) {
            for (int cb = 0; cb < ncb; ++cb) {
                const int L = (int) lroundf(s.ids[(size_t) (t + 1) * W + cb]);
                if (L == pad || L < 0 || L >= av) { ++skipped; continue; }
                int am;
                log_softmax(&logits[((size_t) t * ncb + cb) * av], av, logp, am);
                nll[cb] += -(double) logp[L];
                ++cnt[cb];
            }
        }
    }

    double tot_nll = 0.0; int64_t tot_cnt = 0;
    for (int cb = 0; cb < ncb; ++cb) { tot_nll += nll[cb]; tot_cnt += cnt[cb]; }
    if (tot_cnt == 0) { fprintf(stderr, "perplexity: no events scored\n"); return 1; }

    printf("\n=== perplexity ===\n");
    printf("events: %lld  (skipped pad/oob: %lld)\n", (long long) tot_cnt, (long long) skipped);
    printf("PPL = %.6f   (mean NLL = %.6f nats)\n",
           std::exp(tot_nll / (double) tot_cnt), tot_nll / (double) tot_cnt);
    printf("per-codebook PPL:\n");
    for (int cb = 0; cb < ncb; ++cb) {
        const double p = cnt[cb] ? std::exp(nll[cb] / (double) cnt[cb]) : 0.0;
        printf("  cb%-2d  PPL=%10.4f  (n=%lld)\n", cb, p, (long long) cnt[cb]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// --kl-divergence-base : write reference distributions to base.bin (+ print ref PPL)
// ---------------------------------------------------------------------------
int run_kl_base(const zonos2_model & m, const std::string & base_path,
                const std::vector<sequence> & seqs, const float * spk, int spk_pos) {
    const auto & hp = m.hp;
    const int ncb = (int) hp.n_codebooks, av = (int) hp.audio_vocab, W = ncb + 1;
    const int pad = (int) hp.audio_pad_id;

    FILE * f = fopen(base_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "perplexity: cannot write %s\n", base_path.c_str()); return 1; }

    // header
    fwrite(KLD_MAGIC, 1, 8, f);
    wr<uint32_t>(f, KLD_VERSION);
    wr<uint32_t>(f, (uint32_t) ncb);
    wr<uint32_t>(f, (uint32_t) av);
    wr<uint32_t>(f, (uint32_t) pad);
    // speaker conditioning (reproduced verbatim in pass 2)
    const uint32_t spk_dim = spk ? hp.spk_dim : 0;
    wr<uint32_t>(f, spk_dim);
    wr<int32_t>(f, spk ? spk_pos : -1);
    if (spk_dim) fwrite(spk, sizeof(float), spk_dim, f);
    // sequences: lengths then ids
    wr<uint32_t>(f, (uint32_t) seqs.size());
    for (const auto & s : seqs) wr<uint32_t>(f, (uint32_t) s.n);
    for (const auto & s : seqs) fwrite(s.ids.data(), sizeof(float), s.ids.size(), f);

    // event blocks (iteration order MUST match run_kl_divergence): seq, t, cb
    double tot_nll = 0.0; int64_t tot_cnt = 0, skipped = 0;
    std::vector<float> logits, logp;
    for (const auto & s : seqs) {
        if (!zonos2_logits(m, s.ids.data(), s.n, logits, spk, spk_pos)) { fclose(f); return 1; }
        for (int t = 0; t + 1 < s.n; ++t) {
            for (int cb = 0; cb < ncb; ++cb) {
                const int L = (int) lroundf(s.ids[(size_t) (t + 1) * W + cb]);
                if (L == pad || L < 0 || L >= av) { ++skipped; continue; }
                int am;
                log_softmax(&logits[((size_t) t * ncb + cb) * av], av, logp, am);
                fwrite(logp.data(), sizeof(float), av, f);
                tot_nll += -(double) logp[L];
                ++tot_cnt;
            }
        }
    }
    fclose(f);

    if (tot_cnt == 0) { fprintf(stderr, "perplexity: no events scored\n"); return 1; }
    printf("\n=== kl-divergence base ===\n");
    printf("wrote %s : %lld events, ncb=%d, audio_vocab=%d%s\n", base_path.c_str(),
           (long long) tot_cnt, ncb, av, spk_dim ? " (+speaker)" : "");
    printf("PPL(ref) = %.6f   (skipped pad/oob: %lld)\n",
           std::exp(tot_nll / (double) tot_cnt), (long long) skipped);
    return 0;
}

// ---------------------------------------------------------------------------
// --kl-divergence : score a quant model against base.bin, print the full battery
// ---------------------------------------------------------------------------
int run_kl_divergence(const zonos2_model & m, const std::string & base_path) {
    const auto & hp = m.hp;
    const int ncb = (int) hp.n_codebooks, av = (int) hp.audio_vocab, W = ncb + 1;

    FILE * f = fopen(base_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "perplexity: cannot read %s\n", base_path.c_str()); return 1; }

    char magic[8];
    uint32_t version = 0, b_ncb = 0, b_av = 0, b_pad = 0, b_spk_dim = 0, n_seq = 0;
    int32_t  b_spk_pos = -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, KLD_MAGIC, 8) != 0) {
        fprintf(stderr, "perplexity: %s: bad magic\n", base_path.c_str()); fclose(f); return 1;
    }
    if (!rd(f, version) || version != KLD_VERSION) {
        fprintf(stderr, "perplexity: %s: version %u != %u\n", base_path.c_str(), version, KLD_VERSION);
        fclose(f); return 1;
    }
    rd(f, b_ncb); rd(f, b_av); rd(f, b_pad); rd(f, b_spk_dim); rd(f, b_spk_pos);
    if ((int) b_ncb != ncb || (int) b_av != av) {
        fprintf(stderr, "perplexity: base/model mismatch: base ncb=%u av=%u, model ncb=%d av=%d\n",
                b_ncb, b_av, ncb, av);
        fclose(f); return 1;
    }
    std::vector<float> spk;
    if (b_spk_dim) {
        if (b_spk_dim != hp.spk_dim) {
            fprintf(stderr, "perplexity: base speaker dim %u != model %u\n", b_spk_dim, hp.spk_dim);
            fclose(f); return 1;
        }
        spk.resize(b_spk_dim);
        if (fread(spk.data(), sizeof(float), b_spk_dim, f) != b_spk_dim) {
            fprintf(stderr, "perplexity: %s: truncated speaker block\n", base_path.c_str());
            fclose(f); return 1;
        }
    }
    const float * spk_ptr = spk.empty() ? nullptr : spk.data();

    rd(f, n_seq);
    std::vector<sequence> seqs(n_seq);
    for (auto & s : seqs) { uint32_t n = 0; rd(f, n); s.n = (int) n; }
    for (auto & s : seqs) {
        s.ids.resize((size_t) s.n * W);
        if (fread(s.ids.data(), sizeof(float), s.ids.size(), f) != s.ids.size()) {
            fprintf(stderr, "perplexity: %s: truncated ids block\n", base_path.c_str());
            fclose(f); return 1;
        }
    }

    // accumulators
    std::vector<double> kld_all;
    std::vector<double> kld_cb(ncb, 0.0), agree_cb(ncb, 0.0);
    std::vector<int64_t> cnt_cb(ncb, 0);
    double ref_nll = 0.0, test_nll = 0.0, dlp_sum = 0.0, dlp_sq = 0.0;
    int64_t n_ev = 0, agree = 0;

    std::vector<float> ref_logp((size_t) av), logits, test_logp;
    for (const auto & s : seqs) {
        if (!zonos2_logits(m, s.ids.data(), s.n, logits, spk_ptr, b_spk_pos)) { fclose(f); return 1; }
        for (int t = 0; t + 1 < s.n; ++t) {
            for (int cb = 0; cb < ncb; ++cb) {
                const int L = (int) lroundf(s.ids[(size_t) (t + 1) * W + cb]);
                if (L == (int) b_pad || L < 0 || L >= av) continue; // identical skip to base
                if (fread(ref_logp.data(), sizeof(float), av, f) != (size_t) av) {
                    fprintf(stderr, "perplexity: %s: truncated event block\n", base_path.c_str());
                    fclose(f); return 1;
                }
                int test_am;
                log_softmax(&logits[((size_t) t * ncb + cb) * av], av, test_logp, test_am);

                // KLD(ref || test) = sum_v p_ref[v] (logp_ref[v] - logp_test[v]); ref argmax inline
                double kld = 0.0; int ref_am = 0; float ref_mx = ref_logp[0];
                for (int v = 0; v < av; ++v) {
                    const float lr = ref_logp[v];
                    if (lr > ref_mx) { ref_mx = lr; ref_am = v; }
                    kld += std::exp((double) lr) * (double) (lr - test_logp[v]);
                }
                if (kld < 0.0) kld = 0.0; // guard tiny negative from fp error

                const double dlp = (double) test_logp[L] - (double) ref_logp[L];
                kld_all.push_back(kld);
                kld_cb[cb] += kld; cnt_cb[cb]++;
                if (ref_am == test_am) { ++agree; agree_cb[cb] += 1.0; }
                ref_nll  += -(double) ref_logp[L];
                test_nll += -(double) test_logp[L];
                dlp_sum  += dlp; dlp_sq += dlp * dlp;
                ++n_ev;
            }
        }
    }
    fclose(f);

    if (n_ev == 0) { fprintf(stderr, "perplexity: no events scored\n"); return 1; }
    std::sort(kld_all.begin(), kld_all.end());
    double kld_mean = 0.0; for (double k : kld_all) kld_mean += k; kld_mean /= (double) n_ev;
    const double ppl_ref = std::exp(ref_nll / (double) n_ev);
    const double ppl_tst = std::exp(test_nll / (double) n_ev);

    printf("\n=== kl-divergence vs %s ===\n", base_path.c_str());
    printf("events: %lld\n", (long long) n_ev);
    printf("PPL(ref)   = %.6f\n", ppl_ref);
    printf("PPL(test)  = %.6f   (ratio test/ref = %.6f)\n", ppl_tst, ppl_tst / ppl_ref);
    printf("KLD: mean=%.6f  median=%.6f  p90=%.6f  p95=%.6f  p99=%.6f  max=%.6f  min=%.6f\n",
           kld_mean, quantile(kld_all, 0.50), quantile(kld_all, 0.90), quantile(kld_all, 0.95),
           quantile(kld_all, 0.99), kld_all.back(), kld_all.front());
    printf("Top-1 agreement: %.4f%%\n", 100.0 * (double) agree / (double) n_ev);
    printf("Delta-logprob(true token): mean=%.6f  rms=%.6f\n",
           dlp_sum / (double) n_ev, std::sqrt(dlp_sq / (double) n_ev));
    printf("per-codebook:\n");
    for (int cb = 0; cb < ncb; ++cb) {
        const double km = cnt_cb[cb] ? kld_cb[cb] / (double) cnt_cb[cb] : 0.0;
        const double ag = cnt_cb[cb] ? 100.0 * agree_cb[cb] / (double) cnt_cb[cb] : 0.0;
        printf("  cb%-2d  KLD_mean=%.6f  top1=%7.3f%%  (n=%lld)\n", cb, km, ag, (long long) cnt_cb[cb]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// --imatrix-out : collect per-expert importance (mean activation^2) over the corpus
// ---------------------------------------------------------------------------
int run_imatrix(const zonos2_model & m, const std::vector<sequence> & seqs,
                const float * spk, int spk_pos, const std::string & out_path, int min_hits) {
    const int ne = (int) m.hp.n_expert;
    if (ne == 0) { fprintf(stderr, "imatrix: model has no experts\n"); return 1; }

    struct acc_t { int n_embd = 0, n_ff = 0; std::vector<double> gu, dn; std::vector<int64_t> cnt; };
    std::map<int, acc_t> acc;                                  // per MoE layer

    std::vector<zonos2_moe_act> acts;
    size_t si = 0;
    for (const auto & s : seqs) {
        acts.clear();
        if (!zonos2_moe_capture(m, s.ids.data(), s.n, acts, spk, spk_pos)) return 1;
        for (const auto & a : acts) {
            acc_t & A = acc[a.layer];
            if (A.gu.empty()) {
                A.n_embd = a.n_embd; A.n_ff = a.n_ff;
                A.gu.assign((size_t) ne * a.n_embd, 0.0);
                A.dn.assign((size_t) ne * a.n_ff, 0.0);
                A.cnt.assign(ne, 0);
            }
            for (int t = 0; t < a.n; ++t) {
                for (int j = 0; j < a.k; ++j) {
                    const int e = a.sel[(size_t) t * a.k + j];
                    if (e < 0 || e >= ne) continue;
                    A.cnt[e]++;
                    const float * x  = &a.moe_in[(size_t) t * a.n_embd];
                    double * gu = &A.gu[(size_t) e * a.n_embd];
                    for (int c = 0; c < a.n_embd; ++c) gu[c] += (double) x[c] * x[c];
                    const float * yv = &a.moe_y[((size_t) t * a.k + j) * a.n_ff];
                    double * dn = &A.dn[(size_t) e * a.n_ff];
                    for (int c = 0; c < a.n_ff; ++c) dn[c] += (double) yv[c] * yv[c];
                }
            }
        }
        fprintf(stderr, "imatrix: seq %zu/%zu (%d rows)\n", ++si, seqs.size(), s.n);
    }

    std::map<std::string, imatrix::entry> im;
    int64_t total_hits = 0; int zero_slots = 0, below_thresh = 0, n_layers = 0;
    std::vector<int64_t> all_cnt;                              // every (layer,expert) hit count
    for (auto & kv : acc) {
        const std::string p = "blk." + std::to_string(kv.first) + ".";
        acc_t & A = kv.second; ++n_layers;
        imatrix::entry gate, up, down;
        gate.n_expert = up.n_expert = down.n_expert = (uint32_t) ne;
        gate.n_in = up.n_in = (uint32_t) A.n_embd; down.n_in = (uint32_t) A.n_ff;
        gate.data.resize((size_t) ne * A.n_embd);
        down.data.resize((size_t) ne * A.n_ff);
        for (int e = 0; e < ne; ++e) {
            // Experts seen fewer than min_hits times get a zeroed row: the apply side
            // (quantize-cli) treats an all-zero importance vector as degenerate and falls back
            // to plain RTN, which beats trusting a noise-dominated importance estimate.
            const bool keep = A.cnt[e] >= min_hits;
            const double inv = keep ? 1.0 / (double) A.cnt[e] : 0.0;
            if (A.cnt[e] == 0) ++zero_slots;
            else if (!keep)    ++below_thresh;
            total_hits += A.cnt[e];
            all_cnt.push_back(A.cnt[e]);
            for (int c = 0; c < A.n_embd; ++c) gate.data[(size_t) e*A.n_embd + c] = (float) (A.gu[(size_t) e*A.n_embd + c] * inv);
            for (int c = 0; c < A.n_ff;   ++c) down.data[(size_t) e*A.n_ff   + c] = (float) (A.dn[(size_t) e*A.n_ff   + c] * inv);
        }
        up.data = gate.data;                                   // gate/up share the same input
        im[p + "ffn_gate_exps.weight"] = std::move(gate);
        im[p + "ffn_up_exps.weight"]   = std::move(up);
        im[p + "ffn_down_exps.weight"] = std::move(down);
    }
    if (!imatrix::save(out_path, im)) return 1;

    std::sort(all_cnt.begin(), all_cnt.end());
    const int64_t hmin = all_cnt.empty() ? 0 : all_cnt.front();
    const int64_t hmed = all_cnt.empty() ? 0 : all_cnt[all_cnt.size() / 2];
    const int64_t hmax = all_cnt.empty() ? 0 : all_cnt.back();
    printf("\n=== imatrix ===\n");
    printf("wrote %s : %d MoE layers, %d experts, %lld routed (token,expert) hits\n",
           out_path.c_str(), n_layers, ne, (long long) total_hits);
    printf("per-(layer,expert) hits: min %lld, median %lld, max %lld (min-hits floor = %d)\n",
           (long long) hmin, (long long) hmed, (long long) hmax, min_hits);
    if (zero_slots || below_thresh)
        printf("fallback to RTN: %d slots zero-hit, %d below floor (of %d total) — %s\n",
               zero_slots, below_thresh, (int) all_cnt.size(),
               (zero_slots + below_thresh) * 4 > (int) all_cnt.size()
                   ? "consider more/longer generation traces" : "ok, well-covered");
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const std::string model_path = argv[1];

    enum { NONE, PPL, KLBASE, KLDIV, IMATRIX } mode = NONE;
    bool use_gpu = false;
    std::string base_path, spk_path;
    std::vector<std::string> ids_paths;
    int spk_pos = 0;
    int imat_min_hits = 32;   // experts seen fewer than this many times fall back to RTN

    for (int i = 2; i < argc; ++i) {
        const char * a = argv[i];
        if      (!strcmp(a, "--gpu")) use_gpu = true;
        else if (!strcmp(a, "--cpu")) use_gpu = false;
        else if (!strcmp(a, "--perplexity")) mode = PPL;
        else if (!strcmp(a, "--kl-divergence-base") && i + 1 < argc) { mode = KLBASE; base_path = argv[++i]; }
        else if (!strcmp(a, "--kl-divergence")      && i + 1 < argc) { mode = KLDIV;  base_path = argv[++i]; }
        else if (!strcmp(a, "--imatrix-out")        && i + 1 < argc) { mode = IMATRIX; base_path = argv[++i]; }
        else if (!strcmp(a, "--imatrix-min-hits")   && i + 1 < argc) imat_min_hits = atoi(argv[++i]);
        else if (!strcmp(a, "--speaker")     && i + 1 < argc) spk_path = argv[++i];
        else if (!strcmp(a, "--speaker-pos") && i + 1 < argc) spk_pos  = atoi(argv[++i]);
        else if (a[0] != '-') ids_paths.push_back(a); // positional input_ids npy
        else { usage(argv[0]); return 1; }
    }
    if (mode == NONE) { usage(argv[0]); return 1; }
    if ((mode == PPL || mode == KLBASE || mode == IMATRIX) && ids_paths.empty()) {
        fprintf(stderr, "perplexity: need at least one <ids.npy>\n"); return 1;
    }

    zonos2_model model;
    if (!zonos2_model_load(model, model_path.c_str(), use_gpu)) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    const int W = (int) model.hp.n_codebooks + 1;

    // optional speaker embedding (kl-divergence reads it from the base file instead)
    std::vector<float> spk;
    if (!spk_path.empty() && mode != KLDIV) {
        std::vector<int64_t> sshape;
        if (!npy::load_f32(spk_path, spk, sshape) || spk.size() != model.hp.spk_dim) {
            fprintf(stderr, "perplexity: bad speaker npy %s (want %u-d)\n", spk_path.c_str(), model.hp.spk_dim);
            zonos2_model_free(model); return 1;
        }
        printf("speaker: %zu-d from %s, pos=%d\n", spk.size(), spk_path.c_str(), spk_pos);
    }
    const float * spk_ptr = spk.empty() ? nullptr : spk.data();

    int rc = 1;
    if (mode == KLDIV) {
        rc = run_kl_divergence(model, base_path);
    } else {
        std::vector<sequence> seqs(ids_paths.size());
        bool ok = true;
        for (size_t i = 0; i < ids_paths.size() && ok; ++i) ok = load_ids(ids_paths[i], W, seqs[i]);
        if (ok) {
            rc = (mode == PPL)     ? run_perplexity(model, seqs, spk_ptr, spk_pos)
               : (mode == IMATRIX) ? run_imatrix(model, seqs, spk_ptr, spk_pos, base_path, imat_min_hits)
                                   : run_kl_base(model, base_path, seqs, spk_ptr, spk_pos);
        }
    }

    zonos2_model_free(model);
    return rc;
}
