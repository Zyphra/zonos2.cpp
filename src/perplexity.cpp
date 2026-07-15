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
#include "prune-stats.h"
#include "prune-policy.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// base-file magic + format version; bump version on any layout change.
// v2: per-sequence conditioning (each seq carries its own speaker vector + spk_pos)
// replaces v1's single global speaker block, so one base can cover multi-speaker /
// multi-path corpora. Old v1 bases are rejected (regenerate).
const char     KLD_MAGIC[8] = { 'Z', '2', 'K', 'L', 'D', 'I', 'V', '2' };
const uint32_t KLD_VERSION  = 2;

struct sequence {
    std::vector<float> ids;        // flattened row-major [n, W] as floats (npy convention)
    int                n = 0;
    std::vector<float> spk;        // per-sequence speaker embedding [spk_dim]; empty => no speaker
    int                spk_pos = -1;
    const float * spk_ptr() const { return spk.empty() ? nullptr : spk.data(); }
};

void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s <model.gguf> --perplexity <ids.npy> [more.npy ...] [--speaker s.npy] [--cpu|--gpu]\n"
        "       %s <ref.gguf>   --kl-divergence-base <base.bin> {<ids.npy>... [--speaker s.npy] | --manifest m.txt} [--cpu|--gpu]\n"
        "       %s <quant.gguf> --kl-divergence <base.bin> [--cpu|--gpu]\n"
        "       %s <f16.gguf>   --imatrix-out <imatrix.bin> {<ids.npy>... | --manifest m.txt} [--imatrix-min-hits N] [--cpu|--gpu]\n"
        "       %s <f16.gguf>   --prune-stats <stats.bin> {<ids.npy>... | --manifest m.txt} [--cpu|--gpu]\n"
        "  (--imatrix-out and --prune-stats may be combined: one capture pass, two sidecar outputs)\n"
        "  (--prune-mask <stats.bin> (--keep N | --drop-below-hits H): runtime expert pruning before any\n"
        "   mode above -- sweep the quality knee vs a --kl-divergence base without writing GGUFs)\n"
        "  (--skip-layers L1,L2,...: depth pruning -- bypass whole transformer blocks (attn+FFN) before\n"
        "   any mode above; sweep the block-redundancy knee vs a --kl-divergence base)\n"
        "  (ids.npy: row-major [n, n_codebooks+1] input_ids, e.g. from `zonos2-cli --build-prompt`)\n"
        "  (--manifest: per-line `ids.npy [speaker.npy [spk_pos]]` -- per-sequence conditioning for multi-speaker/multi-path bases)\n",
        a0, a0, a0, a0, a0);
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

// Load a per-sequence speaker .npy ([spk_dim] or [1, spk_dim]) into s.spk; validate dim.
bool load_seq_speaker(const std::string & path, uint32_t spk_dim, int spk_pos, sequence & s) {
    std::vector<int64_t> sshape;
    if (!npy::load_f32(path, s.spk, sshape) || s.spk.size() != spk_dim) {
        fprintf(stderr, "perplexity: bad speaker npy %s (want %u-d)\n", path.c_str(), spk_dim);
        return false;
    }
    s.spk_pos = spk_pos;
    return true;
}

// Manifest: one trace per line, whitespace-separated `ids.npy [speaker.npy [spk_pos]]`.
// A bare ids path => no-speaker (plain TTS) sequence. '#' starts a comment; blank lines skipped.
// This is the "any path" lever: each trace carries its own conditioning, so one base file
// can mix no-speaker, multi-speaker, and any conditioning-bucket variants.
bool load_manifest(const std::string & path, int W, uint32_t spk_dim, std::vector<sequence> & seqs) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "perplexity: cannot read manifest %s\n", path.c_str()); return false; }
    char line[4096];
    int lineno = 0;
    bool ok = true;
    while (ok && fgets(line, sizeof(line), f)) {
        ++lineno;
        if (char * h = strchr(line, '#')) *h = '\0';
        std::string ids_p, spk_p; int spk_pos = 0;
        // tokenize on whitespace
        const char * p = line;
        auto next_tok = [&](std::string & out) {
            while (*p && isspace((unsigned char) *p)) ++p;
            const char * st = p;
            while (*p && !isspace((unsigned char) *p)) ++p;
            out.assign(st, p - st);
            return !out.empty();
        };
        if (!next_tok(ids_p)) continue;       // blank/comment-only line
        std::string tok;
        if (next_tok(spk_p)) { if (next_tok(tok)) spk_pos = atoi(tok.c_str()); }

        seqs.emplace_back();
        sequence & s = seqs.back();
        if (!load_ids(ids_p, W, s)) { ok = false; break; }
        if (!spk_p.empty()) {
            if (spk_dim == 0) { fprintf(stderr, "perplexity: manifest line %d names a speaker but model has spk_dim=0\n", lineno); ok = false; break; }
            if (!load_seq_speaker(spk_p, spk_dim, spk_pos, s)) { ok = false; break; }
        }
    }
    fclose(f);
    return ok;
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
                const std::vector<sequence> & seqs) {
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
    // per-sequence metadata: n, spk_pos, spk_dim (0 => no speaker) -- then ids (+ speaker if any)
    wr<uint32_t>(f, (uint32_t) seqs.size());
    for (const auto & s : seqs) {
        wr<uint32_t>(f, (uint32_t) s.n);
        wr<int32_t>(f, s.spk.empty() ? -1 : s.spk_pos);
        wr<uint32_t>(f, (uint32_t) s.spk.size());     // == hp.spk_dim or 0
    }
    for (const auto & s : seqs) {
        fwrite(s.ids.data(), sizeof(float), s.ids.size(), f);
        if (!s.spk.empty()) fwrite(s.spk.data(), sizeof(float), s.spk.size(), f);
    }

    // event blocks (iteration order MUST match run_kl_divergence): seq, t, cb
    double tot_nll = 0.0; int64_t tot_cnt = 0, skipped = 0; int n_spk = 0;
    std::vector<float> logits, logp;
    for (const auto & s : seqs) {
        if (!s.spk.empty()) ++n_spk;
        if (!zonos2_logits(m, s.ids.data(), s.n, logits, s.spk_ptr(), s.spk_pos >= 0 ? s.spk_pos : 0)) { fclose(f); return 1; }
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
    printf("wrote %s : %lld events, %zu seqs (%d w/ speaker), ncb=%d, audio_vocab=%d\n", base_path.c_str(),
           (long long) tot_cnt, seqs.size(), n_spk, ncb, av);
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
    uint32_t version = 0, b_ncb = 0, b_av = 0, b_pad = 0, n_seq = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, KLD_MAGIC, 8) != 0) {
        fprintf(stderr, "perplexity: %s: bad magic\n", base_path.c_str()); fclose(f); return 1;
    }
    if (!rd(f, version) || version != KLD_VERSION) {
        fprintf(stderr, "perplexity: %s: version %u != %u (regenerate base)\n", base_path.c_str(), version, KLD_VERSION);
        fclose(f); return 1;
    }
    rd(f, b_ncb); rd(f, b_av); rd(f, b_pad);
    if ((int) b_ncb != ncb || (int) b_av != av) {
        fprintf(stderr, "perplexity: base/model mismatch: base ncb=%u av=%u, model ncb=%d av=%d\n",
                b_ncb, b_av, ncb, av);
        fclose(f); return 1;
    }

    // per-sequence metadata: n, spk_pos, spk_dim (0 => no speaker)
    rd(f, n_seq);
    std::vector<sequence> seqs(n_seq);
    std::vector<uint32_t> seq_spk_dim(n_seq, 0);
    for (uint32_t i = 0; i < n_seq; ++i) {
        uint32_t n = 0, sdim = 0; int32_t spos = -1;
        rd(f, n); rd(f, spos); rd(f, sdim);
        seqs[i].n = (int) n;
        seqs[i].spk_pos = spos;
        seq_spk_dim[i] = sdim;
        if (sdim && sdim != hp.spk_dim) {
            fprintf(stderr, "perplexity: base speaker dim %u != model %u\n", sdim, hp.spk_dim);
            fclose(f); return 1;
        }
    }
    for (uint32_t i = 0; i < n_seq; ++i) {
        sequence & s = seqs[i];
        s.ids.resize((size_t) s.n * W);
        if (fread(s.ids.data(), sizeof(float), s.ids.size(), f) != s.ids.size()) {
            fprintf(stderr, "perplexity: %s: truncated ids block\n", base_path.c_str());
            fclose(f); return 1;
        }
        if (seq_spk_dim[i]) {
            s.spk.resize(seq_spk_dim[i]);
            if (fread(s.spk.data(), sizeof(float), s.spk.size(), f) != s.spk.size()) {
                fprintf(stderr, "perplexity: %s: truncated speaker block\n", base_path.c_str());
                fclose(f); return 1;
            }
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
        if (!zonos2_logits(m, s.ids.data(), s.n, logits, s.spk_ptr(), s.spk_pos >= 0 ? s.spk_pos : 0)) { fclose(f); return 1; }
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
                const std::string & out_path, const std::string & prune_path, int min_hits) {
    const int ne = (int) m.hp.n_expert;
    if (ne == 0) { fprintf(stderr, "imatrix: model has no experts\n"); return 1; }

    struct acc_t { int n_embd = 0, n_ff = 0; std::vector<double> gu, dn; std::vector<int64_t> cnt; };
    std::map<int, acc_t> acc;                                  // per MoE layer

    std::vector<zonos2_moe_act> acts;
    size_t si = 0;
    for (const auto & s : seqs) {
        acts.clear();
        // per-sequence speaker (from --manifest); falls back to the global --speaker that main()
        // copies into every seq for the positional-ids path, or none.
        if (!zonos2_moe_capture(m, s.ids.data(), s.n, acts, s.spk_ptr(), s.spk_pos >= 0 ? s.spk_pos : 0)) return 1;
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
    std::map<int, prune_stats::layer>     ps;                  // per-layer MSAN/hit counts for prune-cli
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

        if (!prune_path.empty()) {
            // MSAN_e = mean over routed tokens of ‖down-input‖² = sum_c A.dn[e,c] / cnt_e.
            // Raw cnt (not the min-hits floor) so rarely-routed experts get their true — and
            // typically lowest — score, which is exactly what flags them as prune candidates.
            prune_stats::layer pl;
            pl.top_k = (kv.first < (int) m.layers.size()) ? m.layers[kv.first].top_k : 0;
            pl.cnt.assign(A.cnt.begin(), A.cnt.end());
            pl.msan.resize(ne);
            for (int e = 0; e < ne; ++e) {
                double s = 0.0;
                for (int c = 0; c < A.n_ff; ++c) s += A.dn[(size_t) e*A.n_ff + c];
                pl.msan[e] = (float) (A.cnt[e] > 0 ? s / (double) A.cnt[e] : 0.0);
            }
            ps[kv.first] = std::move(pl);
        }
    }
    if (!out_path.empty()   && !imatrix::save(out_path, im))      return 1;
    if (!prune_path.empty() && !prune_stats::save(prune_path, ps)) return 1;

    std::sort(all_cnt.begin(), all_cnt.end());
    const int64_t hmin = all_cnt.empty() ? 0 : all_cnt.front();
    const int64_t hmed = all_cnt.empty() ? 0 : all_cnt[all_cnt.size() / 2];
    const int64_t hmax = all_cnt.empty() ? 0 : all_cnt.back();
    printf("\n=== expert importance ===\n");
    printf("%d MoE layers, %d experts, %lld routed (token,expert) hits\n",
           n_layers, ne, (long long) total_hits);
    printf("per-(layer,expert) hits: min %lld, median %lld, max %lld (min-hits floor = %d)\n",
           (long long) hmin, (long long) hmed, (long long) hmax, min_hits);
    if (zero_slots || below_thresh)
        printf("fallback to RTN: %d slots zero-hit, %d below floor (of %d total) — %s\n",
               zero_slots, below_thresh, (int) all_cnt.size(),
               (zero_slots + below_thresh) * 4 > (int) all_cnt.size()
                   ? "consider more/longer generation traces" : "ok, well-covered");
    if (!out_path.empty())   printf("wrote imatrix    %s\n", out_path.c_str());
    if (!prune_path.empty()) printf("wrote prune-stats %s\n", prune_path.c_str());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const std::string model_path = argv[1];

    enum { NONE, PPL, KLBASE, KLDIV, IMATRIX } mode = NONE;
    bool use_gpu = false;
    std::string base_path, prune_path, spk_path, manifest_path;
    std::vector<std::string> ids_paths;
    int spk_pos = 0;
    int imat_min_hits = 32;   // experts seen fewer than this many times fall back to RTN
    std::string prune_mask_path;            // --prune-mask: dynamically drop experts before this run
    int mask_keep = -1, mask_drop_hits = -1, mask_drop_lowest = -1;
    double mask_mass_eps = -1.0;
    std::string skip_layers_arg;            // --skip-layers L1,L2,...: bypass whole transformer blocks

    for (int i = 2; i < argc; ++i) {
        const char * a = argv[i];
        if      (!strcmp(a, "--gpu")) use_gpu = true;
        else if (!strcmp(a, "--cpu")) use_gpu = false;
        else if (!strcmp(a, "--perplexity")) mode = PPL;
        else if (!strcmp(a, "--kl-divergence-base") && i + 1 < argc) { mode = KLBASE; base_path = argv[++i]; }
        else if (!strcmp(a, "--kl-divergence")      && i + 1 < argc) { mode = KLDIV;  base_path = argv[++i]; }
        else if (!strcmp(a, "--imatrix-out")        && i + 1 < argc) { mode = IMATRIX; base_path = argv[++i]; }
        else if (!strcmp(a, "--prune-stats")        && i + 1 < argc) { mode = IMATRIX; prune_path = argv[++i]; }
        else if (!strcmp(a, "--imatrix-min-hits")   && i + 1 < argc) imat_min_hits = atoi(argv[++i]);
        else if (!strcmp(a, "--prune-mask")         && i + 1 < argc) prune_mask_path = argv[++i];
        else if (!strcmp(a, "--keep")               && i + 1 < argc) mask_keep      = atoi(argv[++i]);
        else if (!strcmp(a, "--drop-below-hits")    && i + 1 < argc) mask_drop_hits = atoi(argv[++i]);
        else if (!strcmp(a, "--mass-eps")           && i + 1 < argc) mask_mass_eps  = atof(argv[++i]);
        else if (!strcmp(a, "--drop-lowest")        && i + 1 < argc) mask_drop_lowest = atoi(argv[++i]);
        else if (!strcmp(a, "--skip-layers")        && i + 1 < argc) skip_layers_arg = argv[++i];
        else if (!strcmp(a, "--manifest")    && i + 1 < argc) manifest_path = argv[++i];
        else if (!strcmp(a, "--speaker")     && i + 1 < argc) spk_path = argv[++i];
        else if (!strcmp(a, "--speaker-pos") && i + 1 < argc) spk_pos  = atoi(argv[++i]);
        else if (a[0] != '-') ids_paths.push_back(a); // positional input_ids npy
        else { usage(argv[0]); return 1; }
    }
    if (mode == NONE) { usage(argv[0]); return 1; }
    if (!manifest_path.empty() && mode != KLBASE && mode != IMATRIX) {
        fprintf(stderr, "perplexity: --manifest is only supported with --kl-divergence-base / --imatrix-out\n"); return 1;
    }
    if (mode == PPL && ids_paths.empty()) {
        fprintf(stderr, "perplexity: need at least one <ids.npy>\n"); return 1;
    }
    if ((mode == KLBASE || mode == IMATRIX) && ids_paths.empty() && manifest_path.empty()) {
        fprintf(stderr, "perplexity: %s needs <ids.npy>... or --manifest\n",
                mode == KLBASE ? "--kl-divergence-base" : "--imatrix-out"); return 1;
    }

    zonos2_model model;
    if (!zonos2_model_load(model, model_path.c_str(), use_gpu)) {
        fprintf(stderr, "load failed\n"); return 1;
    }

    // --prune-mask: dynamically drop low-MSAN experts before scoring (runtime equivalent of
    // prune-cli, for sweeping the quality knee against a KLD base without rewriting GGUFs).
    if (!prune_mask_path.empty()) {
        if (mask_keep < 0 && mask_drop_hits < 0 && mask_mass_eps < 0.0 && mask_drop_lowest < 0) {
            fprintf(stderr, "perplexity: --prune-mask needs --keep N, --mass-eps E, --drop-lowest N, or --drop-below-hits H\n");
            zonos2_model_free(model); return 1;
        }
        std::map<int, prune_stats::layer> st;
        if (!prune_stats::load(prune_mask_path, st)) { zonos2_model_free(model); return 1; }
        const auto keep = prune_policy::select(st, { mask_keep, mask_drop_hits, mask_mass_eps, mask_drop_lowest });
        if (!zonos2_set_expert_mask(model, keep)) { zonos2_model_free(model); return 1; }
    }

    // --skip-layers: bypass whole transformer blocks (depth pruning) at graph-build time. Comma-
    // separated block indices; the residual stream passes through each listed block unchanged.
    if (!skip_layers_arg.empty()) {
        model.layer_skip.assign(model.hp.n_layer, 0);
        std::string ls; int n_skip = 0;
        for (size_t p = 0; p <= skip_layers_arg.size(); ++p) {
            const char c = p < skip_layers_arg.size() ? skip_layers_arg[p] : ',';
            if (c == ',') {
                if (!ls.empty()) {
                    const int L = atoi(ls.c_str());
                    if (L < 0 || L >= (int) model.hp.n_layer) {
                        fprintf(stderr, "perplexity: --skip-layers index %d out of range [0,%u)\n",
                                L, model.hp.n_layer);
                        zonos2_model_free(model); return 1;
                    }
                    if (!model.layer_skip[L]) { model.layer_skip[L] = 1; ++n_skip; }
                    ls.clear();
                }
            } else if (c != ' ') ls.push_back(c);
        }
        printf("depth-prune: skipping %d/%u blocks [%s]\n", n_skip, model.hp.n_layer, skip_layers_arg.c_str());
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
    } else if (!manifest_path.empty()) {                 // KLBASE / IMATRIX with per-sequence conditioning
        std::vector<sequence> seqs;
        if (load_manifest(manifest_path, W, model.hp.spk_dim, seqs)) {
            printf("manifest: %zu sequences from %s\n", seqs.size(), manifest_path.c_str());
            rc = (mode == IMATRIX) ? run_imatrix(model, seqs, base_path, prune_path, imat_min_hits)
                                   : run_kl_base(model, base_path, seqs);
        }
    } else {
        std::vector<sequence> seqs(ids_paths.size());
        bool ok = true;
        for (size_t i = 0; i < ids_paths.size() && ok; ++i) {
            ok = load_ids(ids_paths[i], W, seqs[i]);
            if (ok && spk_ptr) { seqs[i].spk = spk; seqs[i].spk_pos = spk_pos; }  // global speaker -> every seq
        }
        if (ok) {
            rc = (mode == PPL)     ? run_perplexity(model, seqs, spk_ptr, spk_pos)
               : (mode == IMATRIX) ? run_imatrix(model, seqs, base_path, prune_path, imat_min_hits)
                                   : run_kl_base(model, base_path, seqs);
        }
    }

    zonos2_model_free(model);
    return rc;
}
