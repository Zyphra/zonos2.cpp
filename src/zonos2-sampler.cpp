#include "zonos2-sampler.h"

#include <algorithm>
#include <cmath>

zonos2_sampler::zonos2_sampler(const zonos2_sampling & s, int n_codebooks, int vocab_)
    : sp(s), rng(s.seed), n_cb(n_codebooks), vocab(vocab_), hist(n_codebooks) {}

void zonos2_sampler::accept(const std::vector<int> & frame) {
    for (int cb = 0; cb < n_cb && cb < (int) frame.size(); ++cb) {
        hist[cb].push_back(frame[cb]);
        while ((int) hist[cb].size() > sp.rep_window) hist[cb].pop_front();
    }
}

int zonos2_sampler::sample(const float * in, int cb) {
    if (sp.greedy) {
        int best = 0; float bv = in[0];
        for (int i = 1; i < vocab; ++i) if (in[i] > bv) { bv = in[i]; best = i; }
        return best;
    }

    std::vector<float> logit(in, in + vocab);

    // repetition penalty over this codebook's recent history (first rep_codebooks only)
    if (sp.rep_penalty != 1.0f && cb < sp.rep_codebooks) {
        for (int t : hist[cb]) {
            if (t >= 0 && t < vocab) {
                logit[t] = logit[t] > 0.0f ? logit[t] / sp.rep_penalty : logit[t] * sp.rep_penalty;
            }
        }
    }

    if (sp.temperature > 0.0f) for (auto & l : logit) l /= sp.temperature;

    // candidate set, optionally restricted to top-k by logit
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    int k = (sp.top_k > 0 && sp.top_k < vocab) ? sp.top_k : vocab;
    if (k < vocab) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return logit[a] > logit[b]; });
        idx.resize(k);
    }

    // softmax over candidates
    float mx = -1e30f;
    for (int i : idx) mx = std::max(mx, logit[i]);
    std::vector<float> p(idx.size());
    double sum = 0;
    for (size_t j = 0; j < idx.size(); ++j) { p[j] = std::exp(logit[idx[j]] - mx); sum += p[j]; }
    for (auto & x : p) x = (float) (x / sum);

    // order by prob desc (for top-p) and find max prob (for min-p)
    std::vector<int> ord(idx.size());
    for (size_t j = 0; j < ord.size(); ++j) ord[j] = (int) j;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return p[a] > p[b]; });

    std::vector<char> keep(idx.size(), 1);
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        double c = 0; bool done = false;
        for (int r : ord) { if (done) keep[r] = 0; c += p[r]; if (c >= sp.top_p) done = true; }
    }
    if (sp.min_p > 0.0f) {
        const float thr = sp.min_p * p[ord[0]];
        for (size_t j = 0; j < p.size(); ++j) if (p[j] < thr) keep[j] = 0;
    }

    double s2 = 0;
    for (size_t j = 0; j < p.size(); ++j) { if (!keep[j]) p[j] = 0; s2 += p[j]; }
    if (s2 <= 0) return idx[ord[0]]; // greedy fallback

    std::uniform_real_distribution<double> u(0.0, s2);
    double r = u(rng), acc = 0;
    for (size_t j = 0; j < p.size(); ++j) { acc += p[j]; if (acc >= r) return idx[j]; }
    return idx[ord[0]];
}
