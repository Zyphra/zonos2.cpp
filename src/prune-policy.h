// prune-policy.h — turn prune-stats into a per-layer keep-set.
//
// Shared by prune-cli (which bakes the keep-set into a smaller GGUF) and the runtime --prune-mask
// path (which applies it via zonos2_set_expert_mask) so a mask sweep and the final prune agree
// exactly. Header-only, depends only on prune-stats.h.
#pragma once

#include "prune-stats.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

namespace prune_policy {

struct policy {
    int    keep_n          = -1;   // keep the N highest-MSAN experts per layer (uniform budget)
    int    drop_below_hits = -1;   // (alt) keep experts routed at least this many times
    double mass_eps        = -1.0; // (alt) keep experts whose share of the layer's total MSAN >= eps
    int    drop_lowest     = -1;   // (alt) drop the N lowest-MSAN experts across ALL layers (global)
};

// Per-layer ascending kept expert ids. Experts are ranked by MSAN (then hit count, then id) and
// the layer never drops below its top-k. Only layers that actually lose an expert are returned —
// a layer absent from the result keeps all its experts.
//
// Modes (first set one wins): keep_n (uniform N/layer) > mass_eps (relative MSAN-mass threshold,
// per-layer non-uniform — prunes peaky deep layers hard while protecting flat early layers) >
// drop_below_hits.
inline std::map<int, std::vector<int>> select(const std::map<int, prune_stats::layer> & stats,
                                              const policy & p) {
    std::map<int, std::vector<int>> keep;

    // Global mode: drop the N lowest-MSAN experts across ALL layers at once (subject to each
    // layer's top-k floor). Measures the marginal cost of removing the most-droppable experts
    // anywhere, rather than a fixed budget per layer.
    if (p.drop_lowest >= 0) {
        struct cell { int L, e; float msan; int64_t cnt; };
        std::vector<cell> all;
        for (const auto & kv : stats)
            for (int e = 0; e < (int) kv.second.msan.size(); ++e)
                all.push_back({ kv.first, e, kv.second.msan[e], kv.second.cnt[e] });
        std::sort(all.begin(), all.end(), [](const cell & a, const cell & b) { // lowest first
            if (a.msan != b.msan) return a.msan < b.msan;
            if (a.cnt  != b.cnt)  return a.cnt  < b.cnt;
            return a.L < b.L;
        });
        std::map<int, std::vector<char>> dropped;                    // [L] which experts removed
        int budget = p.drop_lowest;
        for (const cell & c : all) {
            if (budget <= 0) break;
            const prune_stats::layer & pl = stats.at(c.L);
            const int ne = (int) pl.msan.size();
            const int k  = pl.top_k > 0 ? pl.top_k : 1;
            auto & dl = dropped[c.L];
            if (dl.empty()) dl.assign(ne, 0);
            int still = ne; for (char x : dl) if (x) --still;
            if (still <= k) continue;                                // honor this layer's top-k floor
            dl[c.e] = 1; --budget;
        }
        for (const auto & kv : dropped) {
            std::vector<int> ks;
            for (int e = 0; e < (int) kv.second.size(); ++e) if (!kv.second[e]) ks.push_back(e);
            if ((int) ks.size() != (int) stats.at(kv.first).msan.size()) keep[kv.first] = std::move(ks);
        }
        return keep;
    }

    for (const auto & kv : stats) {
        const prune_stats::layer & pl = kv.second;
        const int ne = (int) pl.cnt.size();
        const int k  = pl.top_k > 0 ? pl.top_k : 1;
        if (ne == 0) continue;

        std::vector<int> idx(ne);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {        // best (highest MSAN) first
            if (pl.msan[a] != pl.msan[b]) return pl.msan[a] > pl.msan[b];
            if (pl.cnt[a]  != pl.cnt[b])  return pl.cnt[a]  > pl.cnt[b];
            return a < b;
        });

        int want;
        if (p.keep_n >= 0) {
            want = p.keep_n;
        } else if (p.mass_eps >= 0.0) {                              // keep experts with MSAN share >= eps
            double total = 0.0;
            for (int e = 0; e < ne; ++e) total += pl.msan[e];
            want = 0;
            if (total > 0.0)
                for (int e = 0; e < ne; ++e) if (pl.msan[e] / total >= p.mass_eps) ++want;
        } else {                                                     // drop-below-hits: keep cnt >= H
            want = 0;
            for (int e = 0; e < ne; ++e) if (pl.cnt[e] >= (int64_t) p.drop_below_hits) ++want;
        }
        if (want < k)  want = k;                                     // floor: top-k needs k experts
        if (want > ne) want = ne;
        if (want == ne) continue;                                    // nothing to drop in this layer

        std::vector<int> ks(idx.begin(), idx.begin() + want);
        std::sort(ks.begin(), ks.end());                             // ascending for determinism
        keep[kv.first] = std::move(ks);
    }
    return keep;
}

} // namespace prune_policy
