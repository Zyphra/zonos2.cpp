// Per-codebook sampler: repetition penalty, temperature, top-k, top-p, min-p, multinomial.
#pragma once

#include "zonos2.h"

#include <deque>
#include <random>
#include <vector>

struct zonos2_sampler {
    zonos2_sampling sp;
    std::mt19937 rng;
    int n_cb;
    int vocab;
    std::vector<std::deque<int>> hist; // recent tokens per codebook (for rep penalty)

    zonos2_sampler(const zonos2_sampling & s, int n_codebooks, int vocab_);
    int  sample(const float * logits, int cb);          // logits length == vocab
    void accept(const std::vector<int> & frame);        // push a sampled frame into history
};
