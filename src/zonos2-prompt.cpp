// zonos2-prompt.cpp — build TTS prompt input_ids from raw text in C++.
// Mirrors python/zonos2/tts/prompt.py (TTSPromptBuilder) and the scheduler's
// _with_speaker_frames. Output matches the reference's offline prompt byte-for-byte.
#include "zonos2.h"

#include <string>
#include <vector>

namespace {

// Tokenizer constants (prompt.py): PAD/UNK/BOS/EOS = 0/1/2/3; 192 legacy symbol ids
// precede the 256 byte ids, so a UTF-8 byte b maps to id (b + 192).
constexpr int BOS_ID = 2;
constexpr int EOS_ID = 3;
constexpr int LEGACY_SYMBOL_VOCAB_SIZE = 192;

// Pre-computed silence tokens for 0.2s @ 44.1kHz, 17 frames x 9 codebooks
// (prompt.py _SILENCE_TOKENS_0_2S). Rows 2..15 are identical.
const int SILENCE_0_2S[17][9] = {
    {568, 778, 338, 524, 967, 360, 728, 550,  90},
    {568, 778,  10, 674, 364, 981, 741, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 778, 721, 842, 264, 974, 989, 507, 308},
};

// First conditioning token id: everything below it is normal text vocabulary.
// Conditioning occupies the tail of text_vocab: speaking-rate, quality (per feature),
// speaker-background (clean,noisy), accurate-mode; text_vocab itself is padding.
int base_text_vocab(const zonos2_hparams & hp, int & sum_quality) {
    sum_quality = 0;
    for (int c : hp.cond_quality_bucket_counts) sum_quality += c;
    return (int) hp.text_vocab
         - (int) hp.cond_speaking_rate_buckets
         - sum_quality
         - (int) hp.cond_speaker_bg_buckets
         - (int) hp.cond_accurate_buckets;
}

} // namespace

std::vector<int32_t> zonos2_build_prompt(const zonos2_model & model, const std::string & text,
                                         const zonos2_prompt_options & opt,
                                         int & n_rows, int & spk_pos) {
    const zonos2_hparams & hp = model.hp;
    const int ncb = (int) hp.n_codebooks;
    const int W   = ncb + 1;
    const int pad = (int) hp.audio_pad_id;
    const int tv  = (int) hp.text_vocab;

    int sum_q = 0;
    const int base = base_text_vocab(hp, sum_q);

    std::vector<std::vector<int32_t>> rows;
    auto pad_row = [&](int text_tok) {
        std::vector<int32_t> r(W, pad);
        r[ncb] = text_tok;
        return r;
    };

    spk_pos = -1;

    // --- speaker slot + marker frames (cloning); mirrors scheduler._with_speaker_frames ---
    if (opt.add_speaker_slot) {
        rows.push_back(pad_row(tv)); // speaker slot: audio pad x9, text = text_vocab
        spk_pos = 0;
        if (hp.cond_speaker_bg_buckets > 0) {
            const int bg = base + (int) hp.cond_speaking_rate_buckets + sum_q
                         + (opt.clean_speaker_background ? 0 : 1);
            rows.push_back(pad_row(bg));
            if (hp.cond_accurate_buckets > 0 && opt.accurate_mode) {
                const int acc = base + (int) hp.cond_speaking_rate_buckets + sum_q
                              + (int) hp.cond_speaker_bg_buckets;
                rows.push_back(pad_row(acc));
            }
        }
    }

    // --- speaking-rate row ---
    if (opt.speaking_rate_bucket >= 0) {
        rows.push_back(pad_row(base + opt.speaking_rate_bucket));
    }

    // --- quality rows (default: trailing_silence_s feature -> bucket 3) ---
    std::vector<int> qb = opt.quality_buckets;
    if (qb.empty()) {
        qb.assign(hp.cond_quality_bucket_counts.size(), -1);
        const int f = hp.cond_default_quality_feature;
        if (f >= 0 && f < (int) qb.size()) qb[f] = hp.cond_default_quality_bucket;
    }
    int qprefix = 0; // sum of bucket counts for earlier features
    for (size_t f = 0; f < hp.cond_quality_bucket_counts.size(); ++f) {
        if (f < qb.size() && qb[f] >= 0) {
            rows.push_back(pad_row(base + (int) hp.cond_speaking_rate_buckets + qprefix + qb[f]));
        }
        qprefix += hp.cond_quality_bucket_counts[f];
    }

    // --- text: BOS + (byte + 192) ... + EOS ---
    rows.push_back(pad_row(BOS_ID));
    for (unsigned char ch : text) rows.push_back(pad_row(LEGACY_SYMBOL_VOCAB_SIZE + (int) ch));
    rows.push_back(pad_row(EOS_ID));

    // --- sheared silence: out[t][c] = silence[t-c][c] if t>=c else pad ---
    if (opt.prepend_silence) {
        for (int t = 0; t < 17; ++t) {
            std::vector<int32_t> r(W, pad);
            for (int c = 0; c < ncb && c < 9; ++c) {
                if (t >= c) r[c] = SILENCE_0_2S[t - c][c];
            }
            r[ncb] = tv;
            rows.push_back(r);
        }
    }

    n_rows = (int) rows.size();
    std::vector<int32_t> out;
    out.reserve((size_t) n_rows * W);
    for (const auto & r : rows) out.insert(out.end(), r.begin(), r.end());
    return out;
}
