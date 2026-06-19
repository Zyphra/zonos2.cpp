// zonos2.h — standalone ggml/GGUF port of the ZONOS2 TTS backbone.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct zonos2_hparams {
    uint32_t n_layer      = 0;
    uint32_t n_embd       = 0;
    uint32_t head_dim     = 0;
    uint32_t n_head       = 0;
    uint32_t n_head_kv    = 0;
    uint32_t n_ff         = 0;
    uint32_t n_expert     = 0;
    uint32_t n_ctx_train  = 0;

    float rope_freq_base  = 10000.0f;
    float rms_eps         = 1e-5f;
    float qk_norm_eps     = 1e-6f;
    float logit_softcap   = 15.0f;

    uint32_t n_codebooks  = 0;
    uint32_t codebook_size= 0;
    uint32_t audio_vocab  = 0;   // codebook_size + 2
    uint32_t eoa_id       = 0;
    uint32_t audio_pad_id = 0;
    uint32_t text_vocab   = 0;

    uint32_t spk_dim      = 0;
    uint32_t spk_lda_dim  = 0;

    // conditioning layout for C++ prompt building. Defaults match the Zyphra/ZONOS2
    // release (params.json); overridden by zonos2.cond.* GGUF keys when present.
    uint32_t             cond_speaking_rate_buckets   = 8;
    std::vector<int32_t> cond_quality_bucket_counts   = {12, 12, 12, 8, 8, 8};
    uint32_t             cond_speaker_bg_buckets      = 2;
    uint32_t             cond_accurate_buckets        = 1;
    int32_t              cond_default_quality_feature = 5; // trailing_silence_s
    int32_t              cond_default_quality_bucket  = 3;

    // per layer: 0 => dense FFN, else MoE with this top-k
    std::vector<int32_t> expert_used;
};

struct zonos2_layer {
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * ffn_norm  = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;  // [n_embd, n_head*head_dim]
    struct ggml_tensor * wk        = nullptr;  // [n_embd, n_head_kv*head_dim]
    struct ggml_tensor * wv        = nullptr;  // [n_embd, n_head_kv*head_dim]
    struct ggml_tensor * wo        = nullptr;  // [n_head*head_dim, n_embd]
    struct ggml_tensor * attn_temp = nullptr;  // [n_head], stored as |temp|
    struct ggml_tensor * gater     = nullptr;  // [n_embd, n_head]

    // dense ffn
    struct ggml_tensor * ffn_up    = nullptr;  // [n_embd, n_ff]   (chunk0 of w_in)
    struct ggml_tensor * ffn_gate  = nullptr;  // [n_embd, n_ff]   (chunk1 of w_in, gets SiLU)
    struct ggml_tensor * ffn_down  = nullptr;  // [n_ff, n_embd]

    // moe (sonic): gate=even rows, up=odd rows of w13
    struct ggml_tensor * ffn_gate_exps = nullptr; // [n_embd, n_ff, n_expert]
    struct ggml_tensor * ffn_up_exps   = nullptr; // [n_embd, n_ff, n_expert]
    struct ggml_tensor * ffn_down_exps = nullptr; // [n_ff, n_embd, n_expert]
    struct ggml_tensor * router_down   = nullptr; // [n_embd, router_dim]
    struct ggml_tensor * router_down_b = nullptr; // [router_dim]
    struct ggml_tensor * router_mlp0   = nullptr; // [router_dim, router_dim]
    struct ggml_tensor * router_mlp0_b = nullptr;
    struct ggml_tensor * router_mlp2   = nullptr; // [router_dim, router_dim]
    struct ggml_tensor * router_mlp2_b = nullptr;
    struct ggml_tensor * router_mlp4   = nullptr; // [router_dim, n_expert]
    struct ggml_tensor * router_norm   = nullptr; // [router_dim]
    struct ggml_tensor * router_bias   = nullptr; // [n_expert]
    struct ggml_tensor * router_eda_scale = nullptr; // [router_dim], null on first MoE layer

    bool is_moe = false;
    int  top_k  = 0;
};

struct zonos2_model {
    zonos2_hparams hp;

    std::vector<struct ggml_tensor *> audio_embd; // [n_codebooks], each [n_embd, audio_vocab]
    struct ggml_tensor * text_embd   = nullptr;   // [n_embd, text_vocab+1]
    struct ggml_tensor * spk_lda_w   = nullptr;   // [spk_dim, spk_lda_dim]
    struct ggml_tensor * spk_lda_b   = nullptr;
    struct ggml_tensor * spk_proj_w  = nullptr;   // [spk_lda_dim, n_embd]
    struct ggml_tensor * spk_proj_b  = nullptr;
    struct ggml_tensor * output_norm = nullptr;   // [n_embd]
    struct ggml_tensor * output      = nullptr;   // [n_embd, audio_vocab*n_codebooks]

    std::vector<zonos2_layer> layers;

    // backend / storage
    struct ggml_context *      ctx_w   = nullptr;  // weight tensor metadata (from gguf)
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_type_t buft    = nullptr;
    ggml_backend_buffer_t      buf_w   = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
    bool is_gpu = false;
};

// Load a zonos2 GGUF onto the CPU or first GPU backend. Returns false on error.
bool zonos2_model_load(zonos2_model & model, const char * path, bool use_gpu);
void zonos2_model_free(zonos2_model & model);

// router hidden dimension, derived from the router_down weight (ne1).
int zonos2_router_dim(const zonos2_model & model);

// Run a single prefill forward over `ids` (row-major [n_tokens, n_codebooks+1], values
// as floats), capturing named intermediate tensors as <out_dir>/<name>.npy for numeric
// validation against the PyTorch golden dump. `n_layer_limit` < 0 builds all layers.
// `spk` (optional, [spk_dim] f32) injects a speaker embedding: the column at `spk_pos`
// is overwritten by spk_proj(spk_lda(spk)) before emb_norm. spk == nullptr disables it.
bool zonos2_validate(const zonos2_model & model, const float * ids, int n_tokens,
                     const char * out_dir, int n_layer_limit,
                     const float * spk = nullptr, int spk_pos = 0);

// Run a single prefill forward over `ids` (row-major [n_tokens, n_codebooks+1], values as
// floats) and copy the full post-softcap logits into `out_logits`, resized to
// n_tokens*n_codebooks*audio_vocab in C-order [n_tokens, n_codebooks, audio_vocab]: the value
// for (position t, codebook cb, vocab v) lives at ((size_t)t*n_codebooks + cb)*audio_vocab + v.
// `spk`/`spk_pos` inject a speaker embedding exactly as in zonos2_validate. Returns false on
// failure. NB: prefill self-attention is O(n^2) in memory; keep n within n_ctx_train and split
// long corpora into multiple sequences rather than one giant prefill.
bool zonos2_logits(const zonos2_model & model, const float * ids, int n_tokens,
                   std::vector<float> & out_logits,
                   const float * spk = nullptr, int spk_pos = 0);

// Options for building a TTS prompt from text (mirrors zonos2/tts/prompt.py +
// scheduler speaker frames). Defaults reproduce the reference offline prompt.
struct zonos2_prompt_options {
    bool             prepend_silence        = true; // append 17 sheared-silence frames
    int              speaking_rate_bucket   = -1;   // -1 = no speaking-rate row
    std::vector<int> quality_buckets;               // per-feature bucket (-1 = skip);
                                                    // empty => default {trailing_silence_s: 3}
    bool             add_speaker_slot        = false; // prepend speaker slot (+markers) for cloning
    bool             clean_speaker_background = true;
    bool             accurate_mode           = true;
};

// Build a TTS prompt as row-major input_ids [n_rows, n_codebooks+1] (int32) for `text`.
// Sets n_rows; if a speaker slot was prepended, sets spk_pos to its index (else -1).
std::vector<int32_t> zonos2_build_prompt(const zonos2_model & model, const std::string & text,
                                         const zonos2_prompt_options & opt,
                                         int & n_rows, int & spk_pos);

struct zonos2_sampling {
    float    temperature   = 1.15f;
    int      top_k         = 106;
    float    top_p         = 0.0f;   // 0 disables
    float    min_p         = 0.18f;
    float    rep_penalty   = 1.2f;
    int      rep_window    = 50;
    int      rep_codebooks = 8;
    uint32_t seed          = 0;
    bool     greedy        = false;
};

// Incremental KV cache: per-layer K/V buffers persisted across decode steps.
struct zonos2_context {
    const zonos2_model * model = nullptr;
    int max_seq = 0;
    int n_past  = 0;
    struct ggml_context *      ctx_kv  = nullptr;
    ggml_backend_buffer_t      buf_kv  = nullptr;
    std::vector<struct ggml_tensor *> k_cache; // per layer [head_dim, n_head_kv, max_seq]
    std::vector<struct ggml_tensor *> v_cache;
    ggml_gallocr_t galloc = nullptr;           // reused across prefill + decode steps
};

bool zonos2_context_init(zonos2_context & ctx, const zonos2_model & model, int max_seq);
void zonos2_context_free(zonos2_context & ctx);

// Autoregressive generation. use_kv=true: one prefill + single-token decodes with a KV
// cache (real-time, O(n)). use_kv=false: recompute the full prefill each step (O(n^2),
// reference path). prompt_ids: row-major [n0, n_codebooks+1] (floats). Appends generated
// audio codes to out_codes (flattened [n_frames * n_codebooks]); sets eos_frame. Returns n_frames.
// `spk` (optional, [spk_dim] f32) clones a voice: during prefill the embedding column at
// `spk_pos` (default 0, the prompt's speaker slot) is overwritten by spk_proj(spk_lda(spk)).
int zonos2_generate(const zonos2_model & model, const float * prompt_ids, int n0,
                    int max_frames, const zonos2_sampling & sp,
                    std::vector<int32_t> & out_codes, int & eos_frame, bool use_kv = true,
                    const float * spk = nullptr, int spk_pos = 0);
