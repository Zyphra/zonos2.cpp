// zonos2-cli — load a ZONOS2 GGUF; print summary or run prefill validation.
#include "zonos2.h"
#include "dac.h"
#include "npy.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--cpu|--gpu]\n"
        "       %s <model.gguf> --validate <input_ids.npy> <out_dir> [--speaker <spk.npy>] [--cpu|--gpu]\n"
        "       %s <model.gguf> --generate <input_ids.npy> <out.{npy,wav}> [--dac <dac.gguf>] [--speaker <spk.npy>] ...\n"
        "       %s <model.gguf> --tts \"<text>\" <out.{npy,wav}> [--dac <dac.gguf>] [--speaker <spk.npy>] [--greedy] [--max N] ...\n"
        "       %s <model.gguf> --build-prompt \"<text>\" <out_ids.npy> [--speaker <spk.npy>]\n"
        "  (with --dac, a .wav output is decoded directly; an .npy output also writes a sibling .wav)\n",
        a0, a0, a0, a0, a0);
}

static bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Persist generated codes: codes npy (+ eos) and/or a decoded wav when --dac is set.
static void emit_output(const std::string & out, const std::vector<int32_t> & codes,
                        int n_frames, int ncb, int eos_frame,
                        const std::string & dac_path, bool use_gpu) {
    const bool want_wav = ends_with(out, ".wav");
    if (!want_wav) {
        std::vector<float> cf(codes.begin(), codes.end());
        npy::save_f32(out, cf.data(), { (int64_t) n_frames, (int64_t) ncb });
        const float ef = (float) eos_frame;
        npy::save_f32(out + ".eos.npy", &ef, { (int64_t) 1 });
        printf("wrote %s\n", out.c_str());
    }
    if (want_wav && dac_path.empty()) {
        fprintf(stderr, "error: .wav output requested but no --dac <dac.gguf> given\n");
        return;
    }
    if (!dac_path.empty() && n_frames > 0) {
        const std::string wav = want_wav ? out : out + ".wav";
        dac_model dm;
        if (dac_load(dm, dac_path.c_str(), use_gpu)) {
            std::vector<float> audio;
            if (dac_decode(dm, codes.data(), n_frames, ncb, eos_frame, audio))
                dac_write_wav(wav.c_str(), audio, dm.sample_rate);
            dac_free(dm);
        }
    }
}

static void pr(const char * nm, const ggml_tensor * t) {
    if (!t) { printf("  %-22s <null>\n", nm); return; }
    printf("  %-22s ne=[%5lld,%5lld,%4lld,%2lld] %s\n", nm,
           (long long) t->ne[0], (long long) t->ne[1],
           (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const std::string path = argv[1];
    bool use_gpu = false;
    bool validate = false, generate = false, do_tts = false, do_build_prompt = false;
    std::string ids_path, out_dir, out_codes, spk_path, text, prompt_out, dac_path;
    int n_layer_limit = -1, max_frames = 400, spk_pos = 0;
    bool use_kv = true;
    zonos2_sampling sp;
    for (int i = 2; i < argc; ++i) {
        if      (!strcmp(argv[i], "--gpu")) use_gpu = true;
        else if (!strcmp(argv[i], "--cpu")) use_gpu = false;
        else if (!strcmp(argv[i], "--validate") && i + 2 < argc) {
            validate = true; ids_path = argv[++i]; out_dir = argv[++i];
        }
        else if (!strcmp(argv[i], "--generate") && i + 2 < argc) {
            generate = true; ids_path = argv[++i]; out_codes = argv[++i];
        }
        else if (!strcmp(argv[i], "--tts") && i + 2 < argc) {
            do_tts = true; text = argv[++i]; out_codes = argv[++i];
        }
        else if (!strcmp(argv[i], "--build-prompt") && i + 2 < argc) {
            do_build_prompt = true; text = argv[++i]; prompt_out = argv[++i];
        }
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) n_layer_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max")    && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dac") && i + 1 < argc) dac_path = argv[++i];
        else if (!strcmp(argv[i], "--speaker") && i + 1 < argc) spk_path = argv[++i];
        else if (!strcmp(argv[i], "--speaker-pos") && i + 1 < argc) spk_pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")   && i + 1 < argc) sp.seed = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) sp.greedy = true;
        else if (!strcmp(argv[i], "--recompute")) use_kv = false;
        else { usage(argv[0]); return 1; }
    }

    zonos2_model model;
    if (!zonos2_model_load(model, path.c_str(), use_gpu)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    // optional speaker embedding ([spk_dim] or [1, spk_dim] f32) for voice cloning
    std::vector<float> spk;
    if (!spk_path.empty()) {
        std::vector<int64_t> sshape;
        if (!npy::load_f32(spk_path, spk, sshape)) {
            fprintf(stderr, "failed to load speaker npy %s\n", spk_path.c_str());
            zonos2_model_free(model); return 1;
        }
        if (spk.size() != model.hp.spk_dim) {
            fprintf(stderr, "speaker dim mismatch: got %zu, want %u\n", spk.size(), model.hp.spk_dim);
            zonos2_model_free(model); return 1;
        }
        printf("speaker: %zu-d vector from %s, pos=%d\n", spk.size(), spk_path.c_str(), spk_pos);
    }
    const float * spk_ptr = spk.empty() ? nullptr : spk.data();

    if (validate) {
        std::vector<float> ids;
        std::vector<int64_t> shape;
        if (!npy::load_f32(ids_path, ids, shape) || shape.size() != 2) {
            fprintf(stderr, "validate: bad input_ids npy (need 2-D)\n");
            zonos2_model_free(model); return 1;
        }
        const int n_tokens = (int) shape[0];
        printf("validate: input_ids [%lld, %lld], n_tokens=%d\n",
               (long long) shape[0], (long long) shape[1], n_tokens);
        const bool okv = zonos2_validate(model, ids.data(), n_tokens, out_dir.c_str(), n_layer_limit, spk_ptr, spk_pos);
        zonos2_model_free(model);
        return okv ? 0 : 1;
    }

    if (generate) {
        std::vector<float> ids;
        std::vector<int64_t> shape;
        if (!npy::load_f32(ids_path, ids, shape) || shape.size() != 2) {
            fprintf(stderr, "generate: bad prompt npy (need 2-D)\n");
            zonos2_model_free(model); return 1;
        }
        const int n0 = (int) shape[0];
        printf("generate: prompt [%d, %lld], max=%d, %s, seed=%u, %s\n",
               n0, (long long) shape[1], max_frames, sp.greedy ? "greedy" : "sampling", sp.seed,
               use_kv ? "kv-cache" : "recompute");
        std::vector<int32_t> codes;
        int eos_frame = -1;
        const int n_frames = zonos2_generate(model, ids.data(), n0, max_frames, sp, codes, eos_frame, use_kv, spk_ptr, spk_pos);
        const int ncb = (int) model.hp.n_codebooks;
        printf("generate: %d frames, eos_frame=%d\n", n_frames, eos_frame);
        if (n_frames > 0) {
            printf("  frame0:");
            for (int cb = 0; cb < ncb; ++cb) printf(" %d", codes[cb]);
            printf("\n");
        }
        emit_output(out_codes, codes, n_frames, ncb, eos_frame, dac_path, use_gpu);
        zonos2_model_free(model);
        return 0;
    }

    if (do_build_prompt) {
        zonos2_prompt_options opt;
        if (spk_ptr) opt.add_speaker_slot = true; // prepend speaker slot for cloning
        int n_rows = 0, sp_pos = -1;
        std::vector<int32_t> ids = zonos2_build_prompt(model, text, opt, n_rows, sp_pos);
        const int W = (int) model.hp.n_codebooks + 1;
        std::vector<float> idf(ids.begin(), ids.end());
        npy::save_f32(prompt_out, idf.data(), { (int64_t) n_rows, (int64_t) W });
        printf("build-prompt: \"%s\" -> %d rows x %d, spk_pos=%d -> %s\n",
               text.c_str(), n_rows, W, sp_pos, prompt_out.c_str());
        zonos2_model_free(model);
        return 0;
    }

    if (do_tts) {
        zonos2_prompt_options opt;
        if (spk_ptr) opt.add_speaker_slot = true;
        int n0 = 0, sp_pos = -1;
        std::vector<int32_t> ids = zonos2_build_prompt(model, text, opt, n0, sp_pos);
        std::vector<float> idf(ids.begin(), ids.end());
        const int ncb = (int) model.hp.n_codebooks;
        printf("tts: \"%s\" -> prompt [%d, %d], spk_pos=%d, max=%d, %s, %s\n",
               text.c_str(), n0, ncb + 1, sp_pos, max_frames,
               sp.greedy ? "greedy" : "sampling", use_kv ? "kv-cache" : "recompute");
        std::vector<int32_t> codes;
        int eos_frame = -1;
        const int n_frames = zonos2_generate(model, idf.data(), n0, max_frames, sp, codes,
                                             eos_frame, use_kv, spk_ptr, sp_pos >= 0 ? sp_pos : 0);
        printf("tts: %d frames, eos_frame=%d\n", n_frames, eos_frame);
        if (n_frames > 0) {
            printf("  frame0:");
            for (int cb = 0; cb < ncb; ++cb) printf(" %d", codes[cb]);
            printf("\n");
        }
        emit_output(out_codes, codes, n_frames, ncb, eos_frame, dac_path, use_gpu);
        zonos2_model_free(model);
        return 0;
    }

    const auto & hp = model.hp;
    printf("\n=== ZONOS2 model ===\n");
    printf("layers=%u  embd=%u  head_dim=%u  heads=%u/%u  ff=%u  experts=%u  ctx=%u  router_dim=%d\n",
           hp.n_layer, hp.n_embd, hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_ff,
           hp.n_expert, hp.n_ctx_train, zonos2_router_dim(model));
    printf("rope_base=%.1f  rms_eps=%.2e  qk_eps=%.2e  softcap=%.1f\n",
           hp.rope_freq_base, hp.rms_eps, hp.qk_norm_eps, hp.logit_softcap);
    printf("codebooks=%u  cb_size=%u  audio_vocab=%u  eoa=%u  pad=%u  text_vocab=%u\n",
           hp.n_codebooks, hp.codebook_size, hp.audio_vocab, hp.eoa_id, hp.audio_pad_id, hp.text_vocab);
    printf("speaker: dim=%u  lda=%u\n", hp.spk_dim, hp.spk_lda_dim);

    int n_moe = 0, n_dense = 0, n_eda = 0, n_top2 = 0;
    for (const auto & ly : model.layers) {
        if (ly.is_moe) { n_moe++; if (ly.router_eda_scale) n_eda++; if (ly.top_k == 2) n_top2++; }
        else n_dense++;
    }
    printf("dense=%d  moe=%d (eda=%d, top2=%d)\n", n_dense, n_moe, n_eda, n_top2);

    printf("spot-check tensors:\n");
    pr("audio_embd[0]",  model.audio_embd[0]);
    pr("text_embd",      model.text_embd);
    pr("spk_lda.w",      model.spk_lda_w);
    pr("spk_proj.w",     model.spk_proj_w);
    pr("output",         model.output);
    pr("output_norm",    model.output_norm);
    pr("blk0.attn_q",    model.layers[0].wq);
    pr("blk0.attn_k",    model.layers[0].wk);
    pr("blk0.attn_temp", model.layers[0].attn_temp);
    pr("blk0.gater",     model.layers[0].gater);
    pr("blk0.ffn_gate",  model.layers[0].ffn_gate);
    pr("blk3.gate_exps", model.layers[3].ffn_gate_exps);
    pr("blk3.down_exps", model.layers[3].ffn_down_exps);
    pr("blk3.router_mlp4", model.layers[3].router_mlp4);
    pr("blk3.eda_scale", model.layers[3].router_eda_scale); // expect <null>
    pr("blk4.eda_scale", model.layers[4].router_eda_scale);

    zonos2_model_free(model);
    return 0;
}
