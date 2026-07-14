// zonos2-cli — load a ZONOS2 GGUF; print summary or run prefill validation.
#include "zonos2.h"
#include "zonos2-emotion.h"
#include "zonos2-sampler.h"
#include "dac.h"
#include "spk-encoder.h"
#include "npy.h"
#include "ggml.h"
#include "prune-stats.h"
#include "prune-policy.h"
#include "model-paths.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <vector>

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--cpu|--gpu]\n"
        "       %s <model.gguf> --validate <input_ids.npy> <out_dir> [--speaker <spk.npy>] [--cpu|--gpu]\n"
        "       %s <model.gguf> --generate <input_ids.npy> <out.{npy,wav}> [--dac <dac.gguf>] [--speaker <spk.npy>] ...\n"
        "       %s <model.gguf> --tts \"<text>\" <out.{npy,wav}> [--dac <dac.gguf>] [--speaker <spk.npy>] [--greedy] [--max N] ...\n"
        "       %s <model.gguf> --build-prompt \"<text>\" <out_ids.npy> [--speaker <spk.npy>]\n"
        "       %s <model.gguf> --batch-test \"<t1|t2|...>\" [--slots N] [--max N] [--greedy] [--dac <dac.gguf>] --gpu\n"
        "  (with --dac, a .wav output is decoded directly; an .npy output also writes a sibling .wav)\n"
        "  (--dac / --spk-encoder default to dac.gguf / spk-encoder.gguf next to <model.gguf> when present)\n"
        "  (--dump-ids <ids.npy> on --generate/--tts writes the full teacher-forcing sequence for imatrix calibration)\n"
        "  (--prune-mask <stats.bin> (--keep N | --mass-eps E | --drop-below-hits H): audition a pruned recipe by ear before baking with prune-cli)\n"
        "  conditioning paths (--tts/--build-prompt): [--inaccurate] [--noisy-bg] [--speaking-rate N] [--quality f:b[,f:b...]]\n"
        "  emotion control (--tts/--generate): [--emotion happy=1[,sad=-0.5]] [--emotion-valence X] [--emotion-arousal X]\n"
        "                                      [--emotion-strength X] [--emotion-cfg-scale X] [--emotion-dir DIR]\n"
        "  voice cloning: --speaker <spk.npy> (precomputed) OR --clone <ref_audio> --spk-encoder|--spk <spk-encoder.gguf>\n"
        "                 (--clone encodes the reference in-process via ffmpeg+ECAPA; add --save-speaker <out.npy> to cache it)\n",
        a0, a0, a0, a0, a0, a0);
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
        fprintf(stderr, "error: .wav output requested but no --dac <dac.gguf> given (also auto-detected next to <model.gguf>)\n");
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

static std::vector<std::string> split_pipe(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        size_t p = s.find('|', start);
        out.push_back(s.substr(start, p == std::string::npos ? std::string::npos : p - start));
        if (p == std::string::npos) break;
        start = p + 1;
    }
    return out;
}

static bool parse_emotion_spec(const std::string & s, std::map<std::string, float> & out) {
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t eq = tok.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= tok.size()) return false;
        out[tok.substr(0, eq)] = (float) atof(tok.substr(eq + 1).c_str());
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return true;
}

// Raw decode-throughput probe: prefill `n_slots` identical sequences, warm up the CUDA graph, then
// time `n_steps` batched decode steps, ignoring EOS (tokens are fed back but generation never
// stops). Measures the decode-graph rate independent of model correctness — usable on quants that
// won't generate naturally (e.g. q4_k emits EOS at frame 0). Reports aggregate frames/s.
static int run_decode_bench(zonos2_model & model, const std::string & text, int n_slots, int n_steps) {
    const int ncb = (int) model.hp.n_codebooks, av = (int) model.hp.audio_vocab;
    if (n_slots < 1) n_slots = 1;

    zonos2_prompt_options opt; int n0 = 0, sp_pos = -1;
    std::vector<int32_t> ids = zonos2_build_prompt(model, text, opt, n0, sp_pos);
    std::vector<float> idf(ids.begin(), ids.end());

    zonos2_batch_ctx bc;
    if (!zonos2_batch_init(bc, model, n_slots, n0 + n_steps + 8)) {
        fprintf(stderr, "decode-bench: batch init failed\n"); return 1;
    }
    zonos2_sampling sp; sp.greedy = true;            // argmax feedback; values don't affect timing
    std::vector<zonos2_sampler> smp(n_slots, zonos2_sampler(sp, ncb, av));
    std::vector<zonos2_slot> slots(n_slots);
    std::vector<std::vector<float>> lo(n_slots, std::vector<float>((size_t) av * ncb));
    std::vector<int32_t> sink;                        // throwaway codes
    std::vector<zonos2_slot *> active(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        slots[i] = zonos2_slot{}; slots[i].index = i; slots[i].active = true; slots[i].n_past = n0;
        if (!zonos2_batch_slot_prefill(bc, i, idf.data(), n0, nullptr, 0, lo[i].data())) {
            fprintf(stderr, "decode-bench: prefill slot %d failed\n", i); zonos2_batch_free(bc); return 1;
        }
        active[i] = &slots[i];
    }
    auto step_once = [&]() {
        for (int i = 0; i < n_slots; ++i) {          // sample -> next_ids (ignore done/eos)
            zonos2_slot_sample(model, slots[i], smp[i], lo[i].data(), sink, n_steps + n0 + 16, {});
            slots[i].done = false;
        }
        zonos2_batch_step(bc, active);
        for (int i = 0; i < n_slots; ++i)
            memcpy(lo[i].data(), zonos2_batch_slot_logits(bc, i), (size_t) av * ncb * sizeof(float));
    };

    for (int w = 0; w < 8; ++w) step_once();         // warm up CUDA graph
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) step_once();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double fps = (double) n_slots * n_steps / dt;
    const double rt  = fps / (44100.0 / 512.0);
    printf("decode-bench: slots=%d steps=%d  %.3fs  %.0f frames/s  %.1fx real-time  (%.1f frames/s/slot)\n",
           n_slots, n_steps, dt, fps, rt, fps / n_slots);
    zonos2_batch_free(bc);
    return 0;
}

// Batched-decode validation harness. `texts` is '|'-separated; each becomes one active slot (extra
// slots up to --slots stay idle, exercising idle-slot safety). For every request it (1) runs the
// single-sequence reference (zonos2_generate) and (2) runs the same prompt through the batch
// runtime, then compares. With --greedy and --slots 1 the batch result must be bit-identical to the
// reference (same VEC kernel); for n_slots>1 the decode routes through the MMA kernel so exact
// agreement is reported per frame. Identical prompts in two slots must always match exactly (F.2).
static int run_batch_test(zonos2_model & model, const std::string & texts_joined, int n_slots,
                          int max_frames, const zonos2_sampling & sp_in,
                          const std::string & dac_path, bool use_gpu) {
    const int ncb = (int) model.hp.n_codebooks, av = (int) model.hp.audio_vocab;
    std::vector<std::string> texts = split_pipe(texts_joined);
    const int n_req = (int) texts.size();
    if (n_slots < n_req) n_slots = n_req;

    struct Job {
        std::string text;
        std::vector<float> idf;          // prompt ids [n0*W] floats
        int n0 = 0;
        std::unique_ptr<zonos2_sampler> smp;
        zonos2_slot slot;
        std::vector<float> logits;       // [av*ncb] current sampling source
        std::vector<int32_t> codes;
        int eos_frame = -1;
    };
    std::vector<Job> jobs(n_req);
    int max_n0 = 0;
    for (int i = 0; i < n_req; ++i) {
        zonos2_prompt_options opt; int n0 = 0, sp_pos = -1;
        std::vector<int32_t> ids = zonos2_build_prompt(model, texts[i], opt, n0, sp_pos);
        jobs[i].text = texts[i];
        jobs[i].idf.assign(ids.begin(), ids.end());
        jobs[i].n0 = n0;
        max_n0 = std::max(max_n0, n0);
    }

    // single-sequence reference (current KV path) per request
    std::vector<std::vector<int32_t>> ref_codes(n_req);
    std::vector<int> ref_eos(n_req, -1);
    for (int i = 0; i < n_req; ++i) {
        zonos2_sampling spr = sp_in;
        int eos = -1;
        zonos2_generate(model, jobs[i].idf.data(), jobs[i].n0, max_frames, spr, ref_codes[i], eos, true, nullptr, 0);
        ref_eos[i] = eos;
    }

    zonos2_batch_ctx bc;
    if (!zonos2_batch_init(bc, model, n_slots, max_n0 + max_frames)) {
        fprintf(stderr, "batch-test: batch init failed\n"); return 1;
    }

    for (int i = 0; i < n_req; ++i) {
        jobs[i].smp = std::make_unique<zonos2_sampler>(sp_in, ncb, av);
        jobs[i].slot = zonos2_slot{};
        jobs[i].slot.index  = i;
        jobs[i].slot.active = true;
        jobs[i].slot.n_past = jobs[i].n0;
        jobs[i].logits.assign((size_t) av * ncb, 0.0f);
        if (!zonos2_batch_slot_prefill(bc, i, jobs[i].idf.data(), jobs[i].n0, nullptr, 0, jobs[i].logits.data())) {
            jobs[i].slot.done = true;
            fprintf(stderr, "batch-test: prefill slot %d failed\n", i);
        }
    }

    for (;;) {
        bool any = false;                         // sample phase
        for (int i = 0; i < n_req; ++i) {
            Job & j = jobs[i];
            if (!j.slot.active || j.slot.done) continue;
            zonos2_slot_sample(model, j.slot, *j.smp, j.logits.data(), j.codes, max_frames, {});
            j.eos_frame = j.slot.eos_frame;
            any = true;
        }
        if (!any) break;
        std::vector<zonos2_slot *> step_list;     // step phase
        for (int i = 0; i < n_req; ++i)
            if (jobs[i].slot.active && !jobs[i].slot.done) step_list.push_back(&jobs[i].slot);
        if (step_list.empty()) break;
        zonos2_batch_step(bc, step_list);
        for (zonos2_slot * sp : step_list)
            memcpy(jobs[sp->index].logits.data(), zonos2_batch_slot_logits(bc, sp->index),
                   (size_t) av * ncb * sizeof(float));
    }

    printf("\n=== batch-test: %d requests in %d slots, %s, max=%d ===\n",
           n_req, n_slots, sp_in.greedy ? "greedy" : "sampling", max_frames);
    int fails = 0;
    for (int i = 0; i < n_req; ++i) {
        const auto & b = jobs[i].codes;
        const auto & r = ref_codes[i];
        const int nb = (int) b.size() / ncb, nr = (int) r.size() / ncb, cmp = std::min(nb, nr);
        int first_div = -1, agree = 0;
        for (int f = 0; f < cmp; ++f) {
            bool eq = true;
            for (int cb = 0; cb < ncb; ++cb)
                if (b[(size_t) f * ncb + cb] != r[(size_t) f * ncb + cb]) { eq = false; break; }
            if (eq) ++agree; else if (first_div < 0) first_div = f;
        }
        const bool exact = (nb == nr && agree == cmp);
        printf("  req %d \"%.28s\": batch=%d(eos=%d) ref=%d(eos=%d)  agree=%d/%d first_div=%d %s\n",
               i, jobs[i].text.c_str(), nb, jobs[i].eos_frame, nr, ref_eos[i], agree, cmp, first_div,
               exact ? "EXACT" : "");
        if (n_slots == 1 && sp_in.greedy && !exact) ++fails;  // F.1 must be bit-exact
    }
    for (int i = 0; i < n_req; ++i)
        for (int k = i + 1; k < n_req; ++k)
            if (jobs[i].text == jobs[k].text) {
                const bool same = jobs[i].codes == jobs[k].codes;
                printf("  F.2 slots %d,%d identical prompt -> %s\n", i, k, same ? "MATCH" : "MISMATCH");
                if (!same) ++fails;
            }

    if (!dac_path.empty() && n_req > 0 && !jobs[0].codes.empty()) {
        dac_model dm;
        if (dac_load(dm, dac_path.c_str(), use_gpu)) {
            std::vector<float> audio;
            const int nf = (int) jobs[0].codes.size() / ncb;
            if (dac_decode(dm, jobs[0].codes.data(), nf, ncb, jobs[0].eos_frame, audio))
                dac_write_wav("/tmp/batch_slot0.wav", audio, dm.sample_rate);
            dac_free(dm);
            printf("  wrote /tmp/batch_slot0.wav\n");
        }
    }

    zonos2_batch_free(bc);
    printf("batch-test: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}

// Apply conditioning-path overrides (from --inaccurate/--noisy-bg/--speaking-rate/--quality)
// onto an options struct. quality spec is "feat:bucket[,feat:bucket...]"; features default to -1
// (skip) and are sized to the model's quality-feature count.
static void apply_cond_opts(const zonos2_model & model, zonos2_prompt_options & opt,
                            int inaccurate, int noisy_bg, int rate, const std::string & quality) {
    if (inaccurate) opt.accurate_mode = false;
    if (noisy_bg)   opt.clean_speaker_background = false;
    if (rate >= 0)  opt.speaking_rate_bucket = rate;
    if (!quality.empty()) {
        const size_t nfeat = model.hp.cond_quality_bucket_counts.size();
        opt.quality_buckets.assign(nfeat, -1);
        size_t pos = 0;
        while (pos < quality.size()) {
            size_t comma = quality.find(',', pos);
            const std::string tok = quality.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const size_t colon = tok.find(':');
            if (colon != std::string::npos) {
                const int f = atoi(tok.substr(0, colon).c_str());
                const int b = atoi(tok.substr(colon + 1).c_str());
                if (f >= 0 && f < (int) nfeat) opt.quality_buckets[f] = b;
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const std::string path = argv[1];
    bool use_gpu = false;
    bool validate = false, generate = false, do_tts = false, do_build_prompt = false, do_batch_test = false, do_decode_bench = false;
    std::string ids_path, out_dir, out_codes, spk_path, text, prompt_out, dac_path, dump_ids_path;
    std::string clone_path, spk_encoder_path, save_spk_path; // one-command voice cloning
    std::string prune_mask_path;                             // --prune-mask: runtime expert pruning
    int mask_keep = -1, mask_drop_hits = -1;
    double mask_mass_eps = -1.0;
    int n_layer_limit = -1, max_frames = 400, spk_pos = 0, n_slots = 1;
    bool use_kv = true;
    // conditioning-path overrides (tri-state: <0 = leave prompt-builder default)
    int  cond_inaccurate = 0, cond_noisy_bg = 0, cond_rate = -1;
    std::string cond_quality;          // "feat:bucket[,feat:bucket...]"
    std::string emotion_dir = "emotion_directions";
    std::map<std::string, float> emotion_sliders;
    float emotion_valence = 0.0f, emotion_arousal = 0.0f, emotion_strength = 1.0f;
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
        else if (!strcmp(argv[i], "--batch-test") && i + 1 < argc) {
            do_batch_test = true; text = argv[++i];
        }
        else if (!strcmp(argv[i], "--decode-bench") && i + 1 < argc) {
            do_decode_bench = true; text = argv[++i];
        }
        else if (!strcmp(argv[i], "--slots")  && i + 1 < argc) n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) n_layer_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max")    && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dac") && i + 1 < argc) dac_path = argv[++i];
        else if (!strcmp(argv[i], "--speaker") && i + 1 < argc) spk_path = argv[++i];
        else if (!strcmp(argv[i], "--clone") && i + 1 < argc) clone_path = argv[++i];
        else if ((!strcmp(argv[i], "--spk-encoder") || !strcmp(argv[i], "--spk")) && i + 1 < argc) spk_encoder_path = argv[++i];
        else if (!strcmp(argv[i], "--save-speaker") && i + 1 < argc) save_spk_path = argv[++i];
        else if (!strcmp(argv[i], "--speaker-pos") && i + 1 < argc) spk_pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")   && i + 1 < argc) sp.seed = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) sp.greedy = true;
        else if (!strcmp(argv[i], "--recompute")) use_kv = false;
        else if (!strcmp(argv[i], "--dump-ids") && i + 1 < argc) dump_ids_path = argv[++i];
        else if (!strcmp(argv[i], "--prune-mask")      && i + 1 < argc) prune_mask_path = argv[++i];
        else if (!strcmp(argv[i], "--keep")            && i + 1 < argc) mask_keep      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--drop-below-hits") && i + 1 < argc) mask_drop_hits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mass-eps")        && i + 1 < argc) mask_mass_eps  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--inaccurate")) cond_inaccurate = 1;          // accurate_mode = false
        else if (!strcmp(argv[i], "--noisy-bg"))   cond_noisy_bg = 1;            // clean_speaker_background = false
        else if (!strcmp(argv[i], "--speaking-rate") && i + 1 < argc) cond_rate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quality") && i + 1 < argc) cond_quality = argv[++i];
        else if (!strcmp(argv[i], "--emotion") && i + 1 < argc) {
            if (!parse_emotion_spec(argv[++i], emotion_sliders)) { fprintf(stderr, "bad --emotion spec\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--emotion-valence") && i + 1 < argc) emotion_valence = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--emotion-arousal") && i + 1 < argc) emotion_arousal = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--emotion-strength") && i + 1 < argc) emotion_strength = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--emotion-cfg-scale") && i + 1 < argc) sp.emotion_cfg_scale = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--emotion-dir") && i + 1 < argc) emotion_dir = argv[++i];
        else { usage(argv[0]); return 1; }
    }

    if (!mp_file_exists(path)) {
        fprintf(stderr, "failed to load model %s\n", path.c_str());
        mp_print_download_hint("zonos2-q6_k.gguf (or another quant)", path);
        return 1;
    }

    // Companion discovery, gated on the modes that actually need the file (an unconditional
    // dac pickup would silently start emitting sibling .wavs on .npy outputs).
    if (dac_path.empty() && !out_codes.empty() && ends_with(out_codes, ".wav")) {
        dac_path = mp_find_companion(path, "dac.gguf");
        if (!dac_path.empty()) fprintf(stderr, "zonos2-cli: using dac.gguf found next to model: %s\n", dac_path.c_str());
        else {
            // fail before generation rather than after minutes of decode
            fprintf(stderr, "error: .wav output requested but no --dac <dac.gguf> given (also auto-detected next to <model.gguf>)\n");
            mp_print_download_hint("dac.gguf", mp_parent_dir(path) + "/dac.gguf");
            return 1;
        }
    }
    if (spk_encoder_path.empty() && !clone_path.empty()) {
        spk_encoder_path = mp_find_companion(path, "spk-encoder.gguf");
        if (!spk_encoder_path.empty()) fprintf(stderr, "zonos2-cli: using spk-encoder.gguf found next to model: %s\n", spk_encoder_path.c_str());
    }

    if (!clone_path.empty() && !spk_path.empty()) {
        fprintf(stderr, "error: --clone and --speaker are mutually exclusive (pick reference audio OR a precomputed embedding)\n");
        return 1;
    }
    if (!clone_path.empty() && spk_encoder_path.empty()) {
        fprintf(stderr, "error: --clone <ref_audio> requires --spk-encoder <spk-encoder.gguf> (also auto-detected next to <model.gguf>)\n");
        mp_print_download_hint("spk-encoder.gguf", mp_parent_dir(path) + "/spk-encoder.gguf");
        return 1;
    }
    if (clone_path.empty() && (!spk_encoder_path.empty() || !save_spk_path.empty())) {
        fprintf(stderr, "error: --spk-encoder/--save-speaker only apply with --clone <ref_audio>\n");
        return 1;
    }

    zonos2_model model;
    if (!zonos2_model_load(model, path.c_str(), use_gpu)) {
        fprintf(stderr, "failed to load model %s\n", path.c_str());
        if (!mp_file_exists(path)) mp_print_download_hint("zonos2-q6_k.gguf (or another quant)", path);
        return 1;
    }

    // --prune-mask: dynamically drop low-MSAN experts (runtime equivalent of prune-cli) so a
    // pruned recipe can be auditioned by ear before baking it into a GGUF.
    if (!prune_mask_path.empty()) {
        if (mask_keep < 0 && mask_drop_hits < 0 && mask_mass_eps < 0.0) {
            fprintf(stderr, "error: --prune-mask needs --keep N, --mass-eps E, or --drop-below-hits H\n");
            zonos2_model_free(model); return 1;
        }
        std::map<int, prune_stats::layer> st;
        if (!prune_stats::load(prune_mask_path, st)) { zonos2_model_free(model); return 1; }
        const auto keep = prune_policy::select(st, { mask_keep, mask_drop_hits, mask_mass_eps });
        if (!zonos2_set_expert_mask(model, keep)) { zonos2_model_free(model); return 1; }
    }

    // optional speaker embedding ([spk_dim] or [1, spk_dim] f32) for voice cloning:
    // either a precomputed --speaker npy, or --clone <ref_audio> encoded in-process
    // through the ECAPA speaker encoder (one-command cloning).
    std::vector<float> spk;
    if (!clone_path.empty()) {
        spk_model sm;
        if (!spk_load(sm, spk_encoder_path.c_str())) {
            fprintf(stderr, "failed to load speaker encoder %s\n", spk_encoder_path.c_str());
            if (!mp_file_exists(spk_encoder_path)) mp_print_download_hint("spk-encoder.gguf", spk_encoder_path);
            zonos2_model_free(model); return 1;
        }
        spk = spk_embed_from_file(sm, clone_path.c_str());
        spk_free(sm);
        if (spk.empty()) {
            fprintf(stderr, "clone: failed to encode %s (ffmpeg on PATH?)\n", clone_path.c_str());
            zonos2_model_free(model); return 1;
        }
        if (spk.size() != model.hp.spk_dim) {
            fprintf(stderr, "clone: speaker dim mismatch: encoder gave %zu, model wants %u\n", spk.size(), model.hp.spk_dim);
            zonos2_model_free(model); return 1;
        }
        if (!save_spk_path.empty()) {
            npy::save_f32(save_spk_path, spk.data(), { (int64_t) spk.size() });
            printf("clone: cached embedding -> %s\n", save_spk_path.c_str());
        }
        printf("clone: %zu-d vector from %s via %s, pos=%d\n",
               spk.size(), clone_path.c_str(), spk_encoder_path.c_str(), spk_pos);
    } else if (!spk_path.empty()) {
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
    std::vector<float> spk_emotion_delta;
    const bool emotion_requested = !emotion_sliders.empty() || emotion_valence != 0.0f || emotion_arousal != 0.0f;
    if (emotion_requested) {
        if (spk.empty()) {
            fprintf(stderr, "error: emotion control requires --speaker <spk.npy> or --clone <ref_audio>\n");
            zonos2_model_free(model); return 1;
        }
        zonos2_emotion_directions emotion;
        std::string err;
        if (!zonos2_emotion_load(emotion, emotion_dir, err)) {
            fprintf(stderr, "error: failed to load emotion directions from %s: %s\n", emotion_dir.c_str(), err.c_str());
            zonos2_model_free(model); return 1;
        }
        zonos2_emotion_request ereq;
        ereq.sliders = emotion_sliders;
        ereq.valence = emotion_valence;
        ereq.arousal = emotion_arousal;
        ereq.strength = emotion_strength;
        if (!zonos2_emotion_apply(emotion, ereq, spk, spk_emotion_delta, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            zonos2_model_free(model); return 1;
        }
        if (!spk_emotion_delta.empty() && spk_emotion_delta.size() != model.hp.n_embd) {
            fprintf(stderr, "error: emotion hidden delta dim mismatch: got %zu, want %u\n",
                    spk_emotion_delta.size(), model.hp.n_embd);
            zonos2_model_free(model); return 1;
        }
        if (spk_emotion_delta.empty()) sp.emotion_cfg_scale = 1.0f;
        printf("emotion: space=%s names=%zu axes=%zu strength=%.2f cfg=%.2f\n",
               emotion.space.c_str(), emotion.named.size(), emotion.axes.size(),
               emotion_strength, sp.emotion_cfg_scale);
    }
    const float * spk_ptr = spk.empty() ? nullptr : spk.data();
    const float * spk_delta_ptr = spk_emotion_delta.empty() ? nullptr : spk_emotion_delta.data();

    if (do_batch_test) {
        const int rc = run_batch_test(model, text, n_slots, max_frames, sp, dac_path, use_gpu);
        zonos2_model_free(model);
        return rc;
    }

    if (do_decode_bench) {
        const int rc = run_decode_bench(model, text, n_slots, max_frames);
        zonos2_model_free(model);
        return rc;
    }

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
        const bool okv = zonos2_validate(model, ids.data(), n_tokens, out_dir.c_str(), n_layer_limit,
                                         spk_ptr, spk_pos, spk_delta_ptr);
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
        std::vector<int32_t> codes, full_ids;
        int eos_frame = -1;
        const int n_frames = zonos2_generate(model, ids.data(), n0, max_frames, sp, codes, eos_frame,
                                             use_kv, spk_ptr, spk_pos,
                                             dump_ids_path.empty() ? nullptr : &full_ids,
                                             {}, spk_delta_ptr);
        const int ncb = (int) model.hp.n_codebooks;
        if (!dump_ids_path.empty()) {
            const int W = ncb + 1;
            std::vector<float> ff(full_ids.begin(), full_ids.end());
            npy::save_f32(dump_ids_path, ff.data(), { (int64_t) (full_ids.size() / W), (int64_t) W });
            printf("dump-ids: %zu rows x %d -> %s\n", full_ids.size() / W, W, dump_ids_path.c_str());
        }
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
        apply_cond_opts(model, opt, cond_inaccurate, cond_noisy_bg, cond_rate, cond_quality);
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
        apply_cond_opts(model, opt, cond_inaccurate, cond_noisy_bg, cond_rate, cond_quality);
        int n0 = 0, sp_pos = -1;
        std::vector<int32_t> ids = zonos2_build_prompt(model, text, opt, n0, sp_pos);
        std::vector<float> idf(ids.begin(), ids.end());
        const int ncb = (int) model.hp.n_codebooks;
        printf("tts: \"%s\" -> prompt [%d, %d], spk_pos=%d, max=%d, %s, %s\n",
               text.c_str(), n0, ncb + 1, sp_pos, max_frames,
               sp.greedy ? "greedy" : "sampling", use_kv ? "kv-cache" : "recompute");
        std::vector<int32_t> codes, full_ids;
        int eos_frame = -1;
        const int n_frames = zonos2_generate(model, idf.data(), n0, max_frames, sp, codes,
                                             eos_frame, use_kv, spk_ptr, sp_pos >= 0 ? sp_pos : 0,
                                             dump_ids_path.empty() ? nullptr : &full_ids,
                                             {}, spk_delta_ptr);
        if (!dump_ids_path.empty()) {
            const int W = ncb + 1;
            std::vector<float> ff(full_ids.begin(), full_ids.end());
            npy::save_f32(dump_ids_path, ff.data(), { (int64_t) (full_ids.size() / W), (int64_t) W });
            printf("dump-ids: %zu rows x %d -> %s\n", full_ids.size() / W, W, dump_ids_path.c_str());
        }
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
