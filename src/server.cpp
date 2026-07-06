// zonos2-server — HTTP TTS server mirroring ../ZONOS2's FastAPI surface.
//
// Loads the backbone + DAC + (optional) speaker encoder once and serves:
//   POST /tts/generate            JSON -> streaming float32 PCM (or buffered WAV)
//   POST /v1/audio/speech         OpenAI-compatible TTS
//   GET  /tts/capabilities        feature flags for the current model
//   GET  /tts/speakers            list session-cached speakers
//   POST /tts/speakers            cache a speaker embedding (audio upload or .npy)
//   GET  /tts/speakers/{id}/preview   cached reference audio (WAV)
//   GET  /v1/models, GET/POST /v1, GET /health
//   GET  /                        bundled web UI (web/tts_ui.html)
//
// Generation + DAC decode are serialized behind one mutex (the model is not thread-safe
// and uses CUDA-graph replay). Voice cloning decodes uploaded audio via ffmpeg in-process.
#include "compat.h"

#include "zonos2.h"
#include "zonos2-emotion.h"
#include "dac.h"
#include "spk-encoder.h"
#include "npy.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "zonos2-sampler.h"

using json = nlohmann::json;

// --------------------------------------------------------------------------- state

struct CachedSpeaker {
    std::string id, label, source_type, original_name;
    std::vector<float> emb;     // [spk_dim]
    std::vector<float> pcm24k;  // reference audio for preview (empty for embedding-only)
    double created = 0;
};

struct ReqJob;  // continuous-batching work item (defined after GenReq)

// A DAC decode unit handed from the backbone worker to a pool lane. Carries a snapshot of the codes
// (the worker keeps mutating job->codes, so the lane must not read it live). WINDOW emits a
// streaming block; FINALIZE flushes the streaming tail or does the single buffered decode, then
// marks the channel finished. `drop` skips the decode (client gone) but still finishes the channel.
// Defined here (not after ReqJob) so DacLane's std::deque<DacTask> sees a complete type — libc++
// instantiates the deque block-size on the member, which requires sizeof(DacTask). The
// std::shared_ptr<ReqJob> is fine with ReqJob still incomplete.
struct DacTask {
    std::shared_ptr<ReqJob> job;
    std::vector<int32_t> codes;          // snapshot of job->codes[0 : H*ncb)
    int H = 0;                           // frames in the snapshot
    enum Kind { WINDOW, FINALIZE } kind = WINDOW;
    int f_lo = 0, f_hi = 0, Lc = 0, Rc = 0;
    bool buffered = false;               // FINALIZE: full dac_decode vs windowed tail
    bool finish = false;                 // FINALIZE: signal channel done after this task
    bool drop = false;                   // skip decode, just finish
};

// One DAC pool lane: a thread + its own dac_model instance (ggml backends are not reentrant, so
// parallel decode needs independent instances) + a FIFO task queue. Each request is pinned to one
// lane for its lifetime, so its PCM blocks decode in order; different requests use different lanes.
struct DacLane {
    dac_model dac;
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::deque<DacTask> q;
};

struct ServerState {
    zonos2_model model;
    dac_model    dac;          // metadata (sample_rate) + WAV encoding (no backend compute)
    spk_model    spk;
    bool have_spk = false;
    bool use_gpu  = false;

    int max_frames   = 2000;   // ~23s ceiling; per-request max_tokens clamps below this
    int stream_block = 40;     // frames per streamed PCM block (~0.46s)
    int stream_ctx   = 24;     // conv-context frames each side (>=16 is seam-free); see dac_decode_window
    int batch_slots  = 8;      // continuous-batching width (concurrent in-flight syntheses). The
                               // decode-graph ladder runs the narrowest width covering the active
                               // slots, so raising this no longer slows solo/low-concurrency
                               // requests; only KV-cache memory (~0.18 GB/slot) scales with it.
    int dac_threads  = 4;      // DAC decode pool lanes (parallel decode across requests)
    bool prof = false;         // ZONOS2_PROFILE: emit per-request decode timing
    std::string ui_path = "web/tts_ui.html";
    std::string emotion_dir = "emotion_directions";
    zonos2_emotion_directions emotion;
    bool have_emotion = false;

    // continuous-batching scheduler: one worker thread owns the backbone + batch context and only
    // produces codes; DAC decode is offloaded to the pool so the worker never blocks on it. HTTP
    // handlers enqueue jobs and stream results back through each job's channel.
    std::thread worker;
    std::atomic<bool> stop{false};
    std::mutex sched_mtx;                 // guards `incoming`
    std::condition_variable sched_cv;     // wakes the worker on a new job / shutdown
    std::deque<std::shared_ptr<ReqJob>> incoming;

    std::vector<DacLane> dac_lanes;       // DAC decode pool (size dac_threads)
    uint64_t next_lane = 0;               // round-robin lane assignment (worker thread only)

    std::mutex spk_mtx;          // guards `sessions`
    std::mutex spk_compute_mtx;  // serialize ECAPA encoder compute (shared spk ggml backend, not reentrant)
    std::map<std::string, std::map<std::string, CachedSpeaker>> sessions; // session -> id -> speaker
    std::atomic<uint64_t> id_ctr{0};
};

static const char * QFEATS[6] = {
    "lufs", "estimated_snr", "max_pause", "estimated_bandlimit_hz", "leading_silence_s", "trailing_silence_s"
};

// Conditioning bucket *labels* for the Zyphra/ZONOS2 release (from params.json), surfaced via
// /tts/capabilities so the web UI labels the bins with their ranges instead of raw indices.
// Used only when the loaded model's bucket counts match these; otherwise we fall back to index
// strings (so a model with a different conditioning layout still renders, just unlabeled).
static const std::vector<std::string> SPEAKING_RATE_LABELS = {
    "0-8", "8-11", "11-14", "14-17", "17-21", "21-28", "28-40", "40+"
};
static const std::map<std::string, std::vector<std::string>> QUALITY_LABELS = {
    {"lufs",                   {"-1000--50","-50--45.5","-45.5--41","-41--36.5","-36.5--32","-32--27.5","-27.5--23","-23--18.5","-18.5--14","-14--9.5","-9.5--5","-5+"}},
    {"estimated_snr",          {"-1000-0","0-6","6-12","12-18","18-24","24-30","30-36","36-42","42-48","48-54","54-60","60+"}},
    {"max_pause",              {"0-0.5","0.5-1","1-1.5","1.5-2","2-2.5","2.5-3","3-3.5","3.5-4","4-4.5","4.5-5","5-5.5","5.5-6"}},
    {"estimated_bandlimit_hz", {"495.3-3433","3433-6371","6371-9310","9310-12248","12248-15186","15186-18124","18124-21062","21062-24000"}},
    {"leading_silence_s",      {"0-0.05","0.05-0.1","0.1-0.25","0.25-0.5","0.5-1","1-2","2-4","4+"}},
    {"trailing_silence_s",     {"0-0.05","0.05-0.1","0.1-0.25","0.25-0.5","0.5-1","1-2","2-4","4+"}},
};
static const double QUALITY_DROPOUT = 0.25;   // release default (params.json), per quality feature

// --------------------------------------------------------------------------- helpers

static double jnum (const json & j, const char * k, double d)            { return (j.contains(k) && j[k].is_number())  ? j[k].get<double>() : d; }
static int    jint (const json & j, const char * k, int d)              { return (j.contains(k) && j[k].is_number())  ? (int) j[k].get<double>() : d; }
static bool   jbool(const json & j, const char * k, bool d)             { return (j.contains(k) && j[k].is_boolean()) ? j[k].get<bool>() : d; }
static std::string jstr(const json & j, const char * k, const std::string & d = "") { return (j.contains(k) && j[k].is_string()) ? j[k].get<std::string>() : d; }

static std::string make_id(ServerState & s, const char * prefix) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%010llx", prefix, (unsigned long long) (s.id_ctr++));
    return buf;
}

static std::filesystem::path temp_path(ServerState & s, const char * suffix) {
    auto dir = std::filesystem::temp_directory_path();
    return dir / ("zonos2-" + std::to_string((unsigned long long) (s.id_ctr++)) + suffix);
}

// Base64 decode (tolerates whitespace and a leading "data:...;base64," prefix).
static bool b64_decode(const std::string & in_raw, std::vector<uint8_t> & out) {
    static int8_t T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) T[i] = -1;
        const char * A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[(unsigned char) A[i]] = (int8_t) i;
        T[(unsigned char) '-'] = 62; T[(unsigned char) '_'] = 63;  // url-safe
        init = true;
    }
    std::string in = in_raw;
    const size_t comma = in.find("base64,");
    if (comma != std::string::npos) in = in.substr(comma + 7);

    out.clear();
    int val = 0, bits = 0;
    for (unsigned char c : in) {
        if (c == '=' ) break;
        const int8_t d = T[c];
        if (d < 0) continue;               // skip whitespace / newlines
        val = (val << 6) | d; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t) ((val >> bits) & 0xFF)); }
    }
    return !out.empty();
}

// Cosine fade-out over the final fade_ms milliseconds.
static void apply_fade(std::vector<float> & a, double fade_ms, int sr) {
    if (fade_ms <= 0 || a.empty()) return;
    const size_t n = std::min(a.size(), (size_t) (fade_ms * 0.001 * sr));
    for (size_t i = 0; i < n; ++i) {
        const float t = (float) i / (float) n;            // 0..1 across the fade
        const float g = 0.5f * (1.0f + cosf((float) M_PI * t));
        a[a.size() - n + i] *= g;
    }
}

static void set_json(httplib::Response & res, const json & j, int status = 200) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

// --------------------------------------------------------------------------- request model

struct GenReq {
    std::string text;
    zonos2_sampling sp;                 // defaults already match the reference
    int max_frames = 0;                 // resolved against server cap
    zonos2_prompt_options opt;
    std::vector<float> spk;             // empty => no speaker
    std::vector<float> spk_emotion_delta; // space="proj" delta [n_embd], empty => none
    int spk_pos = 0;
    double fade_out_ms = 0;
    bool stream = true;
    std::string format = "pcm";         // "pcm" (float32) | "wav"
};

// One in-flight synthesis. The HTTP handler fills the request fields + prompt and enqueues it; the
// worker thread owns the runtime fields (slot/sampler/codes) once admitted and streams PCM blocks
// back through the channel (mutex/cv/pcm). For buffered requests the worker pushes one final block.
struct ReqJob {
    // request (handler-filled, read-only to the worker)
    GenReq gr;
    std::vector<float> idf;              // prompt ids [n0*W] as floats
    int n0 = 0, spk_pos = 0;
    bool stream = true;                  // chunked float32 PCM vs buffered (wav / pcm)
    bool wav = false;

    // runtime (worker-owned)
    zonos2_slot slot;
    zonos2_slot cfg_slot;
    int cfg_slot_index = -1;
    std::unique_ptr<zonos2_sampler> smp;
    std::vector<float> logits;           // [av*ncb] current sampling source
    std::vector<float> cfg_logits;       // unconditioned logits for emotion CFG
    std::vector<int32_t> codes;
    int eos_frame = -1;
    int emitted = 0;                     // frames already handed to the DAC pool (streaming)
    int lane = 0;                        // assigned DAC pool lane (decodes this job's blocks in order)

    // decode profiling (ZONOS2_PROFILE): admission->done wall and summed batch_step compute
    std::chrono::steady_clock::time_point t_decode0;
    double decode_step_ms = 0.0;         // summed zonos2_batch_step ms feeding this slot (exact at batch=1)

    // channel: DAC pool -> handler
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<float>> pcm;  // ready PCM blocks (float32)
    bool finished = false;
    bool failed   = false;
    bool client_dropped = false;         // set by the handler when the socket drops
};

// Resolve a speaker vector from a request's speaker_* fields. Returns true if one was set,
// false if none requested. On error sets `err` and returns false.
static bool resolve_speaker(ServerState & s, const json & j, const std::string & session,
                            std::vector<float> & out, std::string & err) {
    const int spk_dim = (int) s.model.hp.spk_dim;

    const std::string emb_id = jstr(j, "speaker_embedding_id");
    if (!emb_id.empty()) {
        std::lock_guard<std::mutex> lk(s.spk_mtx);
        auto sit = s.sessions.find(session);
        if (sit != s.sessions.end()) {
            auto it = sit->second.find(emb_id);
            if (it != sit->second.end()) { out = it->second.emb; return true; }
        }
        err = "unknown speaker_embedding_id '" + emb_id + "'";
        return false;
    }

    std::string audio_b64 = jstr(j, "speaker_audio_base64");
    if (audio_b64.empty()) audio_b64 = jstr(j, "speaker_wav_base64");   // legacy alias
    if (!audio_b64.empty()) {
        if (!s.have_spk) { err = "speaker audio upload requires --spk <encoder.gguf>"; return false; }
        std::vector<uint8_t> bytes;
        if (!b64_decode(audio_b64, bytes)) { err = "invalid speaker_audio_base64"; return false; }
        const auto tmp = temp_path(s, ".audio");
        FILE * f = fopen(tmp.string().c_str(), "wb");
        if (!f) { err = "cannot write temp audio"; return false; }
        fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
        std::vector<float> pcm = spk_decode_audio_file(s.spk, tmp.string().c_str());   // ffmpeg only (no backend)
        std::error_code ec; std::filesystem::remove(tmp, ec);
        if (pcm.empty()) { err = "ffmpeg failed to decode speaker audio"; return false; }
        { std::lock_guard<std::mutex> lk(s.spk_compute_mtx);
          out = spk_embed_from_pcm24k(s.spk, pcm.data(), (int) pcm.size()); }
        if ((int) out.size() != spk_dim) { err = "speaker encoding failed (got " + std::to_string(out.size()) + ", want " + std::to_string(spk_dim) + ")"; out.clear(); return false; }
        return true;
    }

    const std::string emb_b64 = jstr(j, "speaker_embedding_base64");
    if (!emb_b64.empty()) {
        std::vector<uint8_t> bytes;
        if (!b64_decode(emb_b64, bytes)) { err = "invalid speaker_embedding_base64"; return false; }
        const auto tmp = temp_path(s, ".npy");
        FILE * f = fopen(tmp.string().c_str(), "wb");
        if (!f) { err = "cannot write temp embedding"; return false; }
        fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
        std::vector<float> data; std::vector<int64_t> shape;
        const bool ok = npy::load_f32(tmp.string(), data, shape);
        std::error_code ec; std::filesystem::remove(tmp, ec);
        if (!ok) { err = "speaker_embedding_base64 is not a valid <f4 .npy"; return false; }
        if (shape.size() == 1 && (int) data.size() == spk_dim) {
            out = data;
        } else if (shape.size() == 2 && (int) shape[1] == spk_dim) {   // average rows
            out.assign(spk_dim, 0.0f);
            for (int r = 0; r < (int) shape[0]; ++r)
                for (int c = 0; c < spk_dim; ++c) out[c] += data[(size_t) r * spk_dim + c];
            for (int c = 0; c < spk_dim; ++c) out[c] /= (float) shape[0];
        } else {
            err = "speaker embedding shape mismatch (want [" + std::to_string(spk_dim) + "])";
            return false;
        }
        return true;
    }
    return false; // no speaker requested
}

// Map quality_buckets (array or {feature:bucket} object) to a per-feature vector (-1 = skip).
static std::vector<int> parse_quality(const json & j, int n_feat) {
    std::vector<int> q;
    if (!j.contains("quality_buckets")) return q;             // empty => caller uses model default
    const json & qb = j["quality_buckets"];
    if (qb.is_array()) {
        q.assign(n_feat, -1);
        for (int i = 0; i < n_feat && i < (int) qb.size(); ++i)
            if (qb[i].is_number()) q[i] = (int) qb[i].get<double>();
    } else if (qb.is_object()) {
        q.assign(n_feat, -1);
        for (int i = 0; i < n_feat; ++i)
            if (qb.contains(QFEATS[i]) && qb[QFEATS[i]].is_number()) q[i] = (int) qb[QFEATS[i]].get<double>();
    }
    return q;
}

static bool parse_emotion_sliders(const json & j, std::map<std::string, float> & sliders,
                                  std::string & err) {
    sliders.clear();
    if (!j.contains("emotion_sliders") || j["emotion_sliders"].is_null()) return true;
    if (!j["emotion_sliders"].is_object()) {
        err = "emotion_sliders must be an object";
        return false;
    }
    for (auto it = j["emotion_sliders"].begin(); it != j["emotion_sliders"].end(); ++it) {
        if (!it.value().is_number()) {
            err = "emotion_sliders." + it.key() + " must be numeric";
            return false;
        }
        sliders[it.key()] = it.value().get<float>();
    }
    return true;
}

static bool apply_request_emotion(ServerState & s, const json & j, GenReq & out,
                                  std::string & err) {
    out.sp.emotion_cfg_scale = (float) jnum(j, "emotion_cfg_scale", 1.0);
    if (!jbool(j, "emotion_enabled", false)) return true;

    zonos2_emotion_request req;
    if (!parse_emotion_sliders(j, req.sliders, err)) return false;
    req.valence = (float) jnum(j, "emotion_valence", 0.0);
    req.arousal = (float) jnum(j, "emotion_arousal", 0.0);
    req.strength = (float) jnum(j, "emotion_strength", 1.0);
    req.speaker_key = jstr(j, "speaker_embedding_id");

    const bool requested = !req.sliders.empty() || req.valence != 0.0f || req.arousal != 0.0f;
    if (!requested) return true;

    // Mirrors the Python server: emotion control is a no-op without a speaker embedding.
    if (out.spk.empty()) {
        out.sp.emotion_cfg_scale = 1.0f;
        return true;
    }
    if (!s.have_emotion) {
        err = "emotion control requested but no emotion directions are configured";
        return false;
    }

    if (!zonos2_emotion_apply(s.emotion, req, out.spk, out.spk_emotion_delta, err)) return false;
    if (!out.spk_emotion_delta.empty() && (int) out.spk_emotion_delta.size() != (int) s.model.hp.n_embd) {
        err = "emotion hidden delta dim mismatch: got " + std::to_string(out.spk_emotion_delta.size()) +
              ", want " + std::to_string(s.model.hp.n_embd);
        out.spk_emotion_delta.clear();
        return false;
    }
    if (out.spk_emotion_delta.empty()) out.sp.emotion_cfg_scale = 1.0f;
    return true;
}

// Parse a /tts/generate-style JSON body into a GenReq. Returns false (with err) on bad input.
static bool parse_gen_req(ServerState & s, const json & j, const std::string & session,
                          GenReq & out, std::string & err) {
    out.text = jstr(j, "text");
    if (out.text.empty()) out.text = jstr(j, "input");          // OpenAI field
    if (out.text.empty()) { err = "missing 'text'"; return false; }

    out.sp.temperature   = (float) jnum(j, "temperature",   out.sp.temperature);
    out.sp.top_k         =         jint(j, "topk",          out.sp.top_k);
    out.sp.top_p         = (float) jnum(j, "top_p",         out.sp.top_p);
    out.sp.min_p         = (float) jnum(j, "min_p",         out.sp.min_p);
    out.sp.rep_penalty   = (float) jnum(j, "repetition_penalty",  out.sp.rep_penalty);
    out.sp.rep_window    =         jint(j, "repetition_window",   out.sp.rep_window);
    out.sp.rep_codebooks =         jint(j, "repetition_codebooks", out.sp.rep_codebooks);
    if (out.sp.rep_codebooks < 0) out.sp.rep_codebooks = (int) s.model.hp.n_codebooks;
    if (j.contains("seed") && j["seed"].is_number()) out.sp.seed = (uint32_t) j["seed"].get<long long>();
    else { std::random_device rd; out.sp.seed = rd(); }   // no seed given => fresh trial each request
    if (out.sp.temperature <= 0.0f) out.sp.greedy = true;

    const int cap = s.max_frames;
    const int req_max = jint(j, "max_tokens", 0);
    out.max_frames = (req_max > 0) ? std::min(req_max, cap) : cap;

    // conditioning
    const int n_feat = (int) s.model.hp.cond_quality_bucket_counts.size();
    if (jbool(j, "speaking_rate_enabled", false)) {
        if (j.contains("speaking_rate_bucket") && j["speaking_rate_bucket"].is_number())
            out.opt.speaking_rate_bucket = (int) j["speaking_rate_bucket"].get<double>();
    }
    if (!jbool(j, "quality_enabled", true)) out.opt.quality_buckets.assign(n_feat, -1);
    else out.opt.quality_buckets = parse_quality(j, n_feat);   // empty => model default

    out.opt.clean_speaker_background = jbool(j, "clean_speaker_background", false);
    out.opt.accurate_mode            = jbool(j, "accurate_mode", true);

    out.fade_out_ms = jnum(j, "fade_out_ms", 0.0);
    out.stream      = jbool(j, "stream", true);
    out.format      = jstr(j, "format", "pcm");

    // speaker
    if (!resolve_speaker(s, j, session, out.spk, err)) {
        if (!err.empty()) return false;     // requested but failed
    }
    if (!out.spk.empty()) out.opt.add_speaker_slot = true;
    if (!apply_request_emotion(s, j, out, err)) return false;
    return true;
}

// --------------------------------------------------------------------------- synthesis worker
//
// One worker thread owns the backbone and a single zonos2_batch_ctx (batch_slots wide); it only
// produces codes. DAC decode is offloaded to a pool of lanes (each its own dac_model) so the
// backbone never blocks on it. Each tick: admit waiting jobs into free slots (prefill), sample one
// frame per active slot, hand ready blocks to the job's DAC lane, evict finished slots, batch-step
// the rest. The decode graph is static-shape and CUDA-graph-replayed; per-slot prefill runs a
// separate one-off graph (one re-capture per admit). For bit-exact single-stream output use
// --dac-cpu (CPU DAC keeps the backbone CUDA graph undisturbed); GPU DAC is best for throughput.

static void push_pcm(ReqJob & job, std::vector<float> && pcm) {
    if (pcm.empty()) return;
    std::lock_guard<std::mutex> lk(job.m);
    job.pcm.push_back(std::move(pcm));
    job.cv.notify_one();
}

static void job_finish(ReqJob & job, bool failed) {
    std::lock_guard<std::mutex> lk(job.m);
    job.failed   = failed;
    job.finished = true;
    job.cv.notify_one();
}

static void dac_lane_submit(DacLane & lane, DacTask && t) {
    std::lock_guard<std::mutex> lk(lane.m);
    lane.q.push_back(std::move(t));
    lane.cv.notify_one();
}

// DAC pool lane: own a dac_model, decode tasks FIFO, push PCM to the job channel. A job is pinned to
// one lane, so its blocks stay ordered; FINALIZE tasks carry `finish` to close the channel last.
static void dac_lane_loop(ServerState & s, DacLane & lane) {
    const int ncb = lane.dac.n_codebooks, sr = lane.dac.sample_rate;
    // Warm this lane's DAC pipelines (conv_transpose/im2col, etc.) before serving so the first
    // streamed/finalized block doesn't JIT-compile them on the request's critical path.
    if (!getenv("ZONOS2_NO_WARMUP")) {
        const int H = 32;
        std::vector<int32_t> codes((size_t) H * ncb, 0);
        std::vector<float> pcm;
        dac_decode(lane.dac, codes.data(), H, ncb, /*eos=*/-1, pcm);                 // buffered path
        dac_decode_window(lane.dac, codes.data(), H, 0, 16, s.stream_ctx, 0, pcm); // streaming path
    }
    for (;;) {
        DacTask t;
        {
            std::unique_lock<std::mutex> lk(lane.m);
            lane.cv.wait(lk, [&]{ return !lane.q.empty() || s.stop.load(); });
            if (lane.q.empty()) { if (s.stop.load()) break; continue; }
            t = std::move(lane.q.front()); lane.q.pop_front();
        }
        ReqJob & job = *t.job;
        if (!t.drop) {
            std::vector<float> pcm;
            if (t.kind == DacTask::WINDOW) {
                if (dac_decode_window(lane.dac, t.codes.data(), t.H, t.f_lo, t.f_hi, t.Lc, t.Rc, pcm))
                    push_pcm(job, std::move(pcm));
            } else if (t.buffered) {
                if (t.H > 0 && dac_decode(lane.dac, t.codes.data(), t.H, ncb, job.eos_frame, pcm)) {
                    apply_fade(pcm, job.gr.fade_out_ms, sr);
                    push_pcm(job, std::move(pcm));
                }
            } else { // streaming tail: [f_lo, f_hi) with Rc=0, faded
                if (t.f_hi > t.f_lo &&
                    dac_decode_window(lane.dac, t.codes.data(), t.H, t.f_lo, t.f_hi, t.Lc, /*Rc=*/0, pcm)) {
                    apply_fade(pcm, job.gr.fade_out_ms, sr);
                    push_pcm(job, std::move(pcm));
                }
            }
        }
        if (t.finish) job_finish(job, /*failed=*/false);
    }
}

// Streaming: hand every ready block to the job's DAC lane (snapshotting codes, since the worker
// keeps appending). Mirrors the old synth_stream block cadence; decode happens off-thread.
static void stream_submit_ready(ServerState & s, const std::shared_ptr<ReqJob> & job) {
    const int ncb = (int) s.model.hp.n_codebooks;
    const int BLOCK = s.stream_block, CTX = s.stream_ctx, SHEAR = ncb - 1;
    const int H = (int) job->codes.size() / ncb;         // frames generated so far
    while (H - job->emitted >= BLOCK + CTX + SHEAR) {
        const int f_hi = H - CTX - SHEAR;                // hold back conv ctx + shear lookahead
        DacTask t; t.job = job; t.kind = DacTask::WINDOW;
        t.codes.assign(job->codes.begin(), job->codes.begin() + (size_t) H * ncb);
        t.H = H; t.f_lo = job->emitted; t.f_hi = f_hi; t.Lc = CTX; t.Rc = CTX;
        dac_lane_submit(s.dac_lanes[job->lane], std::move(t));
        job->emitted = f_hi;
    }
}

// Eviction: hand the final decode (streaming tail or one buffered decode) to the lane, with finish.
static void submit_finalize(ServerState & s, const std::shared_ptr<ReqJob> & job, bool dropped) {
    const int ncb = (int) s.model.hp.n_codebooks;
    const int nf  = (int) job->codes.size() / ncb;
    DacTask t; t.job = job; t.kind = DacTask::FINALIZE; t.finish = true; t.drop = dropped;
    if (!dropped) {
        t.codes = job->codes; t.H = nf;
        if (job->stream) {
            const int end = (job->eos_frame >= 0 && job->eos_frame < nf) ? job->eos_frame : nf;
            t.buffered = false; t.f_lo = job->emitted; t.f_hi = end; t.Lc = s.stream_ctx; t.Rc = 0;
        } else {
            t.buffered = true;                            // full decode of [0, nf)
        }
    }
    dac_lane_submit(s.dac_lanes[job->lane], std::move(t));
}

// Run a throwaway prefill + a decode step at every ladder width so every backbone graph pipeline
// is JIT-compiled before the first real request. On Metal that compilation is ~0.5s of lazy kernel
// builds that would otherwise land on the first user's time-to-first-audio. The decode ladder runs
// the narrowest graph covering the active slots, so each width has its own pipelines to warm. The
// dummy K/V left in the touched cache bands is overwritten when a real request re-prefills its slot.
static void warmup_backbone(zonos2_batch_ctx & bc) {
    if (getenv("ZONOS2_NO_WARMUP")) return;
    const int W = bc.W, n0 = 16;
    std::vector<float> ids((size_t) n0 * W, 0.0f);          // all-zero codes are valid token ids
    std::vector<float> logits((size_t) bc.av * bc.ncb, 0.0f);
    const auto t0 = std::chrono::steady_clock::now();
    if (!zonos2_batch_slot_prefill(bc, 0, ids.data(), n0, nullptr, -1, logits.data())) return;
    // Step a dummy slot at each ladder width's top column so all decode-graph widths get warmed.
    for (const auto & dg : bc.dec_ladder) {
        zonos2_slot slot;
        slot.index = dg.width - 1; slot.active = true; slot.n_past = n0;
        slot.next_ids.assign(W, 0);
        zonos2_batch_step(bc, { &slot });                   // n_past auto-advances
    }
    const auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "zonos2-server: backbone warmup (%zu ladder widths) in %.0f ms\n",
            bc.dec_ladder.size(), std::chrono::duration<double, std::milli>(t1 - t0).count());
}

static void worker_loop(ServerState & s) {
    const int ncb = (int) s.model.hp.n_codebooks, av = (int) s.model.hp.audio_vocab;
    const int B = s.batch_slots;
    const int slot_cap_frames = s.max_frames + 1024;   // headroom for prompt rows
    zonos2_batch_ctx bc;
    if (!zonos2_batch_init(bc, s.model, B, slot_cap_frames)) {
        fprintf(stderr, "worker: batch init failed; synthesis disabled\n");
        return;
    }
    warmup_backbone(bc);
    std::vector<std::shared_ptr<ReqJob>> slot_job(B);

    auto find_free_slots = [&](int needed, int & primary, int & twin) {
        primary = -1;
        twin = -1;
        for (int i = 0; i < B; ++i) {
            if (slot_job[i]) continue;
            if (primary < 0) primary = i;
            else { twin = i; break; }
        }
        return needed == 1 ? primary >= 0 : (primary >= 0 && twin >= 0);
    };

    while (!s.stop.load()) {
        // 1. admit waiting jobs into free slots (prefill into the slot's cache band)
        for (;;) {
            std::shared_ptr<ReqJob> job;
            { std::lock_guard<std::mutex> lk(s.sched_mtx);
              if (!s.incoming.empty()) job = s.incoming.front(); }
            if (!job) break;
            const bool use_cfg = !job->gr.spk_emotion_delta.empty() && job->gr.sp.emotion_cfg_scale != 1.0f;
            const int needed_slots = use_cfg ? 2 : 1;
            if (needed_slots > B) {
                { std::lock_guard<std::mutex> lk(s.sched_mtx); s.incoming.pop_front(); }
                job_finish(*job, /*failed=*/true);
                continue;
            }
            int primary = -1, twin = -1;
            if (!find_free_slots(needed_slots, primary, twin)) break;
            { std::lock_guard<std::mutex> lk(s.sched_mtx); s.incoming.pop_front(); }

            job->slot = zonos2_slot{};
            job->slot.index = primary; job->slot.active = true; job->slot.n_past = job->n0;
            job->cfg_slot_index = use_cfg ? twin : -1;
            if (use_cfg) {
                job->cfg_slot = zonos2_slot{};
                job->cfg_slot.index = twin;
                job->cfg_slot.active = true;
                job->cfg_slot.n_past = job->n0;
            }
            job->lane = (int) (s.next_lane++ % s.dac_lanes.size()); // round-robin DAC lane
            job->smp = std::make_unique<zonos2_sampler>(job->gr.sp, ncb, av);
            job->logits.assign((size_t) av * ncb, 0.0f);
            job->cfg_logits.assign(use_cfg ? (size_t) av * ncb : 0, 0.0f);
            const float * spk = job->gr.spk.empty() ? nullptr : job->gr.spk.data();
            const float * delta = job->gr.spk_emotion_delta.empty() ? nullptr : job->gr.spk_emotion_delta.data();
            if (!zonos2_batch_slot_prefill(bc, primary, job->idf.data(), job->n0, spk, job->spk_pos,
                                           job->logits.data(), delta)) {
                job_finish(*job, /*failed=*/true);
                continue;                                   // leave slot free
            }
            if (use_cfg && !zonos2_batch_slot_prefill(bc, twin, job->idf.data(), job->n0, spk, job->spk_pos,
                                                      job->cfg_logits.data(), nullptr)) {
                job_finish(*job, /*failed=*/true);
                continue;
            }
            job->t_decode0 = std::chrono::steady_clock::now();
            slot_job[primary] = job;
            if (use_cfg) slot_job[twin] = job;
        }

        // 2. nothing in flight -> block until a job arrives (or shutdown)
        bool any = false;
        for (int i = 0; i < B; ++i) if (slot_job[i]) { any = true; break; }
        if (!any) {
            std::unique_lock<std::mutex> lk(s.sched_mtx);
            if (s.incoming.empty() && !s.stop.load())
                s.sched_cv.wait_for(lk, std::chrono::milliseconds(200));
            continue;
        }

        // 3. sample one frame per active slot from its current logits
        for (int i = 0; i < B; ++i) {
            auto & job = slot_job[i];
            if (!job || job->slot.index != i || job->slot.done) continue;
            { std::lock_guard<std::mutex> lk(job->m);
              if (job->client_dropped) {
                  job->slot.done = true;
                  if (job->cfg_slot_index >= 0) job->cfg_slot.done = true;
                  continue;
              } }
            std::vector<float> guided;
            const float * sample_logits = job->logits.data();
            if (job->cfg_slot_index >= 0) {
                guided.resize((size_t) av * ncb);
                const float scale = job->gr.sp.emotion_cfg_scale;
                for (size_t k = 0; k < guided.size(); ++k)
                    guided[k] = job->cfg_logits[k] + scale * (job->logits[k] - job->cfg_logits[k]);
                sample_logits = guided.data();
            }
            zonos2_slot_sample(s.model, job->slot, *job->smp, sample_logits,
                               job->codes, job->gr.max_frames, {});
            if (job->cfg_slot_index >= 0) {
                job->cfg_slot.next_ids = job->slot.next_ids;
                job->cfg_slot.step = job->slot.step;
                job->cfg_slot.eos_frame = job->slot.eos_frame;
                job->cfg_slot.countdown = job->slot.countdown;
                job->cfg_slot.done = job->slot.done;
            }
            job->eos_frame = job->slot.eos_frame;
            if (job->stream) stream_submit_ready(s, job);
        }

        // 4. evict finished slots: hand the final decode to the DAC lane (which signals the handler
        //    after it drains), then free the slot immediately so the backbone keeps going.
        for (int i = 0; i < B; ++i) {
            auto & job = slot_job[i];
            if (!job || job->slot.index != i || !job->slot.done) continue;
            if (s.prof) {
                const double wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - job->t_decode0).count();
                const int f = job->slot.step;
                fprintf(stderr, "decode[prof]: %d frames  batch_step=%.0fms (%.1f fps)  wall=%.0fms (%.1f fps)\n",
                        f, job->decode_step_ms, f ? f * 1000.0 / job->decode_step_ms : 0.0,
                        wall_ms, f ? f * 1000.0 / wall_ms : 0.0);
            }
            bool dropped; { std::lock_guard<std::mutex> lk(job->m); dropped = job->client_dropped; }
            submit_finalize(s, job, dropped);
            if (job->cfg_slot_index >= 0) slot_job[job->cfg_slot_index].reset();
            slot_job[i].reset();
        }

        // 5. one batched decode step over the still-active slots, refill their logits
        std::vector<zonos2_slot *> step_list;
        for (int i = 0; i < B; ++i)
            if (slot_job[i] && slot_job[i]->slot.index == i && !slot_job[i]->slot.done) {
                step_list.push_back(&slot_job[i]->slot);
                if (slot_job[i]->cfg_slot_index >= 0 && !slot_job[i]->cfg_slot.done)
                    step_list.push_back(&slot_job[i]->cfg_slot);
            }
        if (!step_list.empty()) {
            const auto st0 = std::chrono::steady_clock::now();
            zonos2_batch_step(bc, step_list);
            const double step_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - st0).count();
            for (int i = 0; i < B; ++i) {
                auto & job = slot_job[i];
                if (!job || job->slot.index != i || job->slot.done) continue;
                memcpy(job->logits.data(), zonos2_batch_slot_logits(bc, job->slot.index),
                       (size_t) av * ncb * sizeof(float));
                if (job->cfg_slot_index >= 0)
                    memcpy(job->cfg_logits.data(), zonos2_batch_slot_logits(bc, job->cfg_slot.index),
                           (size_t) av * ncb * sizeof(float));
                job->decode_step_ms += step_ms; // exact at batch=1; upper bound when batched
            }
        }
    }

    // drain: signal any still-occupied slots so their handlers unblock
    for (auto & job : slot_job) if (job) job_finish(*job, /*failed=*/true);
    zonos2_batch_free(bc);
}

// --------------------------------------------------------------------------- request handler

// Shared handler for /tts/generate and /v1/audio/speech. Parses + builds the prompt on the handler
// thread, then hands the job to the worker and streams (chunked) or waits (buffered) for the result.
static void handle_generate(ServerState & s, const httplib::Request & req, httplib::Response & res) {
    json j = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) { set_json(res, {{"error", "invalid JSON body"}}, 400); return; }

    const std::string session = req.get_header_value("X-TTS-Session-ID");
    GenReq gr; std::string err;
    if (!parse_gen_req(s, j, session, gr, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!gr.spk_emotion_delta.empty() && gr.sp.emotion_cfg_scale != 1.0f && s.batch_slots < 2) {
        set_json(res, {{"error", "emotion_cfg_scale requires --batch 2 or higher"}}, 400);
        return;
    }

    auto job = std::make_shared<ReqJob>();
    job->gr = gr;
    {
        zonos2_prompt_options opt = gr.opt;             // CPU-only prompt build, safe off the worker
        int n0 = 0, sp_pos = -1;
        std::vector<int32_t> ids = zonos2_build_prompt(s.model, gr.text, opt, n0, sp_pos);
        job->idf.assign(ids.begin(), ids.end());
        job->n0 = n0;
        job->spk_pos = sp_pos >= 0 ? sp_pos : 0;
    }
    job->stream = gr.stream && gr.format != "wav";
    job->wav    = (gr.format == "wav");

    res.set_header("X-Seed", std::to_string(gr.sp.seed));   // report the seed used (reproduce by sending it back)

    { std::lock_guard<std::mutex> lk(s.sched_mtx); s.incoming.push_back(job); }
    s.sched_cv.notify_one();

    if (job->stream) {
        res.set_header("X-Audio-Sample-Rate", std::to_string(s.dac.sample_rate));
        res.set_header("X-Audio-Channels", "1");
        res.set_header("X-Audio-Format", "float32");
        res.set_chunked_content_provider("audio/pcm",
            [job](size_t /*offset*/, httplib::DataSink & sink) -> bool {
                for (;;) {
                    std::vector<float> block; bool done = false;
                    {
                        std::unique_lock<std::mutex> lk(job->m);
                        job->cv.wait(lk, [&]{ return !job->pcm.empty() || job->finished; });
                        if (!job->pcm.empty()) { block = std::move(job->pcm.front()); job->pcm.pop_front(); }
                        else done = job->finished;
                    }
                    if (!block.empty() &&
                        !sink.write((const char *) block.data(), block.size() * sizeof(float))) {
                        std::lock_guard<std::mutex> lk(job->m); job->client_dropped = true;
                        return false;                       // client dropped; worker evicts next tick
                    }
                    if (done) { sink.done(); return true; }
                }
            });
        return;
    }

    // buffered (wav, or pcm with stream=false): wait for completion, then assemble the body
    std::vector<float> audio;
    bool failed;
    {
        std::unique_lock<std::mutex> lk(job->m);
        job->cv.wait(lk, [&]{ return job->finished; });
        failed = job->failed;
        for (auto & b : job->pcm) audio.insert(audio.end(), b.begin(), b.end());
    }
    if (failed) { set_json(res, {{"error", "generation failed"}}, 500); return; }
    if (job->wav) {
        std::vector<uint8_t> body = dac_wav_bytes(audio, s.dac.sample_rate);
        res.set_content((const char *) body.data(), body.size(), "audio/wav");
    } else {
        res.set_header("X-Audio-Sample-Rate", std::to_string(s.dac.sample_rate));
        res.set_header("X-Audio-Channels", "1");
        res.set_header("X-Audio-Format", "float32");
        res.set_content((const char *) audio.data(), audio.size() * sizeof(float), "audio/pcm");
    }
}

// --------------------------------------------------------------------------- speaker endpoints

static json speaker_json(const CachedSpeaker & sp, bool is_default = false) {
    json o = {
        {"id", sp.id}, {"label", sp.label}, {"source_type", sp.source_type},
        {"original_name", sp.original_name}, {"dimension", (int) sp.emb.size()},
        {"created_at", sp.created}, {"has_preview", !sp.pcm24k.empty()},
    };
    if (is_default) { o["scope"] = "default"; o["is_default"] = true; }
    return o;
}

static void handle_speakers_post(ServerState & s, const httplib::Request & req, httplib::Response & res) {
    const std::string session = req.get_header_value("X-TTS-Session-ID");
    if (session.empty()) { set_json(res, {{"error", "X-TTS-Session-ID header required"}}, 400); return; }

    json j = json::parse(req.body, nullptr, false);
    if (j.is_discarded()) { set_json(res, {{"error", "invalid JSON body"}}, 400); return; }

    CachedSpeaker sp;
    sp.label         = jstr(j, "label", "speaker");
    sp.original_name = jstr(j, "speaker_audio_name");
    if (sp.original_name.empty()) sp.original_name = jstr(j, "speaker_embedding_name");

    std::string audio_b64 = jstr(j, "speaker_audio_base64");
    if (audio_b64.empty()) audio_b64 = jstr(j, "speaker_wav_base64");

    if (!audio_b64.empty()) {
        if (!s.have_spk) { set_json(res, {{"error", "speaker audio upload requires --spk <encoder.gguf>"}}, 400); return; }
        std::vector<uint8_t> bytes;
        if (!b64_decode(audio_b64, bytes)) { set_json(res, {{"error", "invalid speaker_audio_base64"}}, 400); return; }
        const auto tmp = temp_path(s, ".audio");
        FILE * f = fopen(tmp.string().c_str(), "wb");
        if (!f) { set_json(res, {{"error", "cannot write temp audio"}}, 500); return; }
        fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
        std::vector<float> pcm = spk_decode_audio_file(s.spk, tmp.string().c_str());
        std::error_code ec; std::filesystem::remove(tmp, ec);
        if (pcm.empty()) { set_json(res, {{"error", "ffmpeg failed to decode speaker audio"}}, 400); return; }
        { std::lock_guard<std::mutex> lk(s.spk_compute_mtx);
          sp.emb = spk_embed_from_pcm24k(s.spk, pcm.data(), (int) pcm.size()); }
        if ((int) sp.emb.size() != (int) s.model.hp.spk_dim) { set_json(res, {{"error", "speaker encoding failed"}}, 400); return; }
        sp.source_type = "audio";
        sp.pcm24k = std::move(pcm);   // keep decoded reference audio for /preview
    } else {
        std::string err;
        if (!resolve_speaker(s, j, session, sp.emb, err)) {
            set_json(res, {{"error", err.empty() ? std::string("no speaker_audio_base64 / speaker_embedding_base64 provided") : err}}, 400);
            return;
        }
        sp.source_type = "embedding_file";
    }

    sp.id = make_id(s, "spk_");
    sp.created = (double) s.id_ctr.load();
    {
        std::lock_guard<std::mutex> lk(s.spk_mtx);
        s.sessions[session][sp.id] = sp;
    }
    set_json(res, speaker_json(sp));
}

static void handle_speakers_get(ServerState & s, const httplib::Request & req, httplib::Response & res) {
    const std::string session = req.get_header_value("X-TTS-Session-ID");
    json arr = json::array();
    {
        std::lock_guard<std::mutex> lk(s.spk_mtx);
        auto it = s.sessions.find(session);
        if (it != s.sessions.end())
            for (auto & kv : it->second) arr.push_back(speaker_json(kv.second));
    }
    set_json(res, {{"speakers", arr}});
}

static void handle_speaker_preview(ServerState & s, const httplib::Request & req, httplib::Response & res) {
    const std::string session = req.get_header_value("X-TTS-Session-ID");
    const std::string id = req.matches[1];
    std::vector<float> pcm;
    {
        std::lock_guard<std::mutex> lk(s.spk_mtx);
        auto it = s.sessions.find(session);
        if (it != s.sessions.end()) {
            auto sit = it->second.find(id);
            if (sit != it->second.end()) pcm = sit->second.pcm24k;
        }
    }
    if (pcm.empty()) { set_json(res, {{"error", "no preview for speaker '" + id + "'"}}, 404); return; }
    std::vector<uint8_t> wav = dac_wav_bytes(pcm, s.spk.sr);
    res.set_content((const char *) wav.data(), wav.size(), "audio/wav");
}

// --------------------------------------------------------------------------- info endpoints

static void handle_capabilities(ServerState & s, httplib::Response & res) {
    const auto & hp = s.model.hp;
    const int n_feat = (int) hp.cond_quality_bucket_counts.size();
    json qfeat = json::array(), qbuckets = json::object(), qdrop = json::object();
    int qtotal = 0;                                     // total buckets across features (ref semantics)
    for (int i = 0; i < n_feat && i < 6; ++i) {
        const std::string fname = QFEATS[i];
        const int cnt = hp.cond_quality_bucket_counts[i];
        qtotal += cnt;
        qfeat.push_back(fname);
        qdrop[fname] = QUALITY_DROPOUT;
        json labels = json::array();
        auto it = QUALITY_LABELS.find(fname);
        if (it != QUALITY_LABELS.end() && (int) it->second.size() == cnt) {
            for (const auto & l : it->second) labels.push_back(l);            // human-readable ranges
        } else {
            for (int b = 0; b < cnt; ++b) labels.push_back(std::to_string(b)); // fallback: indices
        }
        qbuckets[fname] = labels;
    }
    json sr_labels = json::array();
    if ((int) SPEAKING_RATE_LABELS.size() == (int) hp.cond_speaking_rate_buckets) {
        for (const auto & l : SPEAKING_RATE_LABELS) sr_labels.push_back(l);
    } else {
        for (int b = 0; b < (int) hp.cond_speaking_rate_buckets; ++b) sr_labels.push_back(std::to_string(b));
    }
    json emotion_names = json::array();
    json emotion_axes = json::array();
    if (s.have_emotion && hp.spk_dim > 0) {
        for (const auto & name : zonos2_emotion_names(s.emotion)) emotion_names.push_back(name);
        for (const auto & name : zonos2_emotion_axes(s.emotion)) emotion_axes.push_back(name);
    }
    json caps = {
        {"text_normalization_enabled", false},          // byte-level tokenizer: no NeMo normalizer
        {"text_norm_languages", json::array()},
        {"speaker_enabled", hp.spk_dim > 0},
        {"speaker_embedding_dim", (int) hp.spk_dim},
        {"n_codebooks", (int) hp.n_codebooks},
        {"max_tokens", s.max_frames},
        {"speaking_rate_enabled", hp.cond_speaking_rate_buckets > 0},
        {"speaking_rate_num_buckets", (int) hp.cond_speaking_rate_buckets},
        {"speaking_rate_buckets", sr_labels},
        {"quality_enabled", n_feat > 0},
        {"quality_num_buckets", qtotal},
        {"quality_features", qfeat},
        {"quality_buckets", qbuckets},
        {"quality_dropout_by_feature", qdrop},
        {"default_quality_buckets", {{QFEATS[std::max(0, std::min(5, (int) hp.cond_default_quality_feature))], hp.cond_default_quality_bucket}}},
        {"speaker_background_token_enabled", hp.cond_speaker_bg_buckets > 0},
        {"accurate_mode_token_enabled", hp.cond_accurate_buckets > 0},
        {"speaker_audio_upload", s.have_spk},
        {"speaker_embedding_upload", hp.spk_dim > 0},
        {"speaker_embedding_cache", true},
        {"speaker_embedding_blend", false},             // SLERP blend: not in this port
        {"default_voices_enabled", false},
        {"emotion_enabled", s.have_emotion && hp.spk_dim > 0},
        {"emotion_names", emotion_names},
        {"emotion_axes", emotion_axes},
        {"emotion_calibrated", s.have_emotion && s.emotion.has_calibration},
    };
    set_json(res, caps);
}

// --------------------------------------------------------------------------- main

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseeko(f, 0, SEEK_END); long n = ftello(f); fseeko(f, 0, SEEK_SET);
    std::string s((size_t) n, '\0');
    if (fread(&s[0], 1, (size_t) n, f) != (size_t) n) { fclose(f); return {}; }
    fclose(f);
    return s;
}

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s <model.gguf> --dac <dac.gguf> [--spk <encoder.gguf>] [options]\n"
        "  --host H            bind address (default 127.0.0.1)\n"
        "  --port P            port (default 1919)\n"
        "  --gpu | --cpu       backend (default cpu)\n"
        "  --dac-cpu           run the DAC decoder on CPU even with --gpu (isolates the\n"
        "                      backbone graph; makes streamed audio bit-exact; default on Metal)\n"
        "  --dac-gpu           force GPU DAC (overrides the Metal CPU-DAC default)\n"
        "  --max N             max frames per request (default 2000, ~23s)\n"
        "  --batch N           continuous-batching width / concurrent syntheses (default 8). A\n"
        "                      decode-graph ladder runs the narrowest width covering the active\n"
        "                      slots, so solo requests stay fast at any N; KV cache is ~0.18 GB/slot.\n"
        "                      N>=32 unlocks the expert GEMM kernel for peak aggregate throughput.\n"
        "  --dac-threads N     DAC decode pool lanes, parallel decode across requests (default 4)\n"
        "  --stream-block N    frames per streamed PCM block (default 40)\n"
        "  --stream-context N  conv-context frames each side, >=16 seam-free (default 24)\n"
        "  --tts-emotion-directions-dir DIR\n"
        "                      load emotion direction .npy files (default emotion_directions; empty disables)\n"
        "  --ui PATH           web UI html to serve at / (default web/tts_ui.html)\n", a0);
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    ServerState s;
    std::string model_path = argv[1], dac_path, spk_path, host = "127.0.0.1";
    int port = 1919;
    bool dac_cpu = false, dac_place_set = false;   // dac_place_set: user pinned DAC placement explicitly
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--dac" && i + 1 < argc) dac_path = argv[++i];
        else if (a == "--spk" && i + 1 < argc) spk_path = argv[++i];
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
        else if (a == "--gpu") s.use_gpu = true;
        else if (a == "--cpu") s.use_gpu = false;
        else if (a == "--dac-cpu") { dac_cpu = true;  dac_place_set = true; }
        else if (a == "--dac-gpu") { dac_cpu = false; dac_place_set = true; }
        else if (a == "--max" && i + 1 < argc) s.max_frames = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) s.batch_slots = std::max(1, atoi(argv[++i]));
        else if (a == "--dac-threads" && i + 1 < argc) s.dac_threads = std::max(1, atoi(argv[++i]));
        else if (a == "--stream-block" && i + 1 < argc) s.stream_block = atoi(argv[++i]);
        else if (a == "--stream-context" && i + 1 < argc) s.stream_ctx = atoi(argv[++i]);
        else if (a == "--tts-emotion-directions-dir" && i + 1 < argc) s.emotion_dir = argv[++i];
        else if (a == "--ui" && i + 1 < argc) s.ui_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (dac_path.empty()) { fprintf(stderr, "error: --dac <dac.gguf> is required\n"); return 1; }
    s.prof = getenv("ZONOS2_PROFILE") != nullptr;

    if (!zonos2_model_load(s.model, model_path.c_str(), s.use_gpu)) { fprintf(stderr, "failed to load model\n"); return 1; }
    {
        std::string err;
        s.have_emotion = zonos2_emotion_load(s.emotion, s.emotion_dir, err);
        if (s.have_emotion) {
            fprintf(stderr, "zonos2-server: emotion directions=%s space=%s names=%zu axes=%zu calibrated=%s\n",
                    s.emotion_dir.c_str(), s.emotion.space.c_str(), s.emotion.named.size(), s.emotion.axes.size(),
                    s.emotion.has_calibration ? "yes" : "no");
        } else if (!s.emotion_dir.empty()) {
            fprintf(stderr, "zonos2-server: emotion disabled (%s)\n", err.c_str());
        }
    }

    // On Metal the backbone and DAC share one GPU; running DAC there starves the backbone decode
    // (measured ~2.5x slower end-to-end on M3). Default DAC to CPU when on a Metal GPU unless the
    // user pinned placement with --dac-cpu/--dac-gpu. (For high-concurrency --batch loads GPU DAC
    // may still win on throughput; override with --dac-gpu then.)
    if (s.use_gpu && !dac_place_set) {
        const char * bname = ggml_backend_name(s.model.backend);
        if (bname && (strstr(bname, "Metal") || strstr(bname, "MTL"))) {
            dac_cpu = true;
            fprintf(stderr, "zonos2-server: Metal GPU -> DAC defaults to CPU (avoids backbone contention; --dac-gpu to override)\n");
        }
    }
    if (!dac_load(s.dac, dac_path.c_str(), s.use_gpu && !dac_cpu)) { fprintf(stderr, "failed to load dac\n"); return 1; }
    if (!spk_path.empty()) {
        if (!spk_load(s.spk, spk_path.c_str())) { fprintf(stderr, "failed to load speaker encoder\n"); return 1; }
        s.have_spk = true;
    }

    // DAC decode pool: one dac_model instance + thread per lane (ggml backends are not reentrant).
    s.dac_lanes = std::vector<DacLane>(s.dac_threads);
    for (int i = 0; i < s.dac_threads; ++i) {
        if (!dac_load(s.dac_lanes[i].dac, dac_path.c_str(), s.use_gpu && !dac_cpu)) {
            fprintf(stderr, "failed to load dac for pool lane %d\n", i); return 1;
        }
    }
    for (int i = 0; i < s.dac_threads; ++i)
        s.dac_lanes[i].th = std::thread(dac_lane_loop, std::ref(s), std::ref(s.dac_lanes[i]));

    s.worker = std::thread(worker_loop, std::ref(s));   // owns backbone; produces codes for the pool

    httplib::Server svr;
    svr.set_payload_max_length(64ull * 1024 * 1024);   // allow multi-MB speaker uploads
    svr.set_read_timeout(120);
    svr.set_write_timeout(600);                          // long synthesis streams

    svr.Get ("/health", [](const httplib::Request &, httplib::Response & res) { set_json(res, {{"status", "ok"}}); });
    svr.Get ("/v1",     [](const httplib::Request &, httplib::Response & res) { set_json(res, {{"status", "ok"}}); });
    svr.Post("/v1",     [](const httplib::Request &, httplib::Response & res) { set_json(res, {{"status", "ok"}}); });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        set_json(res, {{"object", "list"}, {"data", json::array({ json{
            {"id", "zonos2"}, {"object", "model"}, {"created", 0}, {"owned_by", "zonos2.cpp"}, {"root", "zonos2"} } })}});
    });

    svr.Get("/tts/capabilities", [&](const httplib::Request &, httplib::Response & res) { handle_capabilities(s, res); });

    svr.Get ("/tts/speakers", [&](const httplib::Request & req, httplib::Response & res) { handle_speakers_get(s, req, res); });
    svr.Post("/tts/speakers", [&](const httplib::Request & req, httplib::Response & res) { handle_speakers_post(s, req, res); });
    svr.Get (R"(/tts/speakers/([^/]+)/preview)", [&](const httplib::Request & req, httplib::Response & res) { handle_speaker_preview(s, req, res); });

    svr.Post("/tts/generate",    [&](const httplib::Request & req, httplib::Response & res) { handle_generate(s, req, res); });
    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        // OpenAI compatibility: input->text, response_format->format (pcm streams, wav buffers)
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { set_json(res, {{"error", "invalid JSON body"}}, 400); return; }
        const std::string rf = jstr(j, "response_format", "pcm");
        if (rf != "pcm" && rf != "wav") { set_json(res, {{"error", "unsupported response_format (use pcm or wav)"}}, 400); return; }
        j["format"] = rf;
        j["stream"] = (rf == "pcm");
        httplib::Request r2 = req; r2.body = j.dump();
        handle_generate(s, r2, res);
    });

    svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        std::string html = read_file(s.ui_path);
        if (html.empty()) { res.set_content("<h1>zonos2-server</h1><p>UI not found at " + s.ui_path + "</p>", "text/html"); return; }
        res.set_content(html, "text/html; charset=utf-8");
    });

    // permissive CORS (mirrors the reference server)
    svr.set_post_routing_handler([](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Allow-Methods", "*");
    });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    fprintf(stderr, "zonos2-server: backbone=%s dac=%s spk=%s backend=%s batch=%d dac-threads=%d\n",
            model_path.c_str(), dac_path.c_str(), s.have_spk ? spk_path.c_str() : "(none)",
            s.use_gpu ? "GPU" : "CPU", s.batch_slots, s.dac_threads);
    fprintf(stderr, "zonos2-server: listening on http://%s:%d  (UI: %s)\n", host.c_str(), port, s.ui_path.c_str());
    const bool ok = svr.listen(host, port);

    s.stop.store(true);                 // stop the worker + DAC pool before returning
    s.sched_cv.notify_all();
    if (s.worker.joinable()) s.worker.join();
    for (auto & lane : s.dac_lanes) {
        { std::lock_guard<std::mutex> lk(lane.m); }
        lane.cv.notify_all();
        if (lane.th.joinable()) lane.th.join();
        dac_free(lane.dac);
    }
    if (!ok) { fprintf(stderr, "error: failed to bind %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
