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
#include "dac.h"
#include "spk-encoder.h"
#include "npy.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

// --------------------------------------------------------------------------- state

struct CachedSpeaker {
    std::string id, label, source_type, original_name;
    std::vector<float> emb;     // [spk_dim]
    std::vector<float> pcm24k;  // reference audio for preview (empty for embedding-only)
    double created = 0;
};

struct ServerState {
    zonos2_model model;
    dac_model    dac;
    spk_model    spk;
    bool have_spk = false;
    bool use_gpu  = false;

    int max_frames   = 2000;   // ~23s ceiling; per-request max_tokens clamps below this
    int stream_block = 40;     // frames per streamed PCM block (~0.46s)
    int stream_ctx   = 24;     // conv-context frames each side (>=16 is seam-free); see dac_decode_window
    std::string ui_path = "web/tts_ui.html";

    std::mutex gpu_mtx;        // serialize generate + decode (single in-flight synthesis)

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
    int spk_pos = 0;
    double fade_out_ms = 0;
    bool stream = true;
    std::string format = "pcm";         // "pcm" (float32) | "wav"
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
    return true;
}

// --------------------------------------------------------------------------- synthesis

// Buffered: generate all frames, decode once, return audio bytes + content type.
static bool synth_buffered(ServerState & s, const GenReq & req,
                           std::vector<uint8_t> & body, std::string & content_type) {
    int n0 = 0, spk_pos = -1;
    zonos2_prompt_options opt = req.opt;
    const float * spk_ptr = req.spk.empty() ? nullptr : req.spk.data();
    std::vector<int32_t> ids = zonos2_build_prompt(s.model, req.text, opt, n0, spk_pos);
    std::vector<float> idf(ids.begin(), ids.end());
    const int ncb = (int) s.model.hp.n_codebooks;

    std::vector<int32_t> codes; int eos_frame = -1;
    const int nf = zonos2_generate(s.model, idf.data(), n0, req.max_frames, req.sp, codes, eos_frame,
                                   /*use_kv=*/true, spk_ptr, spk_pos >= 0 ? spk_pos : 0);
    if (nf <= 0) return false;

    std::vector<float> audio;
    if (!dac_decode(s.dac, codes.data(), nf, ncb, eos_frame, audio)) return false;
    apply_fade(audio, req.fade_out_ms, s.dac.sample_rate);

    if (req.format == "wav") {
        body = dac_wav_bytes(audio, s.dac.sample_rate);
        content_type = "audio/wav";
    } else {
        body.resize(audio.size() * sizeof(float));
        memcpy(body.data(), audio.data(), body.size());
        content_type = "audio/pcm";
    }
    return true;
}

// Streaming: generate with a per-frame callback, decode ready blocks via dac_decode_window,
// and write float32 PCM to the sink as it is produced. Returns false if the client dropped.
static bool synth_stream(ServerState & s, const GenReq & req, httplib::DataSink & sink) {
    int n0 = 0, spk_pos = -1;
    zonos2_prompt_options opt = req.opt;
    const float * spk_ptr = req.spk.empty() ? nullptr : req.spk.data();
    std::vector<int32_t> ids = zonos2_build_prompt(s.model, req.text, opt, n0, spk_pos);
    std::vector<float> idf(ids.begin(), ids.end());
    const int ncb = (int) s.model.hp.n_codebooks;

    const int BLOCK = s.stream_block, CTX = s.stream_ctx, SHEAR = ncb - 1;
    std::vector<int32_t> codes;
    int eos_frame = -1, emitted = 0;
    bool client_ok = true;

    auto on_frame = [&](int idx, const int32_t * /*fr*/, int /*ncb_*/) -> bool {
        const int H = idx + 1;   // frames generated so far (codes already appended)
        while (H - emitted >= BLOCK + CTX + SHEAR) {
            const int f_hi = H - CTX - SHEAR;                  // hold back conv ctx + shear lookahead
            std::vector<float> pcm;
            if (!dac_decode_window(s.dac, codes.data(), H, emitted, f_hi, CTX, CTX, pcm)) return false;
            if (!sink.write((const char *) pcm.data(), pcm.size() * sizeof(float))) { client_ok = false; return false; }
            emitted = f_hi;
        }
        return true;
    };

    const int nf = zonos2_generate(s.model, idf.data(), n0, req.max_frames, req.sp, codes, eos_frame,
                                   /*use_kv=*/true, spk_ptr, spk_pos >= 0 ? spk_pos : 0, nullptr, on_frame);

    // flush the tail: decode [emitted, end) with Rc=0 to match a full decode truncated at eos
    const int end = (eos_frame >= 0 && eos_frame < nf) ? eos_frame : nf;
    if (client_ok && end > emitted) {
        std::vector<float> pcm;
        if (dac_decode_window(s.dac, codes.data(), nf, emitted, end, CTX, /*Rc=*/0, pcm)) {
            apply_fade(pcm, req.fade_out_ms, s.dac.sample_rate);
            if (!sink.write((const char *) pcm.data(), pcm.size() * sizeof(float))) client_ok = false;
        }
    }
    return client_ok;
}

// Shared handler for /tts/generate and /v1/audio/speech.
static void handle_generate(ServerState & s, const httplib::Request & req, httplib::Response & res) {
    json j = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) { set_json(res, {{"error", "invalid JSON body"}}, 400); return; }

    const std::string session = req.get_header_value("X-TTS-Session-ID");
    GenReq gr; std::string err;
    if (!parse_gen_req(s, j, session, gr, err)) { set_json(res, {{"error", err}}, 400); return; }
    res.set_header("X-Seed", std::to_string(gr.sp.seed));   // report the seed used (reproduce by sending it back)

    if (gr.stream && gr.format != "wav") {
        res.set_header("X-Audio-Sample-Rate", std::to_string(s.dac.sample_rate));
        res.set_header("X-Audio-Channels", "1");
        res.set_header("X-Audio-Format", "float32");
        res.set_chunked_content_provider("audio/pcm",
            [&s, gr](size_t /*offset*/, httplib::DataSink & sink) -> bool {
                std::lock_guard<std::mutex> lk(s.gpu_mtx);
                synth_stream(s, gr, sink);
                sink.done();
                return true;
            });
        return;
    }

    // buffered (wav, or pcm with stream=false)
    std::vector<uint8_t> body; std::string ct;
    bool ok;
    { std::lock_guard<std::mutex> lk(s.gpu_mtx); ok = synth_buffered(s, gr, body, ct); }
    if (!ok) { set_json(res, {{"error", "generation failed"}}, 500); return; }
    if (ct == "audio/pcm") {
        res.set_header("X-Audio-Sample-Rate", std::to_string(s.dac.sample_rate));
        res.set_header("X-Audio-Channels", "1");
        res.set_header("X-Audio-Format", "float32");
    }
    res.set_content((const char *) body.data(), body.size(), ct);
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
        "                      backbone CUDA graph; makes streamed audio bit-exact)\n"
        "  --max N             max frames per request (default 2000, ~23s)\n"
        "  --stream-block N    frames per streamed PCM block (default 40)\n"
        "  --stream-context N  conv-context frames each side, >=16 seam-free (default 24)\n"
        "  --ui PATH           web UI html to serve at / (default web/tts_ui.html)\n", a0);
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    ServerState s;
    std::string model_path = argv[1], dac_path, spk_path, host = "127.0.0.1";
    int port = 1919;
    bool dac_cpu = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--dac" && i + 1 < argc) dac_path = argv[++i];
        else if (a == "--spk" && i + 1 < argc) spk_path = argv[++i];
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
        else if (a == "--gpu") s.use_gpu = true;
        else if (a == "--cpu") s.use_gpu = false;
        else if (a == "--dac-cpu") dac_cpu = true;
        else if (a == "--max" && i + 1 < argc) s.max_frames = atoi(argv[++i]);
        else if (a == "--stream-block" && i + 1 < argc) s.stream_block = atoi(argv[++i]);
        else if (a == "--stream-context" && i + 1 < argc) s.stream_ctx = atoi(argv[++i]);
        else if (a == "--ui" && i + 1 < argc) s.ui_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (dac_path.empty()) { fprintf(stderr, "error: --dac <dac.gguf> is required\n"); return 1; }

    if (!zonos2_model_load(s.model, model_path.c_str(), s.use_gpu)) { fprintf(stderr, "failed to load model\n"); return 1; }
    if (!dac_load(s.dac, dac_path.c_str(), s.use_gpu && !dac_cpu)) { fprintf(stderr, "failed to load dac\n"); return 1; }
    if (!spk_path.empty()) {
        if (!spk_load(s.spk, spk_path.c_str())) { fprintf(stderr, "failed to load speaker encoder\n"); return 1; }
        s.have_spk = true;
    }

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

    fprintf(stderr, "zonos2-server: backbone=%s dac=%s spk=%s backend=%s\n",
            model_path.c_str(), dac_path.c_str(), s.have_spk ? spk_path.c_str() : "(none)", s.use_gpu ? "GPU" : "CPU");
    fprintf(stderr, "zonos2-server: listening on http://%s:%d  (UI: %s)\n", host.c_str(), port, s.ui_path.c_str());
    if (!svr.listen(host, port)) { fprintf(stderr, "error: failed to bind %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
