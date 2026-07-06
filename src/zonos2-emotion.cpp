#include "zonos2-emotion.h"

#include "npy.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

namespace {

static std::string read_text(const std::filesystem::path & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool load_vector(const std::filesystem::path & path, int expected_dim,
                        const std::string & name, std::vector<float> & out,
                        std::string & err) {
    std::vector<int64_t> shape;
    if (!npy::load_f32(path.string(), out, shape)) {
        err = "failed to load emotion direction '" + name + "' from " + path.string();
        return false;
    }
    if ((int) out.size() != expected_dim) {
        err = "emotion direction '" + name + "' has dim " + std::to_string(out.size()) +
              ", expected " + std::to_string(expected_dim);
        out.clear();
        return false;
    }
    return true;
}

static bool load_matrix(const std::filesystem::path & path, std::vector<float> & out,
                        int & rows, int & cols, const char * name, std::string & err) {
    std::vector<int64_t> shape;
    if (!npy::load_f32(path.string(), out, shape)) {
        err = std::string("failed to load emotion LDA matrix '") + name + "' from " + path.string();
        return false;
    }
    if (shape.size() != 2) {
        err = std::string("emotion LDA matrix '") + name + "' must be 2-D";
        out.clear();
        return false;
    }
    rows = (int) shape[0];
    cols = (int) shape[1];
    return true;
}

static float l2_norm(const std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double) x * x;
    return (float) std::sqrt(s);
}

static void rescale_to(std::vector<float> & v, float target) {
    const float cur = l2_norm(v);
    if (cur <= 1e-8f) return;
    const float scale = target / cur;
    for (float & x : v) x *= scale;
}

static float calibration_strength(const zonos2_emotion_directions & d,
                                  const std::string & speaker_key,
                                  const std::string & emotion) {
    if (!speaker_key.empty()) {
        auto sit = d.calibration_by_speaker.find(speaker_key);
        if (sit != d.calibration_by_speaker.end()) {
            auto eit = sit->second.find(emotion);
            if (eit != sit->second.end()) return eit->second;
        }
    }
    auto dit = d.calibration_default.find(emotion);
    if (dit != d.calibration_default.end()) return dit->second;
    return d.calibration_global_default;
}

static std::string available_names(const zonos2_emotion_directions & d) {
    std::string s;
    for (const auto & kv : d.named) {
        if (!s.empty()) s += ", ";
        s += kv.first;
    }
    for (const auto & kv : d.axes) {
        if (!s.empty()) s += ", ";
        s += kv.first;
    }
    return s;
}

static bool add_weighted(std::vector<float> & delta, const std::vector<float> & vec, float w) {
    if (vec.size() != delta.size()) return false;
    for (size_t i = 0; i < delta.size(); ++i) delta[i] += w * vec[i];
    return true;
}

static bool combine_delta(const zonos2_emotion_directions & d,
                          const zonos2_emotion_request & req,
                          std::vector<float> & delta, bool & requested,
                          std::string & err) {
    delta.assign((size_t) d.dim, 0.0f);
    requested = false;

    for (const auto & kv : req.sliders) {
        const std::string & name = kv.first;
        const float w0 = kv.second;
        if (w0 == 0.0f) continue;
        const std::vector<float> * vec = nullptr;
        auto nit = d.named.find(name);
        if (nit != d.named.end()) vec = &nit->second;
        auto ait = d.axes.find(name);
        if (!vec && ait != d.axes.end()) vec = &ait->second;
        if (!vec) {
            err = "unknown emotion direction '" + name + "' (available: " + available_names(d) + ")";
            return false;
        }
        const float w = w0 * calibration_strength(d, req.speaker_key, name);
        if (!add_weighted(delta, *vec, w)) {
            err = "emotion direction '" + name + "' has inconsistent dimensions";
            return false;
        }
        requested = true;
    }

    const std::pair<const char *, float> axes[] = {
        {"valence", req.valence},
        {"arousal", req.arousal},
    };
    for (const auto & axis : axes) {
        if (axis.second == 0.0f) continue;
        auto it = d.axes.find(axis.first);
        if (it == d.axes.end()) {
            err = std::string("emotion axis '") + axis.first + "' is not available";
            return false;
        }
        const float w = axis.second * calibration_strength(d, req.speaker_key, axis.first);
        if (!add_weighted(delta, it->second, w)) {
            err = std::string("emotion axis '") + axis.first + "' has inconsistent dimensions";
            return false;
        }
        requested = true;
    }

    if (requested && req.strength != 1.0f) {
        for (float & x : delta) x *= req.strength;
    }
    return true;
}

static bool apply_raw(const zonos2_emotion_directions & d, const std::vector<float> & delta,
                      std::vector<float> & speaker, std::string & err) {
    if ((int) speaker.size() != d.input_dim || d.dim != d.input_dim) {
        err = "raw emotion directions require speaker dim " + std::to_string(d.input_dim) +
              ", got " + std::to_string(speaker.size());
        return false;
    }
    const float target_norm = l2_norm(speaker);
    for (size_t i = 0; i < speaker.size(); ++i) speaker[i] += delta[i];
    rescale_to(speaker, target_norm);
    return true;
}

static bool apply_lda(const zonos2_emotion_directions & d, const std::vector<float> & delta,
                      std::vector<float> & speaker, std::string & err) {
    if ((int) speaker.size() != d.input_dim) {
        err = "LDA emotion directions require speaker dim " + std::to_string(d.input_dim) +
              ", got " + std::to_string(speaker.size());
        return false;
    }
    if (d.lda_weight_rows != d.dim || d.lda_weight_cols != d.input_dim ||
        d.lda_pinv_rows != d.input_dim || d.lda_pinv_cols != d.dim ||
        (int) d.lda_bias.size() != d.dim) {
        err = "LDA emotion matrix shapes do not match manifest dimensions";
        return false;
    }

    std::vector<float> lda((size_t) d.dim, 0.0f);
    for (int r = 0; r < d.dim; ++r) {
        double v = d.lda_bias[(size_t) r];
        for (int c = 0; c < d.input_dim; ++c)
            v += (double) d.lda_weight[(size_t) r * d.input_dim + c] * speaker[(size_t) c];
        lda[(size_t) r] = (float) v;
    }
    const float target_norm = l2_norm(lda);
    for (int r = 0; r < d.dim; ++r) lda[(size_t) r] += delta[(size_t) r];
    rescale_to(lda, target_norm);

    std::vector<float> raw((size_t) d.input_dim, 0.0f);
    for (int i = 0; i < d.input_dim; ++i) {
        double v = 0.0;
        for (int r = 0; r < d.dim; ++r) {
            const float centered = lda[(size_t) r] - d.lda_bias[(size_t) r];
            v += (double) d.lda_pinv[(size_t) i * d.dim + r] * centered;
        }
        raw[(size_t) i] = (float) v;
    }
    speaker = std::move(raw);
    return true;
}

} // namespace

bool zonos2_emotion_available(const zonos2_emotion_directions & d) {
    return d.dim > 0 && (!d.named.empty() || !d.axes.empty());
}

std::vector<std::string> zonos2_emotion_names(const zonos2_emotion_directions & d) {
    return d.named_order;
}

std::vector<std::string> zonos2_emotion_axes(const zonos2_emotion_directions & d) {
    return d.axes_order;
}

bool zonos2_emotion_load(zonos2_emotion_directions & d, const std::string & dir,
                         std::string & err) {
    d = zonos2_emotion_directions();
    err.clear();
    if (dir.empty()) {
        err = "emotion directions disabled";
        return false;
    }

    const std::filesystem::path root = std::filesystem::path(dir);
    const std::filesystem::path manifest_path = root / "manifest.json";
    const std::string manifest_text = read_text(manifest_path);
    if (manifest_text.empty()) {
        err = "no emotion directions manifest at " + manifest_path.string();
        return false;
    }

    json manifest = json::parse(manifest_text, nullptr, false);
    if (manifest.is_discarded() || !manifest.is_object()) {
        err = "invalid emotion directions manifest at " + manifest_path.string();
        return false;
    }

    d.dim = manifest.value("dim", 0);
    d.input_dim = manifest.value("input_dim", d.dim);
    d.space = manifest.value("space", std::string("raw"));
    if (manifest.contains("ref_base_norm") && manifest["ref_base_norm"].is_number()) {
        d.has_ref_base_norm = true;
        d.ref_base_norm = manifest["ref_base_norm"].get<float>();
    }
    if (d.dim <= 0 || d.input_dim <= 0) {
        err = "emotion manifest has invalid dimensions";
        return false;
    }

    if (!manifest.contains("directions") || !manifest["directions"].is_object()) {
        err = "emotion manifest has no directions object";
        return false;
    }

    for (auto it = manifest["directions"].begin(); it != manifest["directions"].end(); ++it) {
        const std::string name = it.key();
        const json & entry = it.value();
        if (!entry.is_object() || !entry.contains("file") || !entry["file"].is_string()) {
            err = "emotion direction '" + name + "' is missing a file";
            return false;
        }
        std::vector<float> vec;
        if (!load_vector(root / entry["file"].get<std::string>(), d.dim, name, vec, err)) {
            return false;
        }
        const std::string kind = entry.value("kind", std::string());
        if (kind == "axis" || name == "valence" || name == "arousal") {
            d.axes[name] = std::move(vec);
            d.axes_order.push_back(name);
        } else {
            d.named[name] = std::move(vec);
            d.named_order.push_back(name);
        }
    }

    if (d.space == "lda") {
        const json lda = manifest.value("lda", json::object());
        if (!lda.is_object() || !lda.contains("weight") || !lda.contains("bias") || !lda.contains("pinv")) {
            err = "LDA emotion directions require lda.weight, lda.bias, and lda.pinv";
            return false;
        }
        if (!load_matrix(root / lda["weight"].get<std::string>(), d.lda_weight,
                         d.lda_weight_rows, d.lda_weight_cols, "weight", err)) return false;
        std::vector<int64_t> bias_shape;
        if (!npy::load_f32((root / lda["bias"].get<std::string>()).string(), d.lda_bias, bias_shape)) {
            err = "failed to load emotion LDA bias";
            return false;
        }
        if (!load_matrix(root / lda["pinv"].get<std::string>(), d.lda_pinv,
                         d.lda_pinv_rows, d.lda_pinv_cols, "pinv", err)) return false;
    }

    const std::filesystem::path cal_path = root / "calibration.json";
    const std::string cal_text = read_text(cal_path);
    if (!cal_text.empty()) {
        json cal = json::parse(cal_text, nullptr, false);
        if (cal.is_object()) {
            d.has_calibration = true;
            d.calibration_global_default = cal.value("global_default", 1.0f);
            if (cal.contains("default") && cal["default"].is_object()) {
                for (auto it = cal["default"].begin(); it != cal["default"].end(); ++it) {
                    if (it.value().is_number()) d.calibration_default[it.key()] = it.value().get<float>();
                }
            }
            if (cal.contains("by_speaker") && cal["by_speaker"].is_object()) {
                for (auto sit = cal["by_speaker"].begin(); sit != cal["by_speaker"].end(); ++sit) {
                    if (!sit.value().is_object()) continue;
                    auto & dst = d.calibration_by_speaker[sit.key()];
                    for (auto eit = sit.value().begin(); eit != sit.value().end(); ++eit) {
                        if (eit.value().is_number()) dst[eit.key()] = eit.value().get<float>();
                    }
                }
            }
        } else {
            fprintf(stderr, "zonos2: ignoring invalid emotion calibration at %s\n", cal_path.string().c_str());
        }
    }

    return zonos2_emotion_available(d);
}

bool zonos2_emotion_apply(const zonos2_emotion_directions & d,
                          const zonos2_emotion_request & req,
                          std::vector<float> & speaker,
                          std::vector<float> & hidden_delta,
                          std::string & err) {
    hidden_delta.clear();
    err.clear();
    if (!zonos2_emotion_available(d)) {
        err = "emotion control requested but no emotion directions are loaded";
        return false;
    }
    if (speaker.empty()) return true;

    std::vector<float> delta;
    bool requested = false;
    if (!combine_delta(d, req, delta, requested, err)) return false;
    if (!requested || req.strength == 0.0f) return true;

    if (d.space == "proj") {
        hidden_delta = std::move(delta);
        return true;
    }
    if (d.space == "raw") return apply_raw(d, delta, speaker, err);
    if (d.space == "lda") return apply_lda(d, delta, speaker, err);

    err = "unsupported emotion direction space '" + d.space + "'";
    return false;
}
