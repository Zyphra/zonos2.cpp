#pragma once

#include <map>
#include <string>
#include <vector>

struct zonos2_emotion_directions {
    int dim = 0;
    int input_dim = 0;
    std::string space = "raw"; // raw | lda | proj
    bool has_ref_base_norm = false;
    float ref_base_norm = 0.0f;

    std::map<std::string, std::vector<float>> named;
    std::map<std::string, std::vector<float>> axes;
    std::vector<std::string> named_order;
    std::vector<std::string> axes_order;

    // Optional calibration.json: strength(speaker, emotion) falls back to
    // default[emotion], then global_default, matching the Python runtime.
    std::map<std::string, std::map<std::string, float>> calibration_by_speaker;
    std::map<std::string, float> calibration_default;
    float calibration_global_default = 1.0f;
    bool has_calibration = false;

    // Optional LDA-space injection matrices.
    std::vector<float> lda_weight;
    std::vector<float> lda_bias;
    std::vector<float> lda_pinv;
    int lda_weight_rows = 0, lda_weight_cols = 0;
    int lda_pinv_rows = 0, lda_pinv_cols = 0;
};

struct zonos2_emotion_request {
    std::map<std::string, float> sliders;
    float valence = 0.0f;
    float arousal = 0.0f;
    float strength = 1.0f;
    std::string speaker_key;
};

bool zonos2_emotion_available(const zonos2_emotion_directions & d);
std::vector<std::string> zonos2_emotion_names(const zonos2_emotion_directions & d);
std::vector<std::string> zonos2_emotion_axes(const zonos2_emotion_directions & d);

// Returns false when manifest.json is missing or invalid. Missing directories are treated as
// "not available" and reported through err for logging, not as fatal model-load failures.
bool zonos2_emotion_load(zonos2_emotion_directions & d, const std::string & dir, std::string & err);

// Apply a request to a speaker vector. For space="proj", speaker is left unchanged and
// hidden_delta receives a [hidden] vector to add after speaker projection. For raw/lda,
// speaker is modified and hidden_delta is empty.
bool zonos2_emotion_apply(const zonos2_emotion_directions & d,
                          const zonos2_emotion_request & req,
                          std::vector<float> & speaker,
                          std::vector<float> & hidden_delta,
                          std::string & err);
