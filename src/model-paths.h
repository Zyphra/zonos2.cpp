// model-paths.h — companion-GGUF discovery and download hints for the CLIs.
// dac.gguf / spk-encoder.gguf are conventionally siblings of the backbone GGUF;
// when the flags are omitted we look there before giving up.
#pragma once

#include <cstdio>
#include <string>

static inline bool mp_file_exists(const std::string & p) {
    if (FILE * f = fopen(p.c_str(), "rb")) { fclose(f); return true; }
    return false;
}

static inline std::string mp_parent_dir(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
}

// Sibling of `backbone` named `name` if it exists, else "".
static inline std::string mp_find_companion(const std::string & backbone, const char * name) {
    const std::string cand = mp_parent_dir(backbone) + "/" + name;
    return mp_file_exists(cand) ? cand : std::string();
}

static inline void mp_print_download_hint(const char * name, const std::string & tried) {
    fprintf(stderr,
        "error: %s not found (tried: %s)\n"
        "hint: download it with\n"
        "  curl -L -o %s https://huggingface.co/Zyphra/ZONOS2-GGUF/resolve/main/%s\n"
        "or run the bundled start-zonos2 script, which fetches all models automatically.\n",
        name, tried.c_str(), tried.c_str(), name);
}
