// exe-path.h — locate the running executable's directory, portably. Used to resolve
// assets and helper binaries (web UI, a bundled/downloaded ffmpeg) relative to the
// binary rather than the process CWD, which is arbitrary for a double-clicked app.
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

static inline std::filesystem::path zonos2_exe_dir() {
    namespace fs = std::filesystem;
    std::error_code ec;
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return fs::path(std::wstring(buf, n)).parent_path();
#elif defined(__APPLE__)
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    std::string b(sz, '\0');
    if (_NSGetExecutablePath(b.data(), &sz) == 0) {
        fs::path p = fs::canonical(b.c_str(), ec);
        if (!ec) return p.parent_path();
    }
#else
    fs::path p = fs::canonical("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
#endif
    return fs::current_path(ec);
}
