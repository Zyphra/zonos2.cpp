// exe-path.h — locate the running executable's directory, portably. Used to resolve
// assets and helper binaries (web UI, a bundled/downloaded ffmpeg) relative to the
// binary rather than the process CWD, which is arbitrary for a double-clicked app.
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
// This header pulls <windows.h>, and translation units that also use httplib (app.cpp)
// include it before <httplib.h>'s <winsock2.h>. Left alone, <windows.h> drags in the
// legacy <winsock.h> (v1), which then hard-conflicts with <winsock2.h>. Block just
// winsock1 via _WINSOCKAPI_ (not WIN32_LEAN_AND_MEAN, which would also strip the OLE/COM
// headers saucer's WebView2 backend needs) and match httplib's NOMINMAX.
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
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
