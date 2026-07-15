// zonos2-app — native desktop shell around zonos2-server (saucer webview).
//
// The whole HTTP server runs in-process on a background thread (src/server.cpp via
// src/server-embed.h) on an ephemeral 127.0.0.1 port; the window is a platform
// webview (WebView2 / WKWebView / WebKitGTK) pointed at it, so the app UI is the
// exact same web/tts_ui.html the server serves to a browser — and the HTTP API
// stays reachable while the app runs.
//
// Flow: with server-style CLI args they pass through verbatim (power users).
// Otherwise a saved config (app.json in the platform config dir) boots straight
// into a loading splash -> UI; missing/invalid config (or --setup) shows a setup
// page with native .gguf pickers first. Closing the window stops the server.
//
// Built only with -DZONOS2_APP=ON (needs a C++23 toolchain; see README).
#include "server-embed.h"
#include "app-pages.h"
#include "exe-path.h"

#include "httplib.h"
#include "json.hpp"

#include <saucer/smartview.hpp>
#include <saucer/modules/desktop.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// --------------------------------------------------------------------------- paths

static fs::path exe_dir() { return zonos2_exe_dir(); }

static fs::path config_path() {
#ifdef _WIN32
    const char * base = getenv("APPDATA");
    if (base) return fs::path(base) / "zonos2" / "app.json";
#elif defined(__APPLE__)
    const char * home = getenv("HOME");
    if (home) return fs::path(home) / "Library" / "Application Support" / "zonos2" / "app.json";
#else
    const char * xdg = getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) return fs::path(xdg) / "zonos2" / "app.json";
    const char * home = getenv("HOME");
    if (home) return fs::path(home) / ".config" / "zonos2" / "app.json";
#endif
    return fs::path("zonos2-app.json");   // last resort: CWD
}

// The web UI file, resolved relative to the executable (a double-clicked app has an
// arbitrary CWD): dist layout (web/ next to the binary), macOS bundle Resources,
// repo layout when running from a build tree, then the server's CWD-relative default.
static std::string resolve_ui_path() {
    const fs::path dir = exe_dir();
    const fs::path cands[] = {
        dir / "web" / "tts_ui.html",
        dir / ".." / "Resources" / "tts_ui.html",
        dir / ".." / "web" / "tts_ui.html",
    };
    std::error_code ec;
    for (const auto & c : cands)
        if (fs::exists(c, ec)) return c.lexically_normal().string();
    return "web/tts_ui.html";
}

// --------------------------------------------------------------------------- config

static json load_config() {
    std::ifstream f(config_path());
    if (!f) return json::object();
    json j = json::parse(f, nullptr, false);
    return j.is_object() ? j : json::object();
}

static std::string save_config(const json & j) {   // returns error message or ""
    const fs::path p = config_path();
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p);
    if (!f) return "cannot write " + p.string();
    f << j.dump(2) << "\n";
    return f.good() ? "" : "cannot write " + p.string();
}

static bool config_launchable(const json & j) {
    std::error_code ec;
    return j.contains("model") && j.contains("dac") &&
           fs::exists(j.value("model", ""), ec) && fs::exists(j.value("dac", ""), ec);
}

// --------------------------------------------------------------------------- ffmpeg
// Voice cloning decodes the reference audio through ffmpeg; plain TTS never touches it.
// The packaged app can't assume ffmpeg is installed, so if none is found we fetch a
// static build into the app config dir (bin/) and point the embedded server at it via
// ZONOS2_FFMPEG. The fetch runs on a detached background thread so startup isn't blocked;
// cloning just fails gracefully (existing "ffmpeg failed" error) until it lands.

static void set_env(const char * k, const std::string & v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    setenv(k, v.c_str(), 1);
#endif
}

static std::string shq(const std::string & s) {   // quote a path for the system() shell
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) q += (c == '\'') ? "'\\''" : std::string(1, c);
    return q + "'";
#endif
}

static fs::path ffmpeg_bin_path() {
#ifdef _WIN32
    return config_path().parent_path() / "bin" / "ffmpeg.exe";
#else
    return config_path().parent_path() / "bin" / "ffmpeg";
#endif
}

static bool ffmpeg_on_path() {
    const char * p = getenv("PATH");
    if (!p) return false;
#ifdef _WIN32
    const char sep = ';'; const char * name = "ffmpeg.exe";
#else
    const char sep = ':'; const char * name = "ffmpeg";
#endif
    std::error_code ec;
    std::stringstream ss{std::string(p)};
    for (std::string dir; std::getline(ss, dir, sep); )
        if (!dir.empty() && fs::exists(fs::path(dir) / name, ec)) return true;
    return false;
}

// Pull the ffmpeg binary out of a BtbN archive (nested at ffmpeg-*/bin/ffmpeg[.exe]).
static bool extract_ffmpeg(const fs::path & archive, const fs::path & dst) {
    std::error_code ec;
    const fs::path tmp = dst.parent_path() / ".ffx";
    fs::remove_all(tmp, ec); fs::create_directories(tmp, ec);
    // tar handles .tar.xz (GNU tar) and .zip (bsdtar ships on Win10 1803+, like curl.exe)
    if (std::system(("tar -xf " + shq(archive.string()) + " -C " + shq(tmp.string())).c_str()) != 0) {
        fs::remove_all(tmp, ec); return false;
    }
    fs::path found;
    for (fs::recursive_directory_iterator it(tmp, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && it->path().filename() == dst.filename() &&
            it->path().parent_path().filename() == "bin") { found = it->path(); break; }
    bool ok = false;
    if (!found.empty()) {
        fs::rename(found, dst, ec);
        if (ec) { ec.clear(); fs::copy_file(found, dst, fs::copy_options::overwrite_existing, ec); }
        ok = !ec;
    }
    fs::remove_all(tmp, ec);
    return ok;
}

static bool download_ffmpeg(const fs::path & dst) {
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);

    std::string url; bool single;
    if (const char * o = getenv("ZONOS2_FFMPEG_URL"); o && *o) { url = o; single = true; }   // test override
#if defined(__APPLE__)
    else { url = "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64"; single = true; }
#elif defined(_WIN32)
    else { url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-lgpl-7.1.zip"; single = false; }
#else
    else { url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-7.1.tar.xz"; single = false; }
#endif

    if (single) {
        const fs::path part = fs::path(dst.string() + ".part");
        if (std::system(("curl -L --fail --retry 3 -o " + shq(part.string()) + " " + shq(url)).c_str()) != 0) return false;
        fs::rename(part, dst, ec);
        if (ec) return false;
    } else {
        const fs::path arc = dst.parent_path() / "ffmpeg-dl.archive";
        if (std::system(("curl -L --fail --retry 3 -o " + shq(arc.string()) + " " + shq(url)).c_str()) != 0) return false;
        const bool ok = extract_ffmpeg(arc, dst);
        fs::remove(arc, ec);
        if (!ok) return false;
    }
#ifndef _WIN32
    fs::permissions(dst, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                         fs::perms::others_read | fs::perms::others_exec, ec);
#endif
    return fs::exists(dst, ec);
}

// Ensure ffmpeg is available and, when we own it, point the server at it via ZONOS2_FFMPEG.
// Call once, before the embedded server starts (sets the env single-threaded); the actual
// fetch is detached.
static void ensure_ffmpeg() {
    if (const char * e = getenv("ZONOS2_FFMPEG"); e && *e) return;   // user/CLI override wins
    const fs::path bin = ffmpeg_bin_path();
    std::error_code ec;
    if (fs::exists(bin, ec)) { set_env("ZONOS2_FFMPEG", bin.string()); return; }
    if (getenv("ZONOS2_SKIP_PATH_FFMPEG") == nullptr && ffmpeg_on_path()) return;  // use system ffmpeg
    // none present: reserve the path now (env set before any server thread), fetch in background
    set_env("ZONOS2_FFMPEG", bin.string());
    std::thread([bin] {
        if (!download_ffmpeg(bin))
            fprintf(stderr, "zonos2-app: ffmpeg download failed — voice cloning disabled until it is available\n");
        else
            fprintf(stderr, "zonos2-app: ffmpeg ready at %s\n", bin.string().c_str());
    }).detach();
}

// --------------------------------------------------------------------------- state

struct app_state {
    zonos2_server_ctl ctl;
    std::thread server_th, watcher_th;
    std::vector<std::string> args;      // owns the server's argv strings
    std::vector<char *>      cargs;
    std::atomic<bool> server_done{false};
    std::atomic<int>  server_rc{0};
    std::atomic<bool> shutdown{false};  // app closing: watcher must bail out
};

static app_state g_state;
static std::vector<std::string> g_passthrough;  // server args given on our command line
static bool g_force_setup = false;

// --------------------------------------------------------------------------- pages

static std::string page(const std::string & body) {
    return std::string("<!DOCTYPE html><html><head><meta charset=\"utf-8\">") +
           ZONOS2_APP_STYLE + "</head><body>" + body + "</body></html>";
}

static std::string html_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) switch (c) {
        case '&': out += "&amp;";  break;
        case '<': out += "&lt;";   break;
        case '>': out += "&gt;";   break;
        default:  out += c;
    }
    return out;
}

static std::string error_page(const std::string & details) {
    std::string body = ZONOS2_APP_ERROR_PAGE;
    const std::string ph = "%DETAILS%";
    if (auto pos = body.find(ph); pos != std::string::npos)
        body.replace(pos, ph.size(), html_escape(details));
    return page(body);
}

// --------------------------------------------------------------------------- server lifecycle

static void stop_server() {
    if (auto * svr = g_state.ctl.svr.load()) svr->stop();
    if (g_state.server_th.joinable())  g_state.server_th.join();
    if (g_state.watcher_th.joinable()) g_state.watcher_th.join();
}

// Waits for the server thread to bind, then for /health, then swaps the webview to
// the served UI. Runs on its own thread; saucer's set_url/set_html marshal to the
// app thread internally. The smartview outlives these threads: they are joined in
// start()'s coroutine frame, which owns it.
static void watch_server(saucer::smartview * webview) {
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::minutes(10);   // model load can be slow on cold disks

    auto fail = [&](const std::string & why) {
        webview->set_html(error_page(
            why + "\n\nConfig: " + config_path().string() +
            "\nTip: run zonos2-app from a terminal to see the full server log."));
    };

    int port = 0;
    while (!g_state.shutdown.load()) {
        port = g_state.ctl.port.load();
        if (port > 0) break;
        if (g_state.ctl.failed.load() || g_state.server_done.load())
            return fail("server exited during startup (code " + std::to_string(g_state.server_rc.load()) + ")");
        if (clock::now() > deadline) return fail("timed out waiting for the server to start");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(2);
    while (!g_state.shutdown.load()) {
        if (auto res = cli.Get("/health"); res && res->status == 200) {
            webview->set_url("http://127.0.0.1:" + std::to_string(port) + "/");
            return;
        }
        if (g_state.ctl.failed.load() || g_state.server_done.load())
            return fail("server exited during startup (code " + std::to_string(g_state.server_rc.load()) + ")");
        if (clock::now() > deadline) return fail("timed out waiting for the server to become healthy");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

static void start_server(saucer::smartview * webview, std::vector<std::string> args) {
    stop_server();   // joins any previous (failed) attempt
    g_state.ctl.port.store(0);
    g_state.ctl.failed.store(false);
    g_state.server_done.store(false);

    g_state.args = std::move(args);
    g_state.cargs.clear();
    for (auto & a : g_state.args) g_state.cargs.push_back(a.data());

    g_state.server_th = std::thread([] {
        const int rc = zonos2_server_main((int) g_state.cargs.size(), g_state.cargs.data(), &g_state.ctl);
        g_state.server_rc.store(rc);
        g_state.server_done.store(true);
    });
    g_state.watcher_th = std::thread([webview] { watch_server(webview); });
}

static std::vector<std::string> args_from_config(const json & cfg) {
    std::vector<std::string> a = {"zonos2-app", cfg.value("model", ""), "--dac", cfg.value("dac", "")};
    if (const std::string spk = cfg.value("spk", ""); !spk.empty()) { a.push_back("--spk"); a.push_back(spk); }
    a.push_back(cfg.value("gpu", true) ? "--gpu" : "--cpu");
    a.insert(a.end(), {"--host", "127.0.0.1", "--port", "0", "--ui", resolve_ui_path()});
    for (const auto & e : cfg.value("extra_args", std::vector<std::string>{})) a.push_back(e);
    return a;
}

// --------------------------------------------------------------------------- saucer app

coco::stray start(saucer::application * app) {
    auto window   = saucer::window::create(app).value();
    auto webview_ = saucer::smartview::create({.window = window});
    if (!webview_) {
        fprintf(stderr, "zonos2-app: failed to create the webview\n");
        app->quit();
        co_return;
    }
    saucer::smartview * webview = &*webview_;
    auto desktop = saucer::modules::desktop{app};

    window->set_title("Zonos2");
    window->set_size({.w = 1100, .h = 800});
    {
        // Window icon where the platform supports it (no-op on WebKitGTK; the
        // Windows taskbar/exe icon comes from the .rc resource, macOS from the bundle).
        std::error_code ec;
        for (const fs::path c : {exe_dir() / "zonos2.png", exe_dir() / ".." / "Resources" / "zonos2.png"})
            if (fs::exists(c, ec)) {
                if (auto ic = saucer::icon::from(c)) window->set_icon(*ic);
                break;
            }
    }

    webview->expose("pick_file", [&desktop](const std::string & /*kind*/) -> std::string {
        auto r = desktop.pick<saucer::modules::picker::type::file>({.filters = {"*.gguf"}});
        return r ? r->string() : std::string{};
    });

    webview->expose("get_config", []() -> std::string { return load_config().dump(); });

    webview->expose("launch", [webview](const std::string & model, const std::string & dac,
                                        const std::string & spk, bool gpu) -> std::string {
        std::error_code ec;
        if (!fs::exists(model, ec)) return "backbone model not found: " + model;
        if (!fs::exists(dac, ec))   return "DAC decoder not found: " + dac;
        if (!spk.empty() && !fs::exists(spk, ec)) return "speaker encoder not found: " + spk;

        json cfg = load_config();
        cfg["model"] = model; cfg["dac"] = dac; cfg["spk"] = spk; cfg["gpu"] = gpu;
        if (auto err = save_config(cfg); !err.empty())
            fprintf(stderr, "zonos2-app: warning: %s\n", err.c_str());

        webview->set_html(page(ZONOS2_APP_SPLASH_PAGE));
        start_server(webview, args_from_config(cfg));
        return "";
    });

    webview->expose("show_setup", [webview] { webview->set_html(page(ZONOS2_APP_SETUP_PAGE)); });

    if (!g_passthrough.empty()) {
        webview->set_html(page(ZONOS2_APP_SPLASH_PAGE));
        start_server(webview, g_passthrough);
    } else if (const json cfg = load_config(); !g_force_setup && config_launchable(cfg)) {
        webview->set_html(page(ZONOS2_APP_SPLASH_PAGE));
        start_server(webview, args_from_config(cfg));
    } else {
        webview->set_html(page(ZONOS2_APP_SETUP_PAGE));
    }

    window->show();
    co_await app->finish();

    g_state.shutdown.store(true);
    stop_server();
}

static void usage() {
    printf("usage: zonos2-app [--setup] | [<model.gguf> --dac <dac.gguf> [server options]]\n"
           "\n"
           "Native desktop app for the zonos2 TTS server.\n"
           "\n"
           "  (no arguments)   launch with the saved config; first run opens a setup page\n"
           "                   to pick the .gguf model files (config: %s)\n"
           "  --setup          reopen the setup page\n"
           "  --help           this help\n"
           "\n"
           "Any other arguments are passed through verbatim to the embedded zonos2-server\n"
           "(run `zonos2-server` for its options); the UI then attaches to whatever it binds.\n",
           config_path().string().c_str());
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }   // before any GUI init (headless-safe)
        if (a == "--setup") { g_force_setup = true; continue; }
        g_passthrough.push_back(a);
    }
    if (!g_passthrough.empty()) g_passthrough.insert(g_passthrough.begin(), "zonos2-app");

    ensure_ffmpeg();   // fetch ffmpeg for voice cloning in the background if none is present

    // Deliberately not forwarding argc/argv: GTK would try to parse the
    // server-passthrough options (--dac, ...) and error out on them.
    auto app = saucer::application::create({.id = "zonos2"});
    if (!app) {
        fprintf(stderr, "zonos2-app: failed to initialize the GUI: %s\n", app.error().message().c_str());
        return 1;
    }
    return app->run(start);
}
