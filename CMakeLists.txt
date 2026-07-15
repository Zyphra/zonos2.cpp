cmake_minimum_required(VERSION 3.18)
project(zonos2-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- zonos2-app (desktop shell) opt-in. Declared here, before add_subdirectory(ggml),
#     specifically so the Apple Objective-C / -C++ languages it needs are enabled in
#     THIS root scope first. saucer's WKWebView backend + saucer-embed inject .m/.mm
#     sources into the top-level zonos2-app target; ggml's Metal backend also enables
#     OBJC — but inside ggml's own subdirectory scope. Whichever scope enables a
#     language *first* is the only one that gets CMAKE_<LANG>_COMPILE_OBJECT, so if
#     ggml (a subdir) wins the race the top-level target is left without the rule and
#     CMake dies at generate with "CMAKE_OBJC_COMPILE_OBJECT not set". Enabling both
#     here, pre-ggml, makes the root scope the first enabler and the rules propagate
#     down to ggml and up to zonos2-app alike. C++23 toolchain still required (README). ---
option(ZONOS2_APP "Build the zonos2-app desktop shell (requires a C++23 toolchain, see README)" OFF)
if (ZONOS2_APP AND APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

# --- CUDA target (H100 = sm_90). Enable with -DGGML_CUDA=ON ---
if (GGML_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)
    endif()
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE) # CUDA-graph replay for real-time decode
endif()

# --- ggml (upstream submodule, pinned to a pristine SHA) ---
# We carry local ggml fixes as patches under patches/ rather than forking the
# submodule. They are (re)applied idempotently at configure time: the reverse
# --check skips an already-patched tree, so re-running cmake is a no-op. If a
# future submodule bump makes a patch no longer apply, configure fails loudly.
find_package(Git REQUIRED)
file(GLOB GGML_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT GGML_PATCHES)
foreach(_patch ${GGML_PATCHES})
    get_filename_component(_name ${_patch} NAME)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply --reverse --check ${_patch}
        RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if (_applied EQUAL 0)
        message(STATUS "ggml patch already applied: ${_name}")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply ${_patch}
            RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if (_rc EQUAL 0)
            message(STATUS "ggml patch applied: ${_name}")
        else()
            message(FATAL_ERROR "Failed to apply ggml patch ${_name}:\n${_err}")
        endif()
    endif()
endforeach()

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

# --- zonos2 library (loader, graph, sampler) ---
add_library(zonos2 STATIC
    src/zonos2.cpp
    src/zonos2-graph.cpp
    src/zonos2-sampler.cpp
    src/zonos2-prompt.cpp
    src/zonos2-emotion.cpp
)
target_include_directories(zonos2 PUBLIC src ggml/include PRIVATE vendor)
target_link_libraries(zonos2 PUBLIC ggml)

# --- CLI (links the DAC decoder so --tts/--generate can emit a wav directly,
#     and the speaker encoder so --clone does one-command voice cloning) ---
add_executable(zonos2-cli src/main.cpp src/dac.cpp src/spk-encoder.cpp)
target_include_directories(zonos2-cli PRIVATE src ggml/include)
target_link_libraries(zonos2-cli PRIVATE zonos2 ggml)

# --- speaker encoder (ECAPA-TDNN): library (shared with the server) + CLI ---
add_executable(spk-encoder-cli src/spk-encoder-cli.cpp src/spk-encoder.cpp)
target_include_directories(spk-encoder-cli PRIVATE src ggml/include)
target_link_libraries(spk-encoder-cli PRIVATE ggml)

# --- DAC vocoder decoder CLI (codes -> wav) ---
add_executable(dac-cli src/dac-cli.cpp src/dac.cpp)
target_include_directories(dac-cli PRIVATE src ggml/include)
target_link_libraries(dac-cli PRIVATE ggml)

# --- gguf -> gguf requantizer (F16 backbone -> Q8_0 / K-quants) ---
add_executable(quantize-cli src/quantize.cpp)
target_include_directories(quantize-cli PRIVATE src ggml/include)
target_link_libraries(quantize-cli PRIVATE ggml)

# --- gguf -> gguf MoE expert pruner (drops low-MSAN experts per layer) ---
add_executable(prune-cli src/prune.cpp)
target_include_directories(prune-cli PRIVATE src ggml/include)
target_link_libraries(prune-cli PRIVATE ggml)

# --- perplexity / KL-divergence eval for quantized backbones (scoring only, no DAC) ---
add_executable(zonos2-perplexity src/perplexity.cpp)
target_include_directories(zonos2-perplexity PRIVATE src ggml/include)
target_link_libraries(zonos2-perplexity PRIVATE zonos2 ggml)

# --- HTTP TTS server (mirrors ../ZONOS2 FastAPI: /tts/generate streaming, /v1/audio/speech,
#     speaker cloning, web UI). Header-only cpp-httplib + nlohmann/json under vendor/. ---
add_executable(zonos2-server src/server.cpp src/dac.cpp src/spk-encoder.cpp)
target_include_directories(zonos2-server PRIVATE src vendor ggml/include)
find_package(Threads REQUIRED)
target_link_libraries(zonos2-server PRIVATE zonos2 ggml Threads::Threads)
if (WIN32)
    target_link_libraries(zonos2-server PRIVATE ws2_32)
endif()

# --- Desktop app: the server embedded in a native webview window (saucer). Opt-in
#     because saucer needs a much newer toolchain than the rest of the project:
#     CMake >= 3.25, C++23 (GCC >= 14 / Clang >= 20 / AppleClang >= 17 / MSVC >= 19.44),
#     and on Linux GTK4 >= 4.12 + WebKitGTK-6.0 + libadwaita + json-glib dev packages.
#     Everything else in this file keeps building with the C++17 / CMake 3.18 floor. ---
# ZONOS2_APP is declared (and its Apple languages enabled) near the top of this file,
# before add_subdirectory(ggml); see the comment there for why the ordering matters.
if (ZONOS2_APP)
    if (CMAKE_VERSION VERSION_LESS 3.31)
        # saucer core says 3.25, but its saucer-embed dependency uses policy CMP0174 (3.31).
        message(FATAL_ERROR "ZONOS2_APP requires CMake >= 3.31 (saucer); this is CMake ${CMAKE_VERSION}")
    endif()

    include(FetchContent)
    FetchContent_Declare(saucer
        GIT_REPOSITORY https://github.com/saucer/saucer
        GIT_TAG        v8.0.5)
    FetchContent_Declare(saucer-desktop      # native file pickers (saucer module)
        GIT_REPOSITORY https://github.com/saucer/desktop
        GIT_TAG        v4.1.0)
    FetchContent_MakeAvailable(saucer saucer-desktop)

    add_executable(zonos2-app src/app.cpp src/server.cpp src/dac.cpp src/spk-encoder.cpp)
    target_compile_definitions(zonos2-app PRIVATE ZONOS2_APP_EMBED)   # server.cpp: no main()
    target_compile_features(zonos2-app PRIVATE cxx_std_23)
    target_include_directories(zonos2-app PRIVATE src vendor ggml/include)
    target_link_libraries(zonos2-app PRIVATE zonos2 ggml saucer::saucer saucer::desktop Threads::Threads)

    if (WIN32)
        # GUI subsystem (no console window) with the icon resource; saucer has no
        # WinMain, so keep the standard main() as the entry point.
        target_sources(zonos2-app PRIVATE assets/app/zonos2.rc)
        set_target_properties(zonos2-app PROPERTIES WIN32_EXECUTABLE TRUE)
        target_link_libraries(zonos2-app PRIVATE ws2_32)
        if (MSVC)
            target_link_options(zonos2-app PRIVATE "/ENTRY:mainCRTStartup")
        endif()
    elseif (APPLE)
        # Zonos2.app bundle; the web UI and icon ship inside Contents/Resources
        # (app.cpp resolves ../Resources/tts_ui.html relative to the executable).
        set(ZONOS2_APP_RESOURCES web/tts_ui.html assets/app/zonos2.icns assets/app/zonos2.png)
        target_sources(zonos2-app PRIVATE ${ZONOS2_APP_RESOURCES})
        set_source_files_properties(${ZONOS2_APP_RESOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
        set_target_properties(zonos2-app PROPERTIES
            OUTPUT_NAME Zonos2
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/assets/app/Info.plist.in
            MACOSX_BUNDLE_BUNDLE_NAME Zonos2
            MACOSX_BUNDLE_GUI_IDENTIFIER cpp.zonos2.app
            MACOSX_BUNDLE_BUNDLE_VERSION 1.0
            MACOSX_BUNDLE_SHORT_VERSION_STRING 1.0
            MACOSX_BUNDLE_ICON_FILE zonos2.icns)
    endif()
endif()
