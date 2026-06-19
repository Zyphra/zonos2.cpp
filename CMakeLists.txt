cmake_minimum_required(VERSION 3.18)
project(zonos2-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
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
)
target_include_directories(zonos2 PUBLIC src ggml/include)
target_link_libraries(zonos2 PUBLIC ggml)

# --- CLI (links the DAC decoder so --tts/--generate can emit a wav directly) ---
add_executable(zonos2-cli src/main.cpp src/dac.cpp)
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
