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

# --- ggml (vendored submodule) ---
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

# --- speaker encoder (ECAPA-TDNN) CLI ---
add_executable(spk-encoder-cli src/spk-encoder.cpp)
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
