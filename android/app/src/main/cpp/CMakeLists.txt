cmake_minimum_required(VERSION 3.22.1)
project("turboquant" LANGUAGES CXX)

# ── Compiler defaults ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Fetch llama.cpp ───────────────────────────────────────────────────────────
# Pins to a specific commit for reproducibility.
# Update the GIT_TAG when upgrading the inference engine.
include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master          # Pin to a specific tag / SHA for production
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)

# Disable everything we don't need to minimize build time
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
set(GGML_METAL             OFF CACHE BOOL "" FORCE)
set(GGML_CUDA              OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN            OFF CACHE BOOL "" FORCE)
set(GGML_SYCL              OFF CACHE BOOL "" FORCE)

# Enable ARM NEON where available
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8-a+simd" CACHE STRING "" FORCE)
endif()

FetchContent_MakeAvailable(llama_cpp)

# ── TurboQuant native library ─────────────────────────────────────────────────
add_library(turboquant SHARED
    turboquant_simd.cpp
    turboquant_bridge.cpp
)

# Headers from llama.cpp (paths vary slightly by version; cover all known layouts)
target_include_directories(turboquant PRIVATE
    ${llama_cpp_SOURCE_DIR}
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/common
)

# Link against llama, ggml, Android system libraries
target_link_libraries(turboquant
    PRIVATE
    llama
    ggml
    android
    log
    m
)

# ── Per-ABI optimisation flags ────────────────────────────────────────────────
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(turboquant PRIVATE
        -O3
        -march=armv8-a+simd
        -ffast-math
        -ftree-vectorize
        -fomit-frame-pointer
    )
    # Expose NEON to our SIMD code
    target_compile_definitions(turboquant PRIVATE __ARM_NEON=1)
endif()

# Strip debug symbols from release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(turboquant PRIVATE -s)
endif()
