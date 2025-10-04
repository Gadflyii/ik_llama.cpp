// Compatibility header for AMX code
// Provides minimal definitions without conflicting with ggml.c
#pragma once

#include "ggml.h"
#include <algorithm>
#include <memory>
#include <type_traits>

// Tile definitions (from upstream common.h)
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4
#define AMX_BLK_SIZE 32

// Tile register indices
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// Parallel utilities (from upstream common.h)
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

// Note: balance211 and parallel_for are in common.h - avoid duplication

// Minimal typedefs needed by AMX code
typedef uint16_t ggml_half;

// FP16 conversion (will use ggml-impl.h version if available)
#ifndef GGML_FP16_TO_FP32
#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#endif

#ifndef GGML_FP32_TO_FP16
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#endif

// CPU-specific FP16 conversion macros (for AMX mmq.cpp)
// Use the public API functions instead of internal lookup table versions
#ifndef GGML_CPU_FP16_TO_FP32
#define GGML_CPU_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#endif

#ifndef GGML_CPU_FP32_TO_FP16
#define GGML_CPU_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)
#endif

// Logging macros (simplified - AMX code uses these for debugging)
#ifndef GGML_LOG_DEBUG
#define GGML_LOG_DEBUG(...) do {} while(0)
#endif

// Memory alignment
#ifndef TENSOR_ALIGNMENT
#define TENSOR_ALIGNMENT 32
#endif

// Forward declare or define ggml_aligned_malloc (used by AMX buffer allocation)
#ifdef __cplusplus
extern "C" {
#endif

static inline void * ggml_aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, TENSOR_ALIGNMENT);
#else
    void * ptr = nullptr;
    int ret = posix_memalign(&ptr, TENSOR_ALIGNMENT, size);
    if (ret != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

// ggml_barrier is declared in ggml-cpu-impl.h and implemented in ggml.c
// (no stub needed here)

#ifdef __cplusplus
}
#endif
