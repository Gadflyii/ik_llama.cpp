// SPDX-License-Identifier: MIT
// Intel AMX (Advanced Matrix Extensions) Support
// Supports both AMX-INT8 (quantized types) and AMX-BF16 (floating point types)

#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Runtime Control (works regardless of compile-time AMX support)
// =============================================================================

// Enable/disable AMX at runtime (opt-in with --amx flag)
void ggml_amx_set_enabled(bool enabled);

// Check if AMX is currently enabled at runtime
bool ggml_amx_is_enabled(void);

// =============================================================================
// AMX-INT8 Support (for quantized types)
// Available on: Intel Xeon 4th Gen (Sapphire Rapids) and newer
// =============================================================================

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX-INT8 initialization
bool ggml_amx_int8_init(void);

// Check if AMX-INT8 is available
bool ggml_amx_int8_available(void);

// Initialize AMX tile configuration
void ggml_amx_tile_config_init(void);

// Check if a quantization type has AMX support
bool ggml_amx_can_handle(enum ggml_type type);

// Get the size of packed buffer needed for AMX-optimized weights
// type: quantization type
// n: number of TILE_N blocks (typically K * N / (TILE_K * TILE_N))
size_t ggml_amx_get_packed_size(enum ggml_type type, int n);

// Pack weights into AMX-optimized format
// type: quantization type
// weights: source weights in standard format
// packed_buffer: destination buffer (must be pre-allocated)
// K, N: matrix dimensions (must be aligned to tile sizes)
// Returns true on success, false if not supported or alignment issues
bool ggml_amx_pack_weights(enum ggml_type type, const void * weights,
                            void * packed_buffer, int64_t K, int64_t N);

// Test function to verify AMX tile operations work
void ggml_amx_test_tiles(void);

// Simple GEMV using AMX (proof of concept)
// Declared with AMX-INT8 section
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
void ggml_amx_gemv_q4_0_q8_0_simple(int K, int N, const void * x, const void * y, float * dst);
#endif

// Quantization functions with AMX-optimized layout
// These functions prepare data in AMX-friendly formats
size_t quantize_iq2_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_iq3_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_iq4_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_q4_0_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                               int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_q8_0_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                               int64_t nrows, int64_t n_per_row, const float * imatrix);

// Trellis quantization types (IKQ fork-specific)
size_t quantize_iq2_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_iq3_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_iq4_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix);

// GEMV (matrix-vector multiplication) operations
void ggml_gemv_iq2_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemv_iq3_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemv_iq4_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemv_q4_0_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                   const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                   int nr, int nc);

// GEMM (matrix-matrix multiplication) operations
void ggml_gemm_iq2_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemm_iq3_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemm_iq4_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc);
void ggml_gemm_q4_0_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                   const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                   int nr, int nc);

#endif // __AMX_INT8__ && __AVX512VNNI__

// =============================================================================
// AMX-BF16 Support (for floating point types)
// Available on: Intel Xeon 4th Gen (Sapphire Rapids) and newer
// =============================================================================

#if defined(__AMX_BF16__)

// AMX-BF16 initialization
bool ggml_amx_bf16_init(void);

// Check if AMX-BF16 is available
bool ggml_amx_bf16_available(void);

// GEMV operations for FP16 input -> FP32 output
void ggml_gemv_f16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                 int nr, int nc);

// GEMM operations for FP16 input -> FP32 output
void ggml_gemm_f16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                 int nr, int nc);

// GEMV operations for BF16 input -> FP32 output
void ggml_gemv_bf16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                  int nr, int nc);

// GEMM operations for BF16 input -> FP32 output
void ggml_gemm_bf16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                  int nr, int nc);

#endif // __AMX_BF16__

#ifdef __cplusplus
}
#endif
