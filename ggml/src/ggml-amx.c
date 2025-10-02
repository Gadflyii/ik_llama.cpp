// SPDX-License-Identifier: MIT
// Intel AMX (Advanced Matrix Extensions) Implementation
// Supports both AMX-INT8 (quantized types) and AMX-BF16 (floating point types)

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-amx.h"

#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#endif

// AMX tile dimensions
#define TILE_M 16
#define TILE_N 16
#define TILE_K_INT8 64  // For INT8: 64 elements per tile row
#define TILE_K_BF16 32  // For BF16: 32 elements per tile row

// Tile register indices
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// =============================================================================
// Common AMX Utilities
// =============================================================================

// Global runtime enable/disable flag (default: disabled, opt-in with --amx)
static bool g_amx_enabled = false;

// Runtime control functions
void ggml_amx_set_enabled(bool enabled) {
    g_amx_enabled = enabled;
    if (enabled) {
        fprintf(stderr, "[AMX] Runtime AMX acceleration ENABLED\n");
    } else {
        fprintf(stderr, "[AMX] Runtime AMX acceleration DISABLED\n");
    }
}

bool ggml_amx_is_enabled(void) {
    return g_amx_enabled;
}

// Tile configuration structure
typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];  // Column bytes for each tile
    uint8_t rows[16];    // Rows for each tile
} tile_config_t;

// =============================================================================
// AMX-INT8 Implementation (Quantized Types)
// =============================================================================

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

static bool g_amx_int8_initialized = false;

bool ggml_amx_int8_init(void) {
    if (g_amx_int8_initialized) {
        return true;
    }

#if defined(__linux__)
    // Request AMX permission via syscall
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "[AMX-INT8] Initialization failed - syscall error\n");
        fprintf(stderr, "[AMX-INT8] This may require kernel 5.16+ and AMX-capable CPU\n");
        return false;
    }
    fprintf(stderr, "[AMX-INT8] Successfully initialized\n");
    g_amx_int8_initialized = true;

    // Run a quick test to verify AMX tiles are working
    fprintf(stderr, "[AMX-INT8] Running tile operation test...\n");
    ggml_amx_test_tiles();

    return true;
#elif defined(_WIN32)
    // Windows AMX support - typically enabled by default on supported CPUs
    fprintf(stderr, "[AMX-INT8] Windows AMX support enabled\n");
    g_amx_int8_initialized = true;
    return true;
#else
    fprintf(stderr, "[AMX-INT8] Unsupported platform\n");
    return false;
#endif
}

bool ggml_amx_int8_available(void) {
    return g_amx_int8_initialized;
}

// Configure AMX tiles for INT8 operations
static inline void configure_tiles_int8(int rows_a, int cols_a, int rows_b, int cols_b) {
    tile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.palette_id = 1;  // AMX palette 1 for INT8

    // Configure tiles for matrix multiplication: C = A * B
    // Tile 0: Matrix A (M x K)
    cfg.rows[0] = rows_a;
    cfg.colsb[0] = cols_a;  // Bytes per row

    // Tile 1: Matrix B (K x N)
    cfg.rows[1] = rows_b;
    cfg.colsb[1] = cols_b;

    // Tile 2: Accumulator C (M x N) - INT32
    cfg.rows[2] = rows_a;
    cfg.colsb[2] = rows_a * 4;  // 4 bytes per INT32 element

    _tile_loadconfig(&cfg);
}

// =============================================================================
// AMX-INT8 Quantization Functions
// =============================================================================

// IQ2_K quantization with AMX-INT8 optimization
size_t quantize_iq2_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ2_K quantization
    // For now, fall back to standard quantization
    // This will be implemented by adapting upstream AMX code

    fprintf(stderr, "[AMX-INT8] IQ2_K quantization not yet implemented, using fallback\n");
    return 0;  // Return bytes written
}

size_t quantize_iq3_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ3_K quantization
    fprintf(stderr, "[AMX-INT8] IQ3_K quantization not yet implemented, using fallback\n");
    return 0;
}

size_t quantize_iq4_k_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ4_K quantization
    fprintf(stderr, "[AMX-INT8] IQ4_K quantization not yet implemented, using fallback\n");
    return 0;
}

size_t quantize_q4_0_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                               int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized Q4_0 quantization
    fprintf(stderr, "[AMX-INT8] Q4_0 quantization not yet implemented, using fallback\n");
    return 0;
}

size_t quantize_q8_0_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                               int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized Q8_0 quantization
    fprintf(stderr, "[AMX-INT8] Q8_0 quantization not yet implemented, using fallback\n");
    return 0;
}

// Trellis quantization types
size_t quantize_iq2_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ2_KT quantization
    fprintf(stderr, "[AMX-INT8] IQ2_KT quantization not yet implemented, using fallback\n");
    return 0;
}

size_t quantize_iq3_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ3_KT quantization
    fprintf(stderr, "[AMX-INT8] IQ3_KT quantization not yet implemented, using fallback\n");
    return 0;
}

size_t quantize_iq4_kt_amx_int8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                 int64_t nrows, int64_t n_per_row, const float * imatrix) {
    // TODO: Implement AMX-optimized IQ4_KT quantization
    fprintf(stderr, "[AMX-INT8] IQ4_KT quantization not yet implemented, using fallback\n");
    return 0;
}

// =============================================================================
// AMX-INT8 GEMV Functions (Matrix-Vector Multiplication)
// =============================================================================

void ggml_gemv_iq2_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMV for IQ2_K
    // This will use _tile_loadd, _tile_dpbssd, _tile_stored
    fprintf(stderr, "[AMX-INT8] IQ2_K GEMV not yet implemented\n");
}

void ggml_gemv_iq3_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMV for IQ3_K
    fprintf(stderr, "[AMX-INT8] IQ3_K GEMV not yet implemented\n");
}

void ggml_gemv_iq4_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMV for IQ4_K
    fprintf(stderr, "[AMX-INT8] IQ4_K GEMV not yet implemented\n");
}

void ggml_gemv_q4_0_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                   const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                   int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMV for Q4_0
    fprintf(stderr, "[AMX-INT8] Q4_0 GEMV not yet implemented\n");
}

// =============================================================================
// AMX-INT8 GEMM Functions (Matrix-Matrix Multiplication)
// =============================================================================

void ggml_gemm_iq2_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMM for IQ2_K
    fprintf(stderr, "[AMX-INT8] IQ2_K GEMM not yet implemented\n");
}

void ggml_gemm_iq3_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMM for IQ3_K
    fprintf(stderr, "[AMX-INT8] IQ3_K GEMM not yet implemented\n");
}

void ggml_gemm_iq4_k_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                    int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMM for IQ4_K
    fprintf(stderr, "[AMX-INT8] IQ4_K GEMM not yet implemented\n");
}

void ggml_gemm_q4_0_q8_0_amx_int8(int n, float * GGML_RESTRICT s, size_t bs,
                                   const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                   int nr, int nc) {
    // TODO: Implement AMX-INT8 GEMM for Q4_0
    fprintf(stderr, "[AMX-INT8] Q4_0 GEMM not yet implemented\n");
}

#endif // __AMX_INT8__ && __AVX512VNNI__

// =============================================================================
// AMX-BF16 Implementation (Floating Point Types)
// =============================================================================

#if defined(__AMX_BF16__)

static bool g_amx_bf16_initialized = false;

bool ggml_amx_bf16_init(void) {
    if (g_amx_bf16_initialized) {
        return true;
    }

#if defined(__linux__)
    // Request AMX permission via syscall
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "[AMX-BF16] Initialization failed - syscall error\n");
        fprintf(stderr, "[AMX-BF16] This may require kernel 5.16+ and AMX-capable CPU\n");
        return false;
    }
    fprintf(stderr, "[AMX-BF16] Successfully initialized\n");
    g_amx_bf16_initialized = true;
    return true;
#elif defined(_WIN32)
    // Windows AMX support
    fprintf(stderr, "[AMX-BF16] Windows AMX support enabled\n");
    g_amx_bf16_initialized = true;
    return true;
#else
    fprintf(stderr, "[AMX-BF16] Unsupported platform\n");
    return false;
#endif
}

bool ggml_amx_bf16_available(void) {
    return g_amx_bf16_initialized;
}

// Configure AMX tiles for BF16 operations
static inline void configure_tiles_bf16(int rows_a, int cols_a_bf16, int rows_b, int cols_b_bf16) {
    tile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.palette_id = 1;  // AMX palette 1

    // Configure tiles for BF16 matrix multiplication
    // Tile 0: Matrix A (M x K) in BF16
    cfg.rows[0] = rows_a;
    cfg.colsb[0] = cols_a_bf16 * 2;  // 2 bytes per BF16 element

    // Tile 1: Matrix B (K x N) in BF16
    cfg.rows[1] = rows_b;
    cfg.colsb[1] = cols_b_bf16 * 2;

    // Tile 2: Accumulator C (M x N) - FP32
    cfg.rows[2] = rows_a;
    cfg.colsb[2] = rows_a * 4;  // 4 bytes per FP32 element

    _tile_loadconfig(&cfg);
}

// =============================================================================
// AMX-BF16 GEMV Functions (F16 input)
// =============================================================================

void ggml_gemv_f16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                 int nr, int nc) {
    // TODO: Implement AMX-BF16 GEMV for F16
    // This will use _tile_loadd, _tile_dpbf16ps, _tile_stored
    fprintf(stderr, "[AMX-BF16] F16 GEMV not yet implemented\n");
}

// =============================================================================
// AMX-BF16 GEMM Functions (F16 input)
// =============================================================================

void ggml_gemm_f16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                 int nr, int nc) {
    // TODO: Implement AMX-BF16 GEMM for F16
    fprintf(stderr, "[AMX-BF16] F16 GEMM not yet implemented\n");
}

// =============================================================================
// AMX-BF16 GEMV Functions (BF16 input)
// =============================================================================

void ggml_gemv_bf16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                  int nr, int nc) {
    // TODO: Implement AMX-BF16 GEMV for BF16
    fprintf(stderr, "[AMX-BF16] BF16 GEMV not yet implemented\n");
}

// =============================================================================
// AMX-BF16 GEMM Functions (BF16 input)
// =============================================================================

void ggml_gemm_bf16_f32_amx_bf16(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                  int nr, int nc) {
    // TODO: Implement AMX-BF16 GEMM for BF16
    fprintf(stderr, "[AMX-BF16] BF16 GEMM not yet implemented\n");
}

#endif // __AMX_BF16__
