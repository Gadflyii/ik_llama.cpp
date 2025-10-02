// AMX Implementation for ik_llama.cpp
// Ported from upstream llama.cpp AMX implementation
// Supports: Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS

#include "ggml-amx.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include <string.h>
#include <assert.h>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

#include <immintrin.h>
#include <stdint.h>

// AMX Configuration
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

// Tile register assignments
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tile_config_t;

// Macro for tile configuration
#define TC_CONFIG_TILE(i, r, cb) tc.rows[i] = r; tc.colsb[i] = cb

// Initialize AMX tile configuration
void ggml_amx_tile_config_init(void) {
    static _Thread_local bool is_first_time = true;

    if (!is_first_time) {
        return;
    }

    static _Thread_local tile_config_t tc;
    tile_config_t current_tc;
    _tile_storeconfig(&current_tc);

    // Load only when config changes
    if (tc.palette_id == 0 ||
        (memcmp(&current_tc.colsb, &tc.colsb, sizeof(uint16_t) * 8) != 0 &&
         memcmp(&current_tc.rows, &tc.rows, sizeof(uint8_t) * 8) != 0)) {

        tc.palette_id = 1;
        tc.start_row = 0;
        TC_CONFIG_TILE(TMM0, 8, 64);
        TC_CONFIG_TILE(TMM1, 8, 64);
        TC_CONFIG_TILE(TMM2, 16, 32);
        TC_CONFIG_TILE(TMM3, 16, 32);
        TC_CONFIG_TILE(TMM4, 16, 64);
        TC_CONFIG_TILE(TMM5, 16, 64);
        TC_CONFIG_TILE(TMM6, 16, 64);
        TC_CONFIG_TILE(TMM7, 16, 64);
        _tile_loadconfig(&tc);
    }

    is_first_time = false;
}

// Helper macros for AVX2/AVX512
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// Helper: Unpack 4-bit nibbles to bytes (for Q4_0, Q4_1)
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// Helper: Pack two sets of nibbles back together
static inline __m512i packNibbles(__m512i r0, __m512i r1) {
    return _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

// Helper: Transpose 8x8 32-bit elements
#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))

static inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    // unpacking 32-bit elements
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    // shuffling the 32-bit elements
    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    // shuffling 128-bit elements
    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

//==============================================================================
// Weight Repacking Functions (pack_B)
//==============================================================================

// Pack Q4_0 weights for AMX
// Layout: quants {TILE_N, TILE_K/2} int8, d {TILE_N} ggml_half
static void pack_B_q4_0(void * RESTRICT packed_B, const block_q4_0 * RESTRICT B, int KB) {
    // Pack quantized values in transposed VNNI format
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];

    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }

    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // Pack nibbles again to fully utilize vector length
    for (int n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
    }

    // Pack scale factors (d)
    ggml_half * d0 = (ggml_half *)((char *)packed_B + TILE_N * TILE_K / 2);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

// Pack Q4_1 weights for AMX
// Layout: quants {TILE_N, TILE_K/2} int8, d {TILE_N} ggml_half, m {TILE_N} ggml_half
static void pack_B_q4_1(void * RESTRICT packed_B, const block_q4_1 * RESTRICT B, int KB) {
    // Pack quantized values (same as Q4_0)
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];

    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }

    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    for (int n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
    }

    // Pack scale factors (d) and mins (m)
    ggml_half * d0 = (ggml_half *)((char *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

// S8S8 compensation for Q8_0 (signed x signed requires compensation)
static inline void s8s8_compensation(void * RESTRICT packed_B) {
    const int offset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    const __m512i off = _mm512_set1_epi8((char)0x80);

    for (int k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((char *)(packed_B) + offset), vcomp);
}

// Pack Q8_0 weights for AMX
// Layout: quants {TILE_N, TILE_K} int8, d {TILE_N} ggml_half, comp {TILE_N} int32
static void pack_B_q8_0(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    __m256i v[8], v2[8];

    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64), v2[n]);
    }

    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64 + 32), v2[n]);
    }

    // Pack scale factors
    ggml_half * d0 = (ggml_half *)((char *)packed_B + TILE_N * TILE_K);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }

    // Calculate and store compensation values
    s8s8_compensation(packed_B);
}

// TODO: Implement pack_B for Q4_K, Q5_K, Q6_K, IQ4_XS
// These are more complex and will be added in subsequent iterations

//==============================================================================
// Public API: Get packed buffer size
//==============================================================================

size_t ggml_amx_get_packed_size(enum ggml_type type, int n) {
    // n is the number of TILE_N blocks
    switch (type) {
        case GGML_TYPE_Q4_0:
            // quants + d
            return n * (TILE_N * TILE_K / 2 + TILE_N * sizeof(ggml_half));
        case GGML_TYPE_Q4_1:
            // quants + d + m
            return n * (TILE_N * TILE_K / 2 + 2 * TILE_N * sizeof(ggml_half));
        case GGML_TYPE_Q8_0:
            // quants + d + compensation
            return n * (TILE_N * TILE_K + TILE_N * sizeof(ggml_half) + TILE_N * sizeof(int32_t));
        default:
            return 0;
    }
}

//==============================================================================
// Public API: Pack weights
//==============================================================================

bool ggml_amx_pack_weights(enum ggml_type type, const void * weights, void * packed_buffer,
                            int64_t K, int64_t N) {
    if (!ggml_amx_is_enabled()) {
        return false;
    }

    // K and N must be aligned to tile dimensions
    if (K % TILE_K != 0 || N % TILE_N != 0) {
        return false;
    }

    const int KB = K / TILE_K;  // Number of K blocks
    const int NB = N / TILE_N;  // Number of N blocks

    // Pack each N tile
    for (int nb = 0; nb < NB; ++nb) {
        size_t tile_size = ggml_amx_get_packed_size(type, 1);
        void * packed_tile = (char *)packed_buffer + nb * tile_size;

        switch (type) {
            case GGML_TYPE_Q4_0: {
                const block_q4_0 * w = (const block_q4_0 *)weights;
                pack_B_q4_0(packed_tile, w + nb * TILE_N * KB, KB);
                break;
            }
            case GGML_TYPE_Q4_1: {
                const block_q4_1 * w = (const block_q4_1 *)weights;
                pack_B_q4_1(packed_tile, w + nb * TILE_N * KB, KB);
                break;
            }
            case GGML_TYPE_Q8_0: {
                const block_q8_0 * w = (const block_q8_0 *)weights;
                pack_B_q8_0(packed_tile, w + nb * TILE_N * KB, KB);
                break;
            }
            default:
                return false;
        }
    }

    return true;
}

//==============================================================================
// Public API: Check if quantization type has AMX support
//==============================================================================

bool ggml_amx_can_handle(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        // TODO: Add Q4_K, Q5_K, Q6_K, IQ4_XS when implemented
            return true;
        default:
            return false;
    }
}

#endif // __AMX_INT8__ && __AVX512VNNI__
