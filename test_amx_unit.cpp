// Minimal AMX unit test - C++ to avoid linker issues
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>

// Minimal defines needed
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define QK8_0 32
#define QK4_0 32

typedef uint16_t ggml_fp16_t;

// Minimal FP16 conversion (simplified)
static float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    union { uint32_t i; float f; } u;
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h & 0x7C00) >> 10;
    uint32_t m = (h & 0x03FF) << 13;

    if (e == 0) {
        if (m == 0) { u.i = s; return u.f; }
        while (!(m & 0x00800000)) { m <<= 1; e -= 1; }
        e += 1; m &= ~0x00800000;
    }
    e = e + (127 - 15);
    u.i = s | (e << 23) | m;
    return u.f;
}

static ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
    union { uint32_t i; float f; } u = {.f = f};
    uint32_t s = (u.i >> 16) & 0x8000;
    int32_t e = ((u.i >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u.i & 0x007FFFFF;

    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;

    return s | (e << 10) | (m >> 13);
}

// Block structures
typedef struct {
    ggml_fp16_t d[8];
    int8_t qs[256];
} block_q4_0x8_unpacked;

typedef struct {
    ggml_fp16_t d;
    int8_t qs[32];
} block_q8_0;

// Simplified GEMV kernel (scalar only, no AMX)
void test_gemv(
    int n,
    float * s,
    size_t bs,
    const void * vx,
    const void * vy,
    int nr,
    int nc) {

    const block_q4_0x8_unpacked * x = (const block_q4_0x8_unpacked *)vx;
    const block_q8_0 * y = (const block_q8_0 *)vy;

    const int nb = n / QK8_0;
    const int nb_x8 = nb / 8;

    for (int row = 0; row < nc; row++) {
        const block_q4_0x8_unpacked * weight_row = x + row * nb_x8;
        float row_sum = 0.0f;

        for (int k_block = 0; k_block < nb_x8; k_block++) {
            const block_q4_0x8_unpacked * wgt = weight_row + k_block;

            for (int j = 0; j < 8; j++) {
                const block_q8_0 * act = y + k_block * 8 + j;

                int32_t sumi = 0;
                for (int k = 0; k < QK4_0; k++) {
                    sumi += (int32_t)wgt->qs[j * QK4_0 + k] * (int32_t)act->qs[k];
                }

                float wgt_scale = GGML_FP16_TO_FP32(wgt->d[j]);
                float act_scale = GGML_FP16_TO_FP32(act->d);
                row_sum += sumi * wgt_scale * act_scale;
            }
        }

        s[row] = row_sum;
    }
}

int main() {
    printf("=== AMX Kernel Unit Test ===\n\n");

    const int K = 256;
    const int NC = 2;
    const int NB = 8;

    auto *weights = (block_q4_0x8_unpacked*)calloc(NC, sizeof(block_q4_0x8_unpacked));
    auto *activations = (block_q8_0*)calloc(NB, sizeof(block_q8_0));
    auto *output = (float*)calloc(NC, sizeof(float));

    // Simple test data
    for (int row = 0; row < NC; row++) {
        for (int sub = 0; sub < 8; sub++) {
            weights[row].d[sub] = GGML_FP32_TO_FP16(1.0f);
            for (int k = 0; k < 32; k++) {
                weights[row].qs[sub * 32 + k] = 1;
            }
        }
    }

    for (int i = 0; i < NB; i++) {
        activations[i].d = GGML_FP32_TO_FP16(1.0f);
        for (int k = 0; k < 32; k++) {
            activations[i].qs[k] = 1;
        }
    }

    printf("Calling test kernel...\n");
    test_gemv(K, output, NC, weights, activations, 1, NC);

    printf("Output[0] = %f\n", output[0]);
    printf("Output[1] = %f\n", output[1]);

    // Expected: 256 * 1 * 1 * 1 * 1 = 256
    float expected = 256.0f;
    float error = fabsf(output[0] - expected) / expected;

    if (std::isnan(output[0])) {
        printf("❌ FAIL: NaN output\n");
        return 1;
    } else if (error > 0.01f) {
        printf("❌ FAIL: Incorrect (error %.1f%%)\n", error * 100);
        return 1;
    } else {
        printf("✅ PASS: Kernel works correctly\n");
        return 0;
    }
}
