// Simple standalone test for AMX kernels
// Tests with known-good synthetic data to verify kernel correctness
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml/src/ggml-amx-repack.h"
#include "ggml/src/ggml-quants.h"

#define TEST_K 256      // K dimension (8 Q8_0 blocks)
#define TEST_NC 2       // 2 output rows
#define TEST_NB 8       // 8 Q8_0 blocks

int main() {
    printf("=== AMX Kernel Standalone Test ===\n\n");

    // Allocate data
    block_q4_0x8_unpacked *weights = calloc(TEST_NC, sizeof(block_q4_0x8_unpacked));
    block_q8_0 *activations = calloc(TEST_NB, sizeof(block_q8_0));
    float *output = calloc(TEST_NC, sizeof(float));

    if (!weights || !activations || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    printf("Creating synthetic test data...\n");

    // Initialize weights: simple pattern for easy verification
    // Weight row 0: all 1s with scale 0.1
    // Weight row 1: all 2s with scale 0.2
    for (int row = 0; row < TEST_NC; row++) {
        for (int sub = 0; sub < 8; sub++) {
            weights[row].d[sub] = GGML_FP32_TO_FP16(0.1f * (row + 1));
            for (int k = 0; k < 32; k++) {
                weights[row].qs[sub * 32 + k] = (row + 1);  // 1 or 2
            }
        }
    }

    // Initialize activations using standard Q8_0 quantization
    // Create F32 input: [3, 3, 3, ...]
    float *f32_input = malloc(TEST_K * sizeof(float));
    for (int i = 0; i < TEST_K; i++) {
        f32_input[i] = 3.0f;
    }

    // Quantize to Q8_0 using GGML's standard function
    quantize_row_q8_0_ref(f32_input, activations, TEST_K);
    free(f32_input);

    // Verify activations were quantized correctly
    printf("Activation block 0: d=%f, qs[0]=%d\n",
           GGML_FP16_TO_FP32(activations[0].d),
           activations[0].qs[0]);

    // Expected results:
    // Row 0: sum(1 * 3) * (0.1 * scale_act) over 256 elements = 768 * 0.1 * scale_act
    // Row 1: sum(2 * 3) * (0.2 * scale_act) over 256 elements = 1536 * 0.2 * scale_act
    float expected_scale_act = GGML_FP16_TO_FP32(activations[0].d);
    float expected0 = 256.0f * 1.0f * 3.0f * 0.1f * (expected_scale_act / 127.0f);
    float expected1 = 256.0f * 2.0f * 3.0f * 0.2f * (expected_scale_act / 127.0f);

    printf("\nExpected outputs:\n");
    printf("  Row 0: ~%.2f\n", expected0);
    printf("  Row 1: ~%.2f\n", expected1);

    // Call GEMV kernel
    printf("\nCalling GEMV kernel...\n");
    ggml_amx_gemv_q4_0_8x8_q8_0(
        TEST_K,         // n: K dimension
        output,         // s: output
        TEST_NC,        // bs: stride
        weights,        // vx: weights
        activations,    // vy: activations
        1,              // nr: always 1 for GEMV
        TEST_NC         // nc: number of rows to compute
    );

    printf("\nActual outputs:\n");
    printf("  Row 0: %.2f", output[0]);
    printf(isnan(output[0]) ? " ❌ NaN!\n" : " ✅\n");
    printf("  Row 1: %.2f", output[1]);
    printf(isnan(output[1]) ? " ❌ NaN!\n" : " ✅\n");

    // Check for correctness
    int passed = 1;
    if (isnan(output[0]) || isnan(output[1])) {
        printf("\n❌ FAIL: Output contains NaN\n");
        printf("   → Kernel has data access or computation bug\n");
        passed = 0;
    } else {
        float error0 = fabsf(output[0] - expected0) / expected0;
        float error1 = fabsf(output[1] - expected1) / expected1;

        if (error0 > 0.1f || error1 > 0.1f) {
            printf("\n⚠️  WARN: Output incorrect (>10%% error)\n");
            printf("   Row 0 error: %.1f%%\n", error0 * 100);
            printf("   Row 1 error: %.1f%%\n", error1 * 100);
            passed = 0;
        } else {
            printf("\n✅ PASS: Kernel computes correctly!\n");
        }
    }

    // Cleanup
    free(weights);
    free(activations);
    free(output);

    return passed ? 0 : 1;
}
