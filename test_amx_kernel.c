// Standalone unit test for AMX kernels
// Tests GEMV kernel with known-good synthetic data
// Compile: gcc -o test_amx test_amx_kernel.c ggml/src/ggml-amx-repack.c -I ggml/include -I ggml/src -lm -mavx512f -mavx512vnni -mamx-int8

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml-amx-repack.h"

// Simple test: 1 output row, 256 elements (8 Q4_0 blocks)
#define TEST_K 256
#define TEST_NC 1
#define TEST_NB 8  // 256 / 32 = 8 Q8_0 blocks
#define TEST_NBX8 1  // 8 / 8 = 1 Q4_0x8 block

int main() {
    printf("AMX Kernel Unit Test\n");
    printf("====================\n\n");

    // Allocate test data
    block_q4_0x8_unpacked *weights = malloc(sizeof(block_q4_0x8_unpacked) * TEST_NBX8);
    block_q8_0 *activations = malloc(sizeof(block_q8_0) * TEST_NB);
    float *output = malloc(sizeof(float) * TEST_NC);

    if (!weights || !activations || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize weights: simple pattern
    // Set all scales to 1.0 for easy math
    // Set all quantized values to 1 (INT8)
    printf("Initializing test data...\n");
    for (int i = 0; i < TEST_NBX8; i++) {
        for (int j = 0; j < 8; j++) {
            weights[i].d[j] = GGML_FP32_TO_FP16(1.0f);  // scale = 1.0
            for (int k = 0; k < 32; k++) {
                weights[i].qs[j * 32 + k] = 1;  // all weights = 1
            }
        }
    }

    // Initialize activations: simple pattern
    // Set all scales to 1.0 for easy math
    // Set all quantized values to 2 (INT8)
    for (int i = 0; i < TEST_NB; i++) {
        activations[i].d = GGML_FP32_TO_FP16(1.0f);  // scale = 1.0
        for (int k = 0; k < 32; k++) {
            activations[i].qs[k] = 2;  // all activations = 2
        }
    }

    // Expected result: sum(1 * 2) over 256 elements = 512
    float expected = 256.0f * 1.0f * 1.0f * 2.0f;  // K * weight_scale * act_scale * (weight_val * act_val)
    printf("Expected output: %.2f\n\n", expected);

    // Call kernel
    printf("Calling GEMV kernel...\n");
    ggml_amx_gemv_q4_0_8x8_q8_0(
        TEST_K,          // n: K dimension
        output,          // s: output buffer
        TEST_NC,         // bs: batch stride
        weights,         // vx: weights
        activations,     // vy: activations
        1,               // nr: number of output rows
        TEST_NC          // nc: number of output elements
    );

    printf("\nResults:\n");
    printf("  Expected: %.2f\n", expected);
    printf("  Actual:   %.2f\n", output[0]);

    float error = fabsf(output[0] - expected);
    float rel_error = error / expected;

    printf("  Error:    %.6f (%.2f%%)\n", error, rel_error * 100.0f);

    // Check for NaN
    if (isnan(output[0])) {
        printf("\n❌ FAIL: Output is NaN!\n");
        printf("   → Kernel has a bug or data is corrupted\n");
    } else if (rel_error < 0.01f) {  // 1% tolerance
        printf("\n✅ PASS: Kernel produces correct output\n");
        printf("   → Bug is in upstream data pipeline, not kernel\n");
    } else {
        printf("\n⚠️  WARN: Output is valid but incorrect\n");
        printf("   → Kernel may have logic error\n");
    }

    // Cleanup
    free(weights);
    free(activations);
    free(output);

    return (isnan(output[0]) || rel_error >= 0.01f) ? 1 : 0;
}
