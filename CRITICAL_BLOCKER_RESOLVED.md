# RESOLVED: AMX Buffer Type Issue

**Date**: 2025-10-03
**Status**: ✅ **RESOLVED**
**Original Status**: 🔴 **BLOCKING AMX TESTING**

---

## Resolution

**Root Cause**: The AMX buffer type was changing `tensor->type = GGML_TYPE_Q4_0_8_8` after repacking, which caused GGML to dispatch to incorrect kernels that couldn't read the repacked data format.

**Fix**:
1. Removed `tensor->type = GGML_TYPE_Q4_0_8_8` line from buffer type (keeps tensor as Q4_0)
2. Disabled AMX buffer type selection until tensor_traits compute override is implemented
3. Model now runs correctly with and without `--amx` flag

**Test Results After Fix:**
```bash
# Without --amx
$ ./build/bin/llama-cli -m Qwen3-30B-Q4_0.gguf -t 1 -n 5 -p "2+2="
Output: 2+2=?
Answer: 4
✅ WORKS

# With --amx (buffer type disabled, runs as baseline)
$ ./build/bin/llama-cli -m Qwen3-30B-Q4_0.gguf -t 1 -n 5 -p "2+2=" --amx
Output: 2+2=4.000
✅ WORKS
```

---

## Original Problem (MISDIAGNOSIS)

Initially appeared that ik_llama.cpp had a base code bug preventing the MoE model from running. Investigation showed:
- Prompt evaluation: `prompt eval time = 0.00 ms / 0 tokens` (appeared to not process prompt)
- Output: Garbage ("test = [1")
- NaN in activation data before first MUL_MAT

**Actual Cause**: This was NOT a base code bug. It was caused by:

1. **AMX buffer type repacked weights**: Q4_0 nibbles → unpacked INT8 format
2. **Changed tensor type**: `tensor->type = GGML_TYPE_Q4_0_8_8`
3. **Registered custom kernels**: GEMV/GEMM for Q4_0_8_8 type
4. **Data format mismatch**:
   - Weights were in unpacked INT8 format
   - But kernels expected standard Q4_0 packed nibble format
   - Reading unpacked data as packed → garbage → NaN

---

## Lessons Learned

1. **Don't change tensor->type without compute override**
   Repacking data format requires overriding the compute dispatch, not just changing type metadata.

2. **tensor_traits is essential**
   Upstream uses tensor_traits to intercept `compute_forward()` and route to custom kernels. Without this, changing tensor type breaks dispatch.

3. **Buffer type alone is insufficient**
   The buffer type can allocate/repack data, but you need tensor_traits to USE that data during inference.

---

## Next Steps

1. ✅ Model verified working (baseline performance)
2. 🔴 Implement tensor_traits compute override system
3. 🔴 Re-enable AMX buffer type with tensor_traits
4. 🔴 Test AMX kernels are called correctly
5. 🔴 Implement AMX tile operations for acceleration
6. 🔴 Benchmark AMX vs baseline

---

## Technical Details

### What Upstream Does (Correct)

```cpp
// amx.cpp - Buffer type sets tensor_traits
static enum ggml_status ggml_backend_amx_buffer_init_tensor(...) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);
    return GGML_STATUS_SUCCESS;
}

// amx.cpp - tensor_traits intercepts compute
class tensor_traits : public ggml::cpu::tensor_traits {
    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);  // Custom compute
            return true;  // Prevent default dispatch
        }
        return false;
    }
};
```

### What We Did (Incorrect)

```cpp
// ggml-backend.cpp - Changed tensor type without compute override
static void ggml_amx_repack_q4_0(struct ggml_tensor * tensor, ...) {
    // Repack nibbles to INT8
    ggml_repack_q4_0_to_q4_0x8(...);

    // ❌ WRONG: Changes type but dispatch still uses default Q4_0 kernels
    tensor->type = GGML_TYPE_Q4_0_8_8;
}
```

Result: GGML's Q4_0 kernels read INT8 unpacked data as if it were packed nibbles → garbage.

### Fix

```cpp
// Don't change type - keep as Q4_0
// Repacking happens but isn't used until we implement tensor_traits
// For now, AMX buffer type is disabled
```

---

## Files Modified

- ✅ `ggml/src/ggml-backend.cpp` - Removed `tensor->type` change
- ✅ `src/llama.cpp` - Disabled AMX buffer type selection
- ✅ `AMX_STATUS.md` - Updated status to "Ready for tensor_traits"
- ✅ `CRITICAL_BLOCKER.md` → `CRITICAL_BLOCKER_RESOLVED.md`

---

**Conclusion**: There was NO base code bug. The issue was entirely in the AMX implementation approach. Model works correctly now.
