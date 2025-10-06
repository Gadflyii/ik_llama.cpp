# AMX Implementation - Final Solution Summary

## Problem Solved ✅

Fixed Intel AMX support to work with all Q/K/V/output attention weights in CPU+GPU hybrid mode.

## Symptoms (Before Fix)

When V weights were placed in AMX buffers alongside Q and K weights with GPU present:
- System would crash or produce NaN errors
- AMX would stop computing ALL Q/K/V projections (not just V)
- Regular CPU backend would try to read VNNI-formatted weights directly
- Result: Corrupted output in MoE router leading to NaN propagation

Test results showed:
- Q+K in AMX + GPU: ✅ Works fine
- Q+K+V in AMX + GPU: ❌ NaN errors

## Root Cause

The CPU backend's `ggml_backend_cpu_supports_buft()` function was rejecting AMX buffers because:

1. AMX buffer type correctly sets `is_host=false` to prevent CUDA from assuming direct memory access
2. CPU backend only accepted buffers where `is_host=true`
3. Backend scheduler calls `ggml_backend_sched_buffer_supported()` to check if CPU can handle AMX buffers
4. CPU backend said "no" → scheduler routed Q/K/V operations to other backends
5. Operations never reached AMX dispatch code
6. Non-AMX backends tried to read VNNI-formatted weights as regular format
7. Result: NaN corruption

## Solution

Two modifications to `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`:

### Fix 1: Accept AMX Buffer Type (lines 979-995)

```cpp
GGML_CALL static bool ggml_backend_cpu_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // CPU backend supports host buffers and also AMX buffers (which are CPU-based but not "host")
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }

#ifdef GGML_USE_AMX
    // AMX buffers are CPU-based even though is_host=false
    if (buft == ggml_backend_amx_buffer_type()) {
        return true;
    }
#endif

    return false;

    GGML_UNUSED(backend);
}
```

**Purpose:** Tells the CPU backend it can handle AMX buffers in addition to regular host buffers.

### Fix 2: Check Buffer Type Constraints (lines 959-971)

```cpp
case GGML_OP_MUL_MAT:
    // For operations with src[0] in AMX buffer, check AMX-specific constraints
    // AMX requires src[1] to be F32 and in host buffer, which regular CPU doesn't care about
    if (op->src[0] && op->src[0]->buffer && op->src[0]->buffer->buft) {
        // Let buffer type decide if it can handle this specific operation
        // This calls AMX's extra_buffer_type::supports_op() which checks:
        // - src[1] is F32
        // - src[1] is in host buffer (not AMX buffer)
        // - Dimensions are contiguous 2D
        return ggml_backend_buft_supports_op(op->src[0]->buffer->buft, op);
    }
    return true;
```

**Purpose:** Delegates MUL_MAT operation support checks to the buffer type, which allows AMX to enforce its specific requirements (activation tensors must be F32 in host buffers).

## Test Results (After Fix)

All scenarios working perfectly:

- ✅ **CPU-only mode:** Q+K+V all computed by AMX
- ✅ **`-ngl 1`:** Q+K+V all computed by AMX, 1 GPU layer
- ✅ **`-ngl 10`:** Q+K+V all computed by AMX, 10 GPU layers
- ✅ **No NaN errors**
- ✅ **Coherent text generation**
- ✅ **AMX verified to be computing Q/K/V projections**

Example output:
```
The capital of France is Paris. What is the capital of Spain?
The capital of Spain is Madrid.
```

## Files Modified

1. `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`
   - Modified `ggml_backend_cpu_supports_buft()` (lines 979-995)
   - Modified `ggml_backend_cpu_supports_op()` (lines 959-971)

No other changes required! The AMX buffer type implementation, tensor traits system, and weight conversion code were all working correctly.

## Key Insights

1. **AMX `is_host=false` is correct** - prevents CUDA from direct memory access to VNNI-formatted data
2. **Scheduler buffer compatibility checks are critical** - `supports_buft()` determines if a backend can handle a buffer type
3. **Buffer-type-specific operation constraints** - AMX requires activation tensors (src[1]) to be F32 in host buffers
4. **Fork's simpler architecture works** - tensor traits via `tensor->extra` field is sufficient, no need for full device registry

## Compatibility

- Works with upstream llama.cpp architecture patterns
- Maintains fork's simpler buffer type system
- No changes to AMX kernels or weight conversion
- No changes to model loading or tensor allocation
- Backward compatible - baseline behavior unchanged when `--amx` not specified

## Future Work

- Performance benchmarking vs baseline CPU and vs upstream AMX
- Additional optimization opportunities in AMX kernels
- Support for additional quantization types
- Integration with other CPU acceleration methods

## References

- Investigation log: `docs/amx_gpu_hybrid_investigation.md`
- Implementation overview: `docs/amx_implementation.md`
- Upstream llama.cpp: `/home/ron/src/llama.cpp`
- Test model: `/mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf`
