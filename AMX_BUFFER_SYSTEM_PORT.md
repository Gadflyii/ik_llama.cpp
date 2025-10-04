# AMX Buffer System Port - Implementation Log
**Date**: 2025-10-03 Evening (continued)
**Status**: 🟡 IN PROGRESS - Buffer list infrastructure added, tensor loading needs update

---

## Critical Discovery: Root Cause of NaN

**Previous Issue**: NaN in `ffn_moe_weights_sum-1` after first matmul
**Root Cause**: AMX buffer type was being used for ALL tensors (F32 embeddings, layer norms, etc.)
**Evidence**:
- First matmul with Q4_0 weights worked correctly (valid inputs, no NaN)
- But F32 tensors (embeddings via `buft_input`) were allocated in AMX buffers
- This broke layer norm and other F32 operations

**Key Finding from Debug Logs**:
```
Input first 8 values: 0.000011 0.000130 0.000172 -0.000004...  ← VALID (after fixing buft_input)
Previously: nan nan nan nan...  ← BROKEN (when buft_input used AMX)
```

---

## Architecture Comparison: Our Fork vs Upstream

### Upstream (llama.cpp numa_read_mirror)
**Buffer Allocation System**:
1. Uses **buffer type lists** (`buft_list_t = std::vector<pair<dev, buft>>`)
2. Each tensor selects buffer type via `select_weight_buft(tensor, op, buft_list)`
3. Tries buffer types in priority order until one supports the operation
4. **3 buffers created** for CPU-only with `--no-host`:
   - `CPU_REPACK` (14904 MB) - Repacked weights
   - `AMX` (729 MB) - VNNI-packed Q4_0/Q4_1 weights
   - `CPU_Mapped` (16217 MB) - mmap'd F32 weights
   - **Total**: ~32 GB (full model)

**Buffer Type Priority**:
```cpp
static buft_list_t make_cpu_buft_list(bool use_extra_bufts, bool no_host) {
    1. ACCEL devices (if any)
    2. GPU host buffer (SKIPPED if no_host=true)  ← --no-host disables this
    3. CPU extra buffer types (AMX, REPACK, etc.) ← AMX is FIRST extra type
    4. CPU (fallback)
}
```

**Key**: When `--no-host` is set, skips GPU host buffer and forces use of extra buffer types (AMX)

### Our Fork (ik_llama.cpp before this session)
**Buffer Allocation System**:
1. Uses **single buffer type** per layer (`model.buft_layer[i] = single_buft`)
2. No fallback mechanism - all tensors in layer use same buffer type
3. **2 buffers created** with `--amx`:
   - `CPU` (410 MB)
   - `AMX` (656 MB)
   - **Total**: ~1 GB (most of model is mmap'd)

**Problem**:
- When `buft_layer = AMX`, ALL tensors try to use AMX buffer
- F32 tensors (embeddings, norms) incorrectly allocated in AMX buffers
- No fallback to CPU buffer for unsupported tensor types

---

## Solution: Port Upstream Buffer List System

### Phase 1: ggml-backend.cpp Infrastructure ✅ COMPLETED

**Added** (`ggml/src/ggml-backend.cpp`):
```cpp
// Line 733-752: Extra buffer types registration
GGML_CALL ggml_backend_buffer_type_t * ggml_backend_cpu_get_extra_bufts(void) {
    static ggml_backend_buffer_type_t bufts[8] = {NULL};

    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        bufts[0] = ggml_backend_amx_buffer_type();  // AMX is FIRST
    #endif

    bufts[idx] = NULL;  // NULL terminator
    return bufts;
}

// Line 755-772: Buffer type operation support check
GGML_CALL bool ggml_backend_buft_supports_op(buft, op) {
    // Checks if buffer type supports operation
    // Used by select_weight_buft to choose buffer type
}
```

**Added to header** (`ggml/include/ggml-backend.h`):
```cpp
// Line 128: Extra buffer types getter
GGML_API GGML_CALL ggml_backend_buffer_type_t * ggml_backend_cpu_get_extra_bufts(void);

// Line 131: Operation support checker
GGML_API GGML_CALL bool ggml_backend_buft_supports_op(ggml_backend_buffer_type_t buft, const struct ggml_tensor * op);
```

### Phase 2: llama.cpp Buffer List System ✅ PARTIALLY COMPLETED

**Added** (`src/llama.cpp` lines 1785-1836):
```cpp
// Buffer type list typedef
using buft_list_t = std::vector<ggml_backend_buffer_type_t>;

// Select first buffer type that supports operation
static ggml_backend_buffer_type_t select_weight_buft(tensor, op, buft_list) {
    for (auto buft : buft_list) {
        if (ggml_backend_buft_supports_op(buft, tensor)) {
            return buft;
        }
    }
    return nullptr;
}

// Create CPU buffer type list
static buft_list_t make_cpu_buft_list(bool use_extra_bufts, bool no_host) {
    buft_list_t list;

    // 1. Extra buffer types (AMX) if requested
    if (use_extra_bufts) {
        ggml_backend_buffer_type_t * extra = ggml_backend_cpu_get_extra_bufts();
        while (*extra) list.push_back(*extra++);
    }

    // 2. Host buffer (for GPU transfers) if needed
    if (!no_host) {
        #ifdef GGML_USE_CUDA
            list.push_back(ggml_backend_cuda_host_buffer_type());
        #endif
    }

    // 3. CPU buffer (always fallback)
    list.push_back(ggml_backend_cpu_buffer_type());

    return list;
}
```

### Phase 3: Tensor Loading Update ⏳ TODO

**Current Problem**: Our fork still uses old model where `model.buft_layer[i]` is a single buffer type.

**What Needs to Change**:
1. Change `model.buft_layer` from `ggml_backend_buffer_type_t` to `buft_list_t`
2. Update tensor allocation code to call `select_weight_buft(tensor, op, model.buft_layer[i])`
3. Create separate buffer for each buffer type actually used
4. Track which tensors use which buffer

**Files to Modify**:
- `src/llama.cpp`: Tensor loading in `llm_load_tensors()`
- `src/llama-impl.h`: Model structure definition
- `src/llama.cpp`: Buffer allocation logic

**Estimated Changes**: ~500-1000 lines across tensor loading logic

---

## Current Status

### ✅ Completed
1. Extra buffer types infrastructure in ggml-backend.cpp
2. AMX registered as first extra buffer type
3. Buffer list creation function (`make_cpu_buft_list`)
4. Buffer type selection function (`select_weight_buft`)
5. Fixed `buft_input` and `buft_output` to NOT use AMX (line 4934, 4978, 5002)

### ⏳ In Progress
- Updating tensor loading to use buffer lists

### ⏸️ Blocked
- Testing with AMX (blocked on tensor loading update)

---

## Next Steps

1. **Update Model Structure** (highest priority)
   - Change `model.buft_layer` type to `buft_list_t` or keep as single buft but select from list during allocation
   - Simpler approach: Keep `model.buft_layer` as single buft, but select it from list at layer setup time

2. **Simplest Implementation Path**:
   Instead of changing the entire tensor loading system, we can:
   - Keep `model.buft_layer[i]` as single `ggml_backend_buffer_type_t`
   - At layer setup time (line 4938), call: `model.buft_layer[i] = select_weight_buft(representative_tensor, op, buft_list)`
   - This requires knowing representative tensor type for each layer

3. **Alternative**: Use buffer list during actual tensor allocation
   - In `llm_load_tensors()`, when allocating each tensor, select buffer type from list
   - This is more flexible but requires more extensive changes

---

## Key Insights

1. **Buffer Type Priority is Critical**:
   - AMX must be FIRST in extra buffer types
   - When `--no-host` is set, host buffer is skipped → AMX gets priority
   - Fallback to CPU buffer ensures F32 tensors work correctly

2. **`--amx` Flag Behavior**:
   - Should set `use_extra_bufts = true` (enable AMX)
   - Should set `no_host = true` (skip host buffer, prefer AMX)
   - This mirrors upstream's `--no-host` behavior

3. **Three-Buffer System Explained**:
   - **AMX buffer**: Q4_0/Q4_1 weights (VNNI-packed)
   - **CPU_REPACK buffer**: Other quantized weights (repacked)
   - **CPU_Mapped buffer**: F32 weights (mmap'd)
   - Each tensor selects appropriate buffer based on type and operation

4. **Why Our Simple Approach Failed**:
   - Using AMX for `buft_layer` meant F32 tensors tried to use AMX
   - AMX doesn't support F32 operations
   - No fallback mechanism existed
   - Result: Layer norm → NaN

---

## Testing Plan (After Tensor Loading Update)

1. Rebuild with buffer list system
2. Test CPU-only with `--amx`:
   ```bash
   numactl --cpunodebind=2 --membind=2 ./build/bin/llama-cli \
     -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
     -t 32 -n 128 -p "Test" --amx
   ```
3. Check buffer allocation logs - should see 3 buffers
4. Verify no NaN errors
5. Test token generation works correctly
6. Compare performance vs non-AMX

---

## Files Modified This Session

1. `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`
   - Added `ggml_backend_cpu_get_extra_bufts()` (lines 733-752)
   - Added `ggml_backend_buft_supports_op()` (lines 755-772)

2. `/home/ron/src/ik_llama.cpp/ggml/include/ggml-backend.h`
   - Added `ggml_backend_cpu_get_extra_bufts()` declaration (line 128)
   - Added `ggml_backend_buft_supports_op()` declaration (line 131)

3. `/home/ron/src/ik_llama.cpp/src/llama.cpp`
   - Added `buft_list_t` typedef (line 1786)
   - Added `select_weight_buft()` function (lines 1789-1801)
   - Added `make_cpu_buft_list()` function (lines 1805-1836)
   - Fixed `buft_input` to not use AMX (line 4934)
   - Fixed `buft_output` to not use AMX (lines 4978, 5002)

---

**Last Updated**: 2025-10-03 23:45
**Next Session**: Complete tensor loading update and test
