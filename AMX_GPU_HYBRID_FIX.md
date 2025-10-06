# AMX+GPU Hybrid Mode Bug Fix

## Issue Summary
AMX implementation crashed with CUDA illegal memory access errors when GPU offloading was enabled (`-ngl > 0`).

## Root Cause
The fork's AMX buffer type incorrectly returned `is_host = true`, misleading the backend scheduler into assuming CUDA could directly access AMX memory. AMX uses special aligned memory that is not accessible by CUDA, causing crashes in hybrid CPU+GPU mode.

## The Fix
**File**: `ggml/src/ggml-cpu/amx/amx.cpp:267-274`

Changed `ggml_backend_amx_buffer_type_is_host()` to return `false`:

```cpp
static bool ggml_backend_amx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // AMX buffers use special aligned memory that may not be directly accessible by GPU backends.
    // In the fork without device registry infrastructure, returning false forces proper buffer
    // copies via the scheduler instead of assuming direct cross-backend memory access.
    // This is critical for CPU+GPU hybrid mode where CUDA cannot directly access AMX memory.
    return false;
}
```

## Technical Details

### Why the Fork Differs from Upstream
- **Upstream (llama.cpp)**: Has device registry infrastructure (added Oct 2024) with `ggml_backend_dev_t` that properly manages cross-device memory transfers
- **Fork (ik_llama.cpp)**: Based on pre-device-registry codebase (July 2024), uses older backend API without device field

### How the Fix Works
By returning `is_host = false`, the backend scheduler (`ggml-backend.cpp:363-376`) now:
1. Detects that AMX and CUDA buffers are incompatible
2. Uses proper copy operations (`ggml_backend_buffer_copy_tensor` or fallback copy via CPU)
3. Avoids direct memory access that would cause illegal access errors

This adapts upstream's device registry behavior to the fork's older backend API.

## Test Results

### Before Fix
```
Fork GPU+AMX (-ngl 1):  CRASHED (CUDA illegal memory access)
Fork GPU+AMX (-ngl 20): CRASHED (CUDA illegal memory access)
Fork GPU+AMX (-ngl 30): CRASHED (CUDA illegal memory access)
```

### After Fix
```
Fork GPU+AMX (-ngl 1):  ✅ WORKS
Fork GPU+AMX (-ngl 20): ✅ WORKS
Fork GPU+AMX (-ngl 30): ✅ WORKS

AMX buffer allocation: 334.12 MiB (with -ngl 15)
CUDA buffer allocation: 5027.13 MiB (with -ngl 15)
No CUDA errors
```

## Usage
AMX now works correctly in all modes:
- **CPU-only** (`--amx`): AMX acceleration for all layers
- **CPU+GPU hybrid** (`--amx -ngl N`): AMX for CPU layers, GPU for offloaded layers, proper data transfer between them

## Related Files
- Fix: `ggml/src/ggml-cpu/amx/amx.cpp`
- Buffer selection: `src/llama.cpp:1824-1854` (`make_cpu_buft_list`)
- Scheduler copy logic: `ggml/src/ggml-backend.cpp:363-377`

## Date
October 5, 2025

## Branch
`Add_AMX_Clean`
