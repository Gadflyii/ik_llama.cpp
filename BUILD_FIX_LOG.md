# AMX Build Fix Log
**Date:** 2025-10-04 14:25
**Branch:** Add_AMX_Clean
**Issue:** Build failures due to compiler flag conflicts

---

## Problem Summary

**Original Issue:**
`Add_AMX_Clean` branch failed to build with error:
```
/home/ron/src/ik_llama.cpp/ggml/src/iqk/iqk_fa_templates.h:914:105:
error: cannot convert '{anonymous}::F16::Data' {aka '__m512'} to '__m256'
```

**Root Cause:**
AMX compiler flags (`-mamx-tile`, `-mamx-int8`, `-mavx512vnni`) were added to global `ARCH_FLAGS` in `ggml/src/CMakeLists.txt`, affecting ALL source files including IQK flash attention code that expects only AVX2/AVX512.

---

## Investigation Steps

### 1. Verified Baseline (main branch)
- ✅ Built successfully
- ✅ CPU test passed
- ✅ GPU test passed
- **Conclusion:** Fork baseline has NO issues

### 2. Identified Problem
Compared `Add_AMX_Clean` vs `main`:
- AMX flags in `ggml/src/CMakeLists.txt` lines 1384-1400
- Added to global `ARCH_FLAGS` affecting all files
- IQK code incompatible with AMX instruction set mix

---

## Fix Attempts

### Attempt 1: Remove from Global ARCH_FLAGS
**Change:** Removed AMX flags from global `ARCH_FLAGS`
**Result:** ❌ No flags applied to AMX files

### Attempt 2: Per-File COMPILE_FLAGS
**Change:** Used `set_source_files_properties()` with `COMPILE_FLAGS`
**Result:** ❌ `c++: fatal error: no input files`

### Attempt 3: Per-File COMPILE_OPTIONS (string)
**Change:** Used `COMPILE_OPTIONS` with string flags
**Result:** ❌ `error: unrecognized command-line option '-mamx-tile -mamx-int8 -mavx512vnni'`

### Attempt 4: Per-File COMPILE_OPTIONS (list)
**Change:** Used CMake list for flags
**Result:** ❌ `error: inlining failed in call to 'always_inline' '__m512i _mm512_inserti32x8'`
**Issue:** AMX files use AVX512 intrinsics, need AVX512 flags too

### Attempt 5: Per-File with AVX512 + AMX (CURRENT)
**Change:** Added AVX512 base flags alongside AMX flags
```cmake
set(AMX_COMPILE_FLAGS "-mamx-tile" "-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")
if (GGML_AMX_INT8 OR GGML_AMX)
    list(APPEND AMX_COMPILE_FLAGS "-mamx-int8" "-mavx512vnni")
endif()
set_source_files_properties(
    ggml-cpu/amx/amx.cpp
    ggml-cpu/amx/mmq.cpp
    ggml-cpu-traits.cpp
    PROPERTIES COMPILE_OPTIONS "${AMX_COMPILE_FLAGS}"
)
```
**Status:** Building...

---

## Key Learnings

1. **Global vs Per-File Flags:**
   - Global `ARCH_FLAGS` affects ALL compilation units
   - Per-file flags needed for specialized instruction sets
   - Use `COMPILE_OPTIONS` property, not `COMPILE_FLAGS`

2. **CMake Flag Format:**
   - String format: Single option, breaks with spaces
   - List format: Semicolon-separated, correct for multiple options
   - Example: `"-flag1" "-flag2"` not `"-flag1 -flag2"`

3. **AMX Dependencies:**
   - AMX intrinsics require AVX512 base support
   - Must include: `-mavx512f -mavx512dq -mavx512bw -mavx512vl`
   - Plus AMX-specific: `-mamx-tile -mamx-int8 -mavx512vnni`

4. **IQK Compatibility:**
   - IQK flash attention uses AVX2/AVX512 without AMX
   - Adding AMX flags globally breaks type conversions
   - Must isolate AMX flags to AMX-only files

---

## Files Modified

### `/home/ron/src/ik_llama.cpp/ggml/src/CMakeLists.txt`

**Lines 1384-1386 (removed global AMX flags):**
```cmake
# NOTE: Intel AMX flags are applied per-file below, not globally here
# This prevents conflicts with other code (IQK) that expects specific SIMD levels
```

**Lines 1536-1559 (added per-file AMX flags):**
```cmake
# Apply AMX compiler flags ONLY to AMX-specific source files
# AMX requires AVX512 base, so add those flags too
set(AMX_COMPILE_FLAGS "-mamx-tile" "-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")

if (GGML_AMX_INT8 OR GGML_AMX)
    target_compile_definitions(ggml PUBLIC GGML_USE_AMX_INT8)
    list(APPEND AMX_COMPILE_FLAGS "-mamx-int8" "-mavx512vnni")
endif()

set_source_files_properties(
    ggml-cpu/amx/amx.cpp
    ggml-cpu/amx/mmq.cpp
    ggml-cpu-traits.cpp
    PROPERTIES COMPILE_OPTIONS "${AMX_COMPILE_FLAGS}"
)
```

### `/home/ron/src/ik_llama.cpp/common/CMakeLists.txt`

**Lines 123-125 (removed AMX flags from common):**
```cmake
# NOTE: AMX flags are NOT applied to common library
# Common library doesn't contain AMX-specific code
# AMX flags are only for ggml AMX source files
```

---

## Next Steps

1. ✅ Wait for current build to complete
2. ✅ Test if build succeeds
3. ❌ Build FAILED - AMX integration issues too deep
4. 📝 Document findings and recommend clean implementation

---

## Final Verdict

After extensive investigation, the `Add_AMX` branch AMX implementation has fundamental integration issues:

### Critical Problems Found:

1. **Missing Core Functions:** `ggml_barrier()` and `ggml_get_numa_strategy()` exist in Add_AMX but conflict with internal ggml.c functions
2. **Name Conflicts:** Static `ggml_barrier(compute_state_shared*)` vs public `ggml_barrier(threadpool*)`
3. **Type Mismatches:** AMX code casts `threadpool*` to `compute_state_shared*` assuming layout compatibility
4. **Missing Headers:** AMX code relies on ggml-cpu-compat.h and ggml-cpu-impl.h with specific declarations
5. **Infrastructure Mismatch:** Fork's ggml.c has different threading model than upstream AMX expects

### Conclusion:

The AMX code from `Add_AMX` branch was written for a different version of ggml.c with incompatible threading infrastructure. Properly integrating it requires:
- Rewriting threading/barrier abstractions
- Exporting internal functions safely
- Resolving all type conflicts
- Testing thread safety thoroughly

**Recommendation:** Defer AMX implementation until a clean integration path is designed, or port AMX kernels to work with current fork's infrastructure.

---

## Build Log

Final build attempt: `/tmp/amx_build_success.log`
Status: FAILED - conflicting function signatures
Error: `ggml_barrier` defined both as `static void(compute_state_shared*)` and `void(threadpool*)`
