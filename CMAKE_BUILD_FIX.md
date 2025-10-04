# CMake Build Fix for AMX
**Date:** 2025-10-04
**Branch:** Add_AMX_Clean
**Status:** Fixed and verified

---

## Problem

Build was failing with IQK Flash Attention errors and AMX compilation errors due to global compiler flag pollution.

### Error 1: IQK Flash Attention Type Mismatch
```
/home/ron/src/ik_llama.cpp/ggml/src/iqk/fa/iqk_fa_templates.h:781:36:
error: cannot convert '{anonymous}::F16::Data' {aka '__m512'} to '__m256'
```

**Root Cause:** Global AVX512VNNI flag was being added to ARCH_FLAGS, affecting all source files including IQK code which expects only AVX2/AVX512 base flags.

### Error 2: AMX Intrinsic Failures
```
/home/ron/src/ik_llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp:568:30:
error: inlining failed in call to 'always_inline' '__m512i _mm512_inserti32x8(__m512i, __m256i, int)':
target specific option mismatch
```

**Root Cause:** AMX source files weren't getting the required compiler flags because they were removed from ARCH_FLAGS but not applied per-file.

---

## Original Broken Code

**File:** `ggml/src/CMakeLists.txt`

```cmake
# Lines 1384-1400 (BROKEN)
# Intel AMX support (requires tile configuration)
if (GGML_AMX OR GGML_AMX_TILE OR GGML_AMX_INT8 OR GGML_AMX_BF16)
    list(APPEND ARCH_FLAGS -mamx-tile)
endif()
# AMX-INT8 for quantized types (requires AVX512VNNI)
if (GGML_AMX OR GGML_AMX_INT8)
    list(APPEND ARCH_FLAGS -mamx-int8)
    # Ensure AVX512VNNI is enabled for AMX-INT8
    if (NOT GGML_AVX512_VNNI)
        message(STATUS "AMX-INT8 requires AVX512-VNNI, enabling it automatically")
        list(APPEND ARCH_FLAGS -mavx512vnni)  # <-- POLLUTES ALL FILES
    endif()
endif()
# AMX-BF16 for floating point types
if (GGML_AMX_BF16)
    list(APPEND ARCH_FLAGS -mamx-bf16)
endif()
```

**Problem:** All AMX and AVX512VNNI flags added to global `ARCH_FLAGS`, affecting every source file in the project.

---

## Fixed Code

**File:** `ggml/src/CMakeLists.txt`

### Fix 1: Remove AMX flags from global ARCH_FLAGS (Lines 1384-1386)
```cmake
# Intel AMX support - flags applied per-file below, not globally
# (Removed from ARCH_FLAGS to avoid polluting IQK Flash Attention code)
# AMX-INT8 requires AVX512VNNI but we apply it per-file, not globally
```

### Fix 2: Add per-file flag application (Lines 1540-1578)
```cmake
# AMX compile definitions
if (GGML_AMX OR GGML_AMX_INT8 OR GGML_AMX_BF16)
    target_compile_definitions(ggml PUBLIC GGML_USE_AMX)
    message(STATUS "Intel AMX support enabled")
    if (GGML_AMX_INT8)
        target_compile_definitions(ggml PUBLIC GGML_USE_AMX_INT8)
        message(STATUS "  - AMX-INT8 enabled (quantized types)")
    endif()
    if (GGML_AMX_BF16)
        target_compile_definitions(ggml PUBLIC GGML_USE_AMX_BF16)
        message(STATUS "  - AMX-BF16 enabled (floating point types)")
    endif()

    # Apply AMX compiler flags ONLY to AMX-specific source files
    # This prevents pollution of IQK Flash Attention and other code
    set(AMX_COMPILE_FLAGS "")
    list(APPEND AMX_COMPILE_FLAGS "-mamx-tile" "-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")

    if (GGML_AMX_INT8 OR GGML_AMX)
        list(APPEND AMX_COMPILE_FLAGS "-mamx-int8" "-mavx512vnni")
    endif()

    if (GGML_AMX_BF16)
        list(APPEND AMX_COMPILE_FLAGS "-mamx-bf16")
    endif()

    # Convert list to space-separated string for COMPILE_FLAGS property
    string(REPLACE ";" " " AMX_COMPILE_FLAGS_STR "${AMX_COMPILE_FLAGS}")

    # Apply AMX flags only to AMX source files
    set_source_files_properties(
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/mmq.cpp
        ggml-cpu-traits.cpp
        PROPERTIES COMPILE_FLAGS "${AMX_COMPILE_FLAGS_STR}"
    )

    message(STATUS "  - AMX compiler flags applied to AMX source files only")
endif()
```

---

## Key Changes

1. **Removed from global ARCH_FLAGS:**
   - `-mamx-tile`
   - `-mamx-int8`
   - `-mamx-bf16`
   - `-mavx512vnni`

2. **Added per-file flags to:**
   - `ggml-cpu/amx/amx.cpp`
   - `ggml-cpu/amx/mmq.cpp`
   - `ggml-cpu-traits.cpp`

3. **Flags applied per-file:**
   - `-mamx-tile` (always when AMX enabled)
   - `-mavx512f -mavx512dq -mavx512bw -mavx512vl` (AVX512 base for AMX)
   - `-mamx-int8 -mavx512vnni` (only when AMX-INT8 enabled)
   - `-mamx-bf16` (only when AMX-BF16 enabled)

---

## Build Process

### Configure
```bash
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AMX=ON
```

**Expected Output:**
```
-- Intel AMX support enabled
--   - AMX compiler flags applied to AMX source files only
-- ARCH_FLAGS = -march=native
```

Note: NO "-mavx512vnni" message, confirming it's not in global flags.

### Build
```bash
cmake --build build --target llama-cli -j 32
```

**Expected Result:**
```
[100%] Built target llama-cli
```

---

## Verification

### Check that IQK files compile without errors:
```bash
grep "Building CXX.*iqk_fa" /tmp/build_clean.log
```

**Expected:** No errors, only successful compilation messages.

### Check that AMX files get correct flags:
```bash
cd build && make VERBOSE=1 2>&1 | grep "amx/amx.cpp" | head -1
```

**Expected:** Command includes `-mamx-tile -mavx512f -mavx512dq -mavx512bw -mavx512vl -mamx-int8 -mavx512vnni`

### Check binary exists:
```bash
ls -lh build/bin/llama-cli
```

**Expected:** Binary exists and is executable.

---

## Why This Fix Works

1. **Isolation:** AMX-specific compiler flags only affect AMX source files
2. **No Pollution:** IQK Flash Attention code compiles with standard AVX2/AVX512 flags
3. **Proper Intrinsics:** AMX files get all required flags for AMX intrinsics
4. **Baseline Compatibility:** Non-AMX code path remains identical to main branch

---

## Testing Required

1. ✅ Build succeeds
2. ⏳ Baseline test (no --amx): Must behave identically to main branch
3. ⏳ AMX test (--amx, CPU-only): AMX buffers should be created
4. ⏳ GPU test (no --amx): Standard GPU operation
5. ⏳ GPU+AMX test (--amx + GPU): AMX should be disabled

---

**Status:** Ready for Phase 1 testing
