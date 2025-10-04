# Phase 1 Analysis: AMX Isolation & GPU Fix
**Created:** 2025-10-04 16:40
**Branch:** Add_AMX_Clean
**Task:** Identify and fix AMX conditional activation issues

---

## Problem Statement

Current Add_AMX_Clean implementation changes baseline behavior even without `--amx` flag. Need to ensure:

1. WITHOUT `--amx`: Must be identical to main branch
2. WITH `--amx`: Enable AMX buffers for CPU-only inference
3. GPU must work perfectly in all cases

---

## Key Changes vs Main Branch

### 1. CLI Flag Addition
**Files:** `common/common.h`, `common/common.cpp`
- Added `bool use_amx` parameter
- Added `--amx` command line flag
- Passed through to model params

**Status:** ✅ Clean - only activates when flag present

### 2. Model Parameters
**Files:** `include/llama.h`
- Added `bool use_amx` to `llama_model_params`
- Passed to tensor loading function

**Status:** ✅ Clean - only used when explicitly set

### 3. Buffer Type Function Signature
**Files:** `src/llama-impl.h`, `src/llama.cpp`

**MAIN BRANCH:**
```cpp
ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer);
```

**ADD_AMX_CLEAN:**
```cpp
ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer, bool use_amx = false);
```

**Status:** ✅ Safe - default value maintains backward compatibility

### 4. Buffer Type Selection Logic
**File:** `src/llama.cpp:llama_default_buffer_type_cpu()`

**MAIN BRANCH:**
```cpp
ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer) {
    ggml_backend_buffer_type_t buft = nullptr;

#if defined(GGML_USE_CUDA)
    if (host_buffer) {
        buft = ggml_backend_cuda_host_buffer_type();
    }
#elif defined(GGML_USE_SYCL)
    if (host_buffer) {
        buft = ggml_backend_sycl_host_buffer_type();
    }
#endif

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }
    return buft;
}
```

**ADD_AMX_CLEAN:**
```cpp
ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer, bool use_amx) {
    ggml_backend_buffer_type_t buft = nullptr;

#ifdef GGML_USE_AMX
    // Enable AMX buffer type when requested
    if (use_amx && !host_buffer) {
        buft = ggml_backend_amx_buffer_type();
        if (buft != nullptr) {
            LLAMA_LOG_INFO("%s: using AMX buffer type\n", __func__);
            return buft;
        }
    }
#endif

    // [SAME AS MAIN BRANCH from here]
#if defined(GGML_USE_CUDA)
    if (host_buffer) {
        buft = ggml_backend_cuda_host_buffer_type();
    }
#elif defined(GGML_USE_SYCL)
    if (host_buffer) {
        buft = ggml_backend_sycl_host_buffer_type();
    }
#endif

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }
    return buft;
}
```

**Analysis:**
- ✅ AMX code only runs when `use_amx=true` AND `!host_buffer`
- ✅ Falls through to identical main branch logic otherwise
- ✅ Should be safe

### 5. Tensor Loading Logic (CRITICAL SECTION)
**File:** `src/llama.cpp:llm_load_tensors()`

**MAIN BRANCH:**
```cpp
static bool llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        int mla_attn,
        enum llama_split_mode split_mode,
        int main_gpu,
        const float * tensor_split,
        bool use_mlock,
        bool validate_quants,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {

    const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);

    // Input layer
    model.buft_input = llama_default_buffer_type_cpu(true);

    model.buft_layer.resize(n_layer);

    // CPU layers
    for (int i = 0; i < i_gpu_start; ++i) {
        model.buft_layer[i] = llama_default_buffer_type_cpu(true);
    }

    // [GPU layer allocation logic]
    // ...
}
```

**ADD_AMX_CLEAN:**
```cpp
static bool llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        int mla_attn,
        enum llama_split_mode split_mode,
        int main_gpu,
        const float * tensor_split,
        bool use_mlock,
        bool validate_quants,
        bool use_amx,  // ADDED PARAMETER
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {

    const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);

    // NEW LOGIC
    bool need_host_buffer = (n_gpu_layers > 0);
    bool use_extra_bufts = use_amx;
    bool no_host = use_amx && (n_gpu_layers == 0);

    // Input layer
    model.buft_input = llama_default_buffer_type_cpu(need_host_buffer, false);

    model.buft_layer.resize(n_layer);
    model.buft_layer_list.resize(n_layer);  // NEW FIELD

    // CPU layers
    for (int i = 0; i < i_gpu_start; ++i) {
        // Legacy buft_layer
        model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, false);

        // NEW: Buffer list
        model.buft_layer_list[i] = make_cpu_buft_list(use_extra_bufts, no_host);
    }

    // [GPU layer allocation logic - modified]
    // ...
}
```

**ISSUES IDENTIFIED:**

❌ **Issue 1: `need_host_buffer` logic differs from main**
- Main always passes `true` to `llama_default_buffer_type_cpu()`
- Add_AMX_Clean passes `need_host_buffer = (n_gpu_layers > 0)`
- When `n_gpu_layers = 0`, passes `false` instead of `true`
- This changes baseline behavior!

❌ **Issue 2: `buft_layer_list` always created**
- Main branch doesn't have this field
- Add_AMX_Clean creates it even when `use_amx=false`
- Adds memory overhead
- Could affect buffer allocation

❌ **Issue 3: Buffer counting logic modified**
- Add_AMX_Clean counts buffers from both `buft_layer` and `buft_layer_list`
- Changes how buffer contexts are created
- Could affect GPU allocation

---

## Root Cause Analysis

### Problem 1: Host Buffer Logic Change

**Main Branch Behavior:**
- Always uses `host_buffer=true` for CPU layers
- Gets CUDA host buffer when CUDA available
- Falls back to CPU buffer otherwise

**Add_AMX_Clean Behavior:**
- Uses `host_buffer = (n_gpu_layers > 0)`
- When CPU-only: `host_buffer=false`, gets CPU buffer
- When GPU present: `host_buffer=true`, gets CUDA host buffer

**Impact:**
- CPU-only inference: Different buffer type (CPU vs CUDA host)
- May affect memory allocation strategy
- May affect performance
- **VIOLATES "no baseline changes" requirement**

### Problem 2: Buffer List System

**Issue:** The `buft_layer_list` system exists even when AMX disabled.

**Should be:** Only create buffer lists when AMX enabled.

### Problem 3: Per-Tensor Buffer Selection

**New Functions Added:**
- `select_weight_buft()` - selects buffer for each tensor
- `make_cpu_buft_list()` - creates buffer priority list

**Issue:** These functions are called even when `use_amx=false`

---

## Fix Strategy

### Goal
WITHOUT `--amx`: Behavior MUST match main branch exactly.

### Required Changes

#### 1. Fix Host Buffer Logic
```cpp
// OLD (Add_AMX_Clean):
bool need_host_buffer = (n_gpu_layers > 0);
model.buft_input = llama_default_buffer_type_cpu(need_host_buffer, false);

// FIX: Match main branch
if (use_amx && n_gpu_layers == 0) {
    // AMX path: can optimize host buffer usage
    model.buft_input = llama_default_buffer_type_cpu(false, false);
} else {
    // BASELINE path: MUST match main exactly
    model.buft_input = llama_default_buffer_type_cpu(true, false);
}
```

#### 2. Conditional Buffer List Creation
```cpp
// Only create buffer lists when AMX enabled
if (use_amx) {
    model.buft_layer_list.resize(n_layer);
    // ... populate lists
} else {
    // Baseline: don't create lists at all
    // Just use legacy buft_layer exactly as main branch does
}
```

#### 3. Conditional Buffer Selection
```cpp
// In tensor allocation code:
if (use_amx && model.buft_layer_list.size() > 0) {
    // Use per-tensor selection with buffer lists
    buft = select_weight_buft(tensor, op, model.buft_layer_list[layer]);
} else {
    // Baseline: use legacy buft_layer exactly as main
    buft = model.buft_layer[layer].buft;
}
```

#### 4. Fix CPU Layer Allocation
```cpp
// CPU layers
for (int i = 0; i < i_gpu_start; ++i) {
    if (use_amx && n_gpu_layers == 0) {
        // AMX CPU-only path
        model.buft_layer[i] = llama_default_buffer_type_cpu(false, false);
        model.buft_layer_list[i] = make_cpu_buft_list(true, true);
    } else {
        // BASELINE path: MUST match main exactly
        model.buft_layer[i] = llama_default_buffer_type_cpu(true, false);
        // Don't create buft_layer_list
    }
}
```

---

## Implementation Plan

### Step 1: Backup Current State
- ✅ Already backed up to ~/src/claudebackup/

### Step 2: Modify llm_load_tensors()
- Change host_buffer logic to match main baseline
- Make buffer list creation conditional on `use_amx`
- Ensure GPU path unchanged from main

### Step 3: Modify Tensor Allocation
- Make per-tensor selection conditional
- Fallback to main branch logic when AMX disabled

### Step 4: Test
- Build without --amx, test (must match main exactly)
- Build with --amx, CPU-only test (AMX should activate)
- Build with --amx, GPU test (AMX should be disabled)

---

## Success Criteria

### Test 1: Baseline (no --amx, no GPU)
```bash
env CUDA_VISIBLE_DEVICES='' ./build/bin/llama-cli -m MODEL -t 64 -n 10 -p "Test"
```
**Expected:** Buffer allocation IDENTICAL to main branch

### Test 2: Baseline (no --amx, with GPU)
```bash
./build/bin/llama-cli -m MODEL -t 64 -ngl 10 -n 10 -p "Test"
```
**Expected:** Buffer allocation IDENTICAL to main branch, no errors

### Test 3: AMX (--amx, no GPU)
```bash
env CUDA_VISIBLE_DEVICES='' ./build/bin/llama-cli -m MODEL -t 64 -n 10 -p "Test" --amx
```
**Expected:** AMX buffers created, performance gain

### Test 4: AMX Rejection (--amx, with GPU)
```bash
./build/bin/llama-cli -m MODEL -t 64 -ngl 10 -n 10 -p "Test" --amx
```
**Expected:** NO AMX buffers, uses host buffers, no errors

---

## Files to Modify

1. `src/llama.cpp` - Main fixes
   - Line ~5000: llm_load_tensors() logic
   - Line ~1782: llama_default_buffer_type_cpu() (likely OK as-is)
   - Line ~1790: make_cpu_buft_list() (add conditional)
   - Tensor allocation loops (multiple locations)

2. Test and verify - no other files should need changes

---

**Next Action:** Implement fixes in src/llama.cpp
