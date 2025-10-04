# Phase 1 Fixes Applied
**Date:** 2025-10-04 16:50
**Branch:** Add_AMX_Clean
**Status:** Fixes implemented, ready for testing

---

## Changes Made

### Goal
Fix conditional AMX activation to ensure baseline behavior matches main branch exactly when --amx flag is not set.

### Files Modified

**File:** `src/llama.cpp`

---

## Specific Changes

### 1. Input Layer Buffer (Line ~5008)
**Before:**
```cpp
bool need_host_buffer = (n_gpu_layers > 0);
model.buft_input = llama_default_buffer_type_cpu(need_host_buffer, false);
```

**After:**
```cpp
// CRITICAL FIX: Match main branch exactly
model.buft_input = llama_default_buffer_type_cpu(true);
```

**Reason:** Main always uses `host_buffer=true`. Our conditional logic changed this for CPU-only, violating "no baseline changes" requirement.

---

### 2. Buffer List Creation (Line ~5012-5015)
**Before:**
```cpp
model.buft_layer.resize(n_layer);
model.buft_layer_list.resize(n_layer);  // Always created!
```

**After:**
```cpp
model.buft_layer.resize(n_layer);

// Only create buffer lists when AMX is actually enabled
if (use_amx) {
    model.buft_layer_list.resize(n_layer);
}
```

**Reason:** Buffer lists should only exist when AMX is enabled, not in baseline mode.

---

### 3. CPU Layer Allocation (Line ~5018-5025)
**Before:**
```cpp
for (int i = 0; i < i_gpu_start; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, false);
    model.buft_layer_list[i] = make_cpu_buft_list(use_extra_bufts, no_host);
}
```

**After:**
```cpp
for (int i = 0; i < i_gpu_start; ++i) {
    // BASELINE: Must match main branch exactly when --amx not set
    model.buft_layer[i] = llama_default_buffer_type_cpu(true);

    // AMX path: Only populate buffer list when --amx flag is set
    if (use_amx) {
        bool use_extra_bufts = (n_gpu_layers == 0);  // Only enable AMX for pure CPU
        bool no_host = (n_gpu_layers == 0);          // Skip host buffer if CPU-only
        model.buft_layer_list[i] = make_cpu_buft_list(use_extra_bufts, no_host);
    }
}
```

**Reason:**
- Baseline: Always use `host_buffer=true` (matches main)
- AMX: Only create buffer list when `use_amx=true`
- AMX: Only enable AMX buffers for pure CPU (no GPU layers)

---

### 4. GPU Layer Allocation - Split Mode (Line ~5062-5068)
**Before:**
```cpp
for (int i = i_gpu_start; i < n_layer; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_offload(model, layer_gpu);

    buft_list_t gpu_list;
    gpu_list.push_back(llama_default_buffer_type_offload(model, layer_gpu));
    model.buft_layer_list[i] = gpu_list;  // Always created!
}
```

**After:**
```cpp
for (int i = i_gpu_start; i < n_layer; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_offload(model, layer_gpu);

    // GPU layers: buffer list is just GPU buffer (no AMX on GPU layers)
    // Only create if AMX enabled (though it won't be used for GPU layers)
    if (use_amx) {
        buft_list_t gpu_list;
        gpu_list.push_back(llama_default_buffer_type_offload(model, layer_gpu));
        model.buft_layer_list[i] = gpu_list;
    }
}
```

**Reason:** Don't create buffer lists when AMX disabled.

---

### 5. GPU Layer Allocation - Other Split Modes (Line ~5092-5098)
**Before:**
```cpp
for (int i = i_gpu_start; i < n_layer; ++i) {
    model.buft_layer[i] = {...};

    buft_list_t gpu_list;
    gpu_list.push_back(split_buft);
    model.buft_layer_list[i] = gpu_list;  // Always created!
}
```

**After:**
```cpp
for (int i = i_gpu_start; i < n_layer; ++i) {
    model.buft_layer[i] = {...};

    // GPU layers: buffer list is just GPU buffer (no AMX on GPU layers)
    // Only create if AMX enabled (though it won't be used for GPU layers)
    if (use_amx) {
        buft_list_t gpu_list;
        gpu_list.push_back(split_buft);
        model.buft_layer_list[i] = gpu_list;
    }
}
```

**Reason:** Same as above - conditional creation.

---

### 6. Output Layer - Split Mode (Line ~5075)
**Before:**
```cpp
model.buft_output = llama_default_buffer_type_cpu(need_host_buffer, false);
```

**After:**
```cpp
model.buft_output = llama_default_buffer_type_cpu(true);
```

**Reason:** Match main branch exactly (always `host_buffer=true`).

---

### 7. Output Layer - Other Modes (Line ~5107)
**Before:**
```cpp
model.buft_output = llama_default_buffer_type_cpu(need_host_buffer, false);
```

**After:**
```cpp
model.buft_output = llama_default_buffer_type_cpu(true);
```

**Reason:** Match main branch exactly (always `host_buffer=true`).

---

### 8. Buffer Type Counting (Line ~5124-5140)
**Before:**
```cpp
// Always counted buffer lists
for (int i = 0; i < n_layer; ++i) {
    for (auto buft : model.buft_layer_list[i]) {
        unique_bufts.insert(buft);
    }
}
```

**After:**
```cpp
// Only when AMX is enabled (buffer lists only exist in AMX mode)
if (use_amx && model.buft_layer_list.size() > 0) {
    for (int i = 0; i < n_layer; ++i) {
        for (auto buft : model.buft_layer_list[i]) {
            unique_bufts.insert(buft);
        }
    }
}
```

**Reason:** Don't try to count buffer lists when they don't exist (baseline mode).

---

### 9. Buffer Selection Lambda (Line ~5223)
**Before:**
```cpp
auto select_layer_buft = [&model, &ml](int layer_idx, const std::string & tensor_name) -> ggml_backend_buffer_type_t {
    const buft_list_t & buft_list = model.buft_layer_list[layer_idx];
    if (buft_list.empty()) {
        return model.buft_layer[layer_idx].buft;
    }
    // ... per-tensor selection logic
};
```

**After:**
```cpp
auto select_layer_buft = [&model, &ml, use_amx](int layer_idx, const std::string & tensor_name) -> ggml_backend_buffer_type_t {
    // When AMX disabled OR buffer lists not created: use legacy path (matches main branch)
    if (!use_amx || model.buft_layer_list.size() == 0 || layer_idx >= (int)model.buft_layer_list.size()) {
        return model.buft_layer[layer_idx].buft;  // BASELINE: always use legacy when !AMX
    }

    const buft_list_t & buft_list = model.buft_layer_list[layer_idx];
    if (buft_list.empty()) {
        return model.buft_layer[layer_idx].buft;
    }
    // ... per-tensor selection logic (AMX path)
};
```

**Reason:**
- Baseline mode: Always use legacy buffer type
- AMX mode: Use per-tensor selection from buffer list

---

## Expected Behavior After Fixes

### Without --amx Flag
- ✅ Input layer: Uses `host_buffer=true` (matches main)
- ✅ CPU layers: Use `host_buffer=true` (matches main)
- ✅ GPU layers: Use same buffer types as main
- ✅ Output layer: Uses `host_buffer=true` (matches main)
- ✅ Buffer lists: Not created
- ✅ Buffer counting: Same as main
- ✅ Tensor allocation: Uses legacy path (matches main)
- ✅ **RESULT:** Identical to main branch

### With --amx Flag (CPU-only)
- ✅ Input layer: Uses CPU buffer (no AMX for F32)
- ✅ CPU layers: Create buffer lists with AMX priority
- ✅ Tensor allocation: Per-tensor selection from buffer list
- ✅ AMX buffers: Created for compatible tensors (Q4_0, Q8_0, etc.)
- ✅ F32 tensors: Use CPU buffer (skip AMX)
- ✅ **RESULT:** AMX acceleration enabled

### With --amx Flag (with GPU)
- ✅ AMX disabled (CPU layers still exist for first N layers)
- ✅ CPU layers: Create buffer lists but no AMX in list (GPU present)
- ✅ GPU layers: Standard GPU buffers
- ✅ **RESULT:** No AMX, standard GPU operation

---

## Next Steps

1. Build the project
2. Test without --amx (must match main exactly)
3. Test with --amx, CPU-only (AMX should activate)
4. Test with --amx, GPU (AMX should not activate)
5. Document results

---

## Build Command

```bash
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AMX=ON

cmake --build build --target llama-cli -j 32
```

---

**Status:** Ready for build and test
