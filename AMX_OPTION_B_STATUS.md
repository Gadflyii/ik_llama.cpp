# AMX Option B Implementation Status
**Date**: 2025-10-03 Late Evening
**Goal**: Implement proper per-tensor buffer selection (Option B - complete solution)

---

## ✅ COMPLETED

### 1. Buffer List Infrastructure
**Files Modified**: `src/llama.cpp`

**Added to Model Structure** (line 2532-2537):
```cpp
// Modern buffer list approach (for per-tensor selection)
// Each layer has a list of buffer types to try in priority order
std::vector<buft_list_t> buft_layer_list;

// Legacy single buffer type per layer (deprecated, kept for compatibility)
std::vector<layer_buft> buft_layer;
```

### 2. Layer Buffer Setup
**Location**: `src/llama.cpp` lines 4985-5070

**CPU Layers** (lines 4999-5007):
```cpp
for (int i = 0; i < i_gpu_start; ++i) {
    // Legacy: single buffer type (kept for compatibility)
    model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, false);

    // Modern: buffer list with fallback priority
    // Priority: AMX (if enabled) → CPU (always fallback)
    model.buft_layer_list[i] = make_cpu_buft_list(use_extra_bufts, no_host);
}
```

**GPU Layers** (lines 5039-5043, 5066-5069):
```cpp
// GPU layers: buffer list is just GPU buffer (no AMX on GPU layers)
buft_list_t gpu_list;
gpu_list.push_back(llama_default_buffer_type_offload(model, layer_gpu));
model.buft_layer_list[i] = gpu_list;
```

### 3. Buffer Counting for Contexts
**Location**: `src/llama.cpp` lines 5089-5108

**Implementation**:
- Counts legacy buft_layer for backward compatibility
- Counts unique buffer types from buft_layer_list
- Avoids duplicate context creation

### 4. Helper Functions
**Location**: `src/llama.cpp` lines 5180-5239

**select_layer_buft** (lines 5181-5199):
- Selects buffer type for a tensor from layer's buffer list
- Uses `select_weight_buft` to check compatibility
- Falls back to first in list or legacy buffer

**create_tensor_for_layer** (lines 5219-5239):
- Wraps tensor creation with automatic buffer selection
- Respects buffer type overrides
- Ready to use for layer weight tensors

---

## ⏳ IN PROGRESS

### Current Issue: AMX Contexts Created But Not Used

**Problem**:
- Buffer lists are populated with AMX buffer types
- Contexts are created for AMX buffers
- BUT: No tensors are actually allocated using the new `create_tensor_for_layer` function
- Result: AMX contexts are empty or have wrong tensors
- When `ggml_backend_alloc_ctx_tensors_from_buft` runs, it fails

**Evidence**:
```bash
# With --amx flag:
llama_model_load: error loading model: unable to allocate backend buffer

# Without --amx flag:
Works fine - baseline not broken
```

---

## 📋 NEXT STEPS

### Immediate Action Required

**Step 1: Update Layer Weight Tensor Creation**

Need to replace `create_tensor(ctx_for_layer(i), ...)` calls with `create_tensor_for_layer(i, ...)` for these critical tensors:

**Critical Weight Matrices** (need AMX):
- `attn_q.weight` - Attention query projection
- `attn_k.weight` - Attention key projection
- `attn_v.weight` - Attention value projection
- `attn_output.weight` - Attention output projection
- `ffn_gate.weight` / `ffn_gate_exps` - FFN gate (for MoE: per-expert)
- `ffn_up.weight` / `ffn_up_exps` - FFN up projection (for MoE: per-expert)
- `ffn_down.weight` / `ffn_down_exps` - FFN down projection (for MoE: per-expert)

**Non-Weight Tensors** (should use CPU, not AMX):
- `attn_norm.weight` - RMS norm (F32)
- `ffn_norm.weight` - RMS norm (F32)
- Any bias terms (F32)

**Locations to Update**:
1. Find layer loops: `for (int i = 0; i < n_layer; ++i)`
2. Locate weight tensor creation inside loops
3. Replace pattern:
   ```cpp
   // OLD:
   layer.attn_q = create_tensor(ctx_for_layer(i), tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head});

   // NEW:
   layer.attn_q = create_tensor_for_layer(i, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head});
   ```

**Estimated Scope**:
- ~10-20 architecture cases (LLAMA, QWEN, DEEPSEEK, etc.)
- ~6-8 weight tensors per architecture
- Total: ~100-150 tensor creation calls to update

**Systematic Approach**:
1. Start with ONE architecture (e.g., `LLM_ARCH_LLAMA`)
2. Update ALL weight matrices in that architecture
3. Test that architecture works with --amx
4. Repeat for other architectures

---

## 🔧 IMPLEMENTATION GUIDE

### Pattern to Follow

**For Each Architecture Block**:
```cpp
case LLM_ARCH_LLAMA:
    {
        // ... setup code ...

        for (int i = 0; i < n_layer; ++i) {
            // CONTEXT SELECTION - Keep for non-weight tensors
            ggml_context * ctx_layer = ctx_for_layer(i);
            ggml_context * ctx_split = ctx_for_layer_split(i);

            auto & layer = model.layers[i];

            // NORM TENSORS - Use OLD method (these are F32, use legacy context)
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

            // WEIGHT TENSORS - Use NEW method (these are Q4_0, use buffer list)
            layer.attn_q = create_tensor_for_layer(i, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head});
            layer.attn_k = create_tensor_for_layer(i, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_head_k * n_head_kv});
            layer.attn_v = create_tensor_for_layer(i, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_head_v * n_head_kv});
            layer.attn_output = create_tensor_for_layer(i, tn(LLM_TENSOR_ATTN_OUTPUT, "weight", i), {n_embd, n_embd});

            // FFN weights
            layer.ffn_gate = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
            layer.ffn_up = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff});
            layer.ffn_down = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        }
    }
    break;
```

### Special Case: MoE Architectures

For MoE (Mixture of Experts) like QWEN3MOE:
```cpp
case LLM_ARCH_QWEN3MOE:
    {
        for (int i = 0; i < n_layer; ++i) {
            // ... shared attention weights (same as above) ...

            // Per-expert FFN weights
            if (n_expert > 0) {
                layer.ffn_gate_exps = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_expert, n_expert});
                layer.ffn_up_exps = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_expert, n_expert});
                layer.ffn_down_exps = create_tensor_for_layer(i, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_expert, n_embd, n_expert});
            }
        }
    }
    break;
```

---

## 🧪 TESTING PLAN

### After Updating ONE Architecture

**Step 1: Build**
```bash
cd /home/ron/src/ik_llama.cpp/build
cmake --build . --target llama-cli -j 32
```

**Step 2: Test Without AMX (Baseline)**
```bash
numactl --cpunodebind=2 --membind=2 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 32 -n 10 -p "Test" | grep -i buffer
```

**Expected**: Should work, show CPU buffer allocation

**Step 3: Test With AMX**
```bash
numactl --cpunodebind=2 --membind=2 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 32 -n 10 -p "Test" --amx 2>&1 | tee /tmp/amx_test.log
```

**Expected**:
- Should NOT fail with "unable to allocate backend buffer"
- Should show multiple buffers (AMX + CPU)
- Should show buffer sizes adding up to ~32GB total
- Should generate valid tokens (no NaN)

**Step 4: Check Buffer Allocation**
```bash
grep "buffer size" /tmp/amx_test.log
```

**Expected Output** (similar to upstream):
```
llm_load_tensors:        AMX buffer size = ~700 MiB
llm_load_tensors:        CPU buffer size = ~15000 MiB
llm_load_tensors:    CPU_Mapped buffer size = ~16000 MiB
Total: ~32 GB
```

**Step 5: Check for NaN**
```bash
grep -i "nan\|Oops" /tmp/amx_test.log
```

**Expected**: NO NaN errors

---

## 📊 ARCHITECTURE PRIORITY

Update in this order:
1. **QWEN3MOE** - Current test model (highest priority)
2. **LLAMA** - Most common architecture
3. **DEEPSEEK2** - Complex MoE with MLA
4. **Others** - As needed

---

## 🔍 DEBUGGING

### If "unable to allocate backend buffer" Persists

**Check 1**: Verify buffer types are valid
```cpp
// Add logging to make_cpu_buft_list
LLAMA_LOG_INFO("Buffer list for layer %d: %zu types\n", i, model.buft_layer_list[i].size());
for (auto buft : model.buft_layer_list[i]) {
    LLAMA_LOG_INFO("  - %s\n", ggml_backend_buft_name(buft));
}
```

**Check 2**: Verify contexts are created
```cpp
// Add logging after ctx_map population
for (auto & it : ctx_map) {
    LLAMA_LOG_INFO("Context for buft %s: %p\n", ggml_backend_buft_name(it.first), it.second);
}
```

**Check 3**: Verify tensor allocation
```cpp
// In create_tensor_for_layer, add logging
LLAMA_LOG_INFO("Tensor %s: selected buft %s\n", name.c_str(), ggml_backend_buft_name(selected_buft));
```

### If NaN Still Occurs

- Check which tensors went to AMX vs CPU
- Verify F32 tensors (norms, biases) are NOT in AMX buffers
- Verify Q4_0/Q4_1 weight tensors ARE in AMX buffers
- Check that `select_weight_buft` is working correctly

---

## 📝 FILES MODIFIED SO FAR

1. **ggml/src/ggml-backend.cpp** - AMX header include moved to top
2. **src/llama.cpp**:
   - Line 2532-2537: Added `buft_layer_list` to model
   - Line 4985-5070: Layer buffer setup with lists
   - Line 5089-5108: Buffer counting with deduplication
   - Line 5180-5199: `select_layer_buft` helper
   - Line 5219-5239: `create_tensor_for_layer` function

---

## ⏭️ HANDOFF TO NEXT SESSION

**Current State**:
- ✅ Infrastructure complete and compiles
- ✅ Baseline (no --amx) still works
- ❌ With --amx flag: fails at buffer allocation
- ⏳ Need to update ~100-150 tensor creation calls

**Next Developer Should**:
1. Read this document
2. Start with QWEN3MOE architecture (it's the test model)
3. Find the `case LLM_ARCH_QWEN3MOE:` block (around line 5400-5800)
4. Update weight tensor creation to use `create_tensor_for_layer`
5. Test with the commands above
6. Repeat for other architectures

**Estimated Time**: 4-6 hours for complete implementation + testing

---

**END OF STATUS DOCUMENT**
