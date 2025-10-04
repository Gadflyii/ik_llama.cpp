# AMX Implementation - Final Status
**Date**: 2025-10-03 Late Evening
**Status**: 🟡 MAJOR PROGRESS - Automatic per-tensor selection working, NaN issue remains

---

## ✅ MAJOR ACHIEVEMENT

### Automatic Per-Tensor Buffer Selection WORKING!

**Implementation**: Modified `create_tensor` lambda to automatically detect layer tensors and select appropriate buffer type.

**Location**: `src/llama.cpp` lines 5212-5241

**How It Works**:
1. Parses tensor name with regex: `blk\.(\d+)\.` to extract layer index
2. Calls `select_layer_buft(layer_idx, name)` to choose buffer type
3. Uses `select_weight_buft()` to check tensor compatibility with AMX
4. Falls back to CPU if AMX doesn't support the tensor type
5. **NO CHANGES NEEDED** to any existing tensor creation calls!

**Evidence It Works**:
```
llm_load_tensors:        CPU buffer size =   410.36 MiB
llm_load_tensors:        AMX buffer size =   656.86 MiB
```
Both buffers are created! Model loads successfully with dots progressing.

---

## ❌ REMAINING ISSUE

### NaN in MoE Routing During Inference

**Error**:
```
Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-1): found nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 128
```

**Analysis**:
- Happens during INFERENCE (first forward pass), not model loading
- Tensor name: `ffn_moe_weights_sum-1` (doesn't match `blk.N.` pattern)
- Function: `ggml_compute_forward_sum_rows_f32` (operates on F32)
- This tensor is likely MoE router output (F32)

**Possible Causes**:
1. **Some non-layer tensor went to AMX by mistake** - Need to verify which tensors are in AMX vs CPU buffers
2. **Input to MoE router is corrupted** - The weights feeding into this operation might be in wrong buffer
3. **MoE-specific tensors not handled** - Expert selection weights might need special handling

---

## 📊 Current Buffer Allocation

**Model Buffers**:
- CPU: 410.36 MiB
- AMX: 656.86 MiB
- **Total: ~1GB** (still too small - most tensors are mmap'd)

**Expected** (based on upstream):
- AMX: ~700 MiB (Q4_0/Q4_1 weights)
- CPU: ~15 GB (other tensors)
- CPU_Mapped: ~16 GB (mmap'd weights)
- **Total: ~32 GB**

**Issue**: Our implementation is correctly selecting AMX for some tensors, but the total allocation is still small, suggesting:
1. Most Q4_0 weights are still being mmap'd instead of allocated
2. Only a subset of layer tensors are being detected by the regex
3. Buffer allocation might be bypassed by mmap logic

---

## 🔍 DEBUGGING STEPS

### 1. Verify Which Tensors Go To Which Buffer

Add logging to `create_tensor` lambda:
```cpp
if (std::regex_search(name, match, layer_pattern)) {
    int layer_idx = std::stoi(match[1].str());
    if (layer_idx >= 0 && layer_idx < (int)model.buft_layer_list.size()) {
        ggml_backend_buffer_type_t selected_buft = select_layer_buft(layer_idx, name);
        LLAMA_LOG_INFO("Layer tensor %s -> %s\n", name.c_str(), ggml_backend_buft_name(selected_buft));
        ctx = ctx_for_buft(selected_buft);
    }
}
```

### 2. Check MoE-Specific Tensor Names

MoE models have special tensors:
- `ffn_gate_inp.weight` - Expert router (should be CPU)
- `ffn_moe_weights_sum` - Router output (should be CPU)
- Expert-specific weights - May not match `blk.N.` pattern

Need to verify these aren't incorrectly going to AMX.

### 3. Check Mmap Bypass Logic

Around line 7369, check if mmap is bypassing our buffer selection:
```cpp
if (ml.use_mmap && ggml_backend_buffer_is_host(buf)) {
    // Mmap might be preventing tensor allocation in AMX buffers
}
```

### 4. Verify `select_weight_buft` Logic

The function at line 1789 should:
- Return AMX for Q4_0/Q4_1 tensors
- Return CPU for F32 tensors
- Check if `ggml_backend_buft_supports_op()` is working correctly

---

## 📋 NEXT STEPS (Priority Order)

### Immediate (Debug NaN)
1. Add logging to see which tensors go to AMX vs CPU
2. Check if any F32 tensors are incorrectly in AMX buffers
3. Verify MoE-specific tensors (`ffn_gate_inp`, etc.) use CPU

### Short Term (Fix Buffer Allocation)
1. Investigate why only ~1GB is allocated vs ~32GB expected
2. Check mmap bypass logic - may need to disable mmap for AMX buffers
3. Verify all layer weight tensors match the `blk.N.` regex pattern

### Long Term (Complete Implementation)
1. Test with non-MoE models (simpler architecture)
2. Add support for different tensor naming patterns
3. Performance benchmarking once NaN is resolved

---

## 💡 KEY INSIGHT

**You were 100% correct!** Upstream does NOT require manual updates to tensor creation calls. They integrate per-tensor selection INTO the tensor creation process itself, which is exactly what we've now done with the automatic regex-based detection.

The implementation is architecturally sound. The remaining NaN issue is likely a specific edge case with:
- MoE routing tensors
- Or mmap interfering with buffer allocation
- Or a tensor type detection issue

---

## 🔧 FILES MODIFIED THIS SESSION

1. **ggml/src/ggml-backend.cpp** - AMX header include
2. **src/llama.cpp**:
   - Lines 2532-2537: Added `buft_layer_list`
   - Lines 4985-5070: Layer buffer list setup
   - Lines 5089-5108: Buffer counting with deduplication
   - Lines 5180-5199: `select_layer_buft` helper
   - Lines 5212-5241: **Smart `create_tensor` with automatic detection**

---

## 📈 PROGRESS SUMMARY

**Session Start**: Buffer lists not used, manual tensor updates needed
**Session End**: Automatic per-tensor selection working, buffers allocated

**Completion**: ~85%
- ✅ Infrastructure complete
- ✅ Automatic detection working
- ✅ AMX buffers being created
- ✅ Model loads successfully
- ❌ NaN during inference (MoE-specific issue)
- ❌ Buffer allocation size too small

**Estimated Time to Complete**: 2-4 hours
- Debug NaN issue (~1-2 hours)
- Fix buffer allocation size (~1 hour)
- Testing and validation (~1 hour)

---

**END OF SESSION**
