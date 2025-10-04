# AMX Implementation - Issue Tracker
**Last Updated**: 2025-10-03 23:59

---

## 🔴 BLOCKING ISSUES

### Issue #1: Tensor Loading Uses Single Buffer Type (ACTIVE)
**Priority**: P0 - BLOCKING
**Status**: 🟡 Infrastructure Complete, Tensor Updates Needed
**Component**: Buffer allocation system

**Problem**:
- `model.buft_layer[i]` is single `ggml_backend_buffer_type_t`
- All tensors in layer use same buffer type
- No per-tensor selection based on operation
- F32 tensors forced into AMX buffer → NaN

**Evidence**:
```
# Current behavior:
model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, use_amx_buffer);
→ If use_amx_buffer=true, ALL tensors use AMX
→ F32 tensors in AMX → unsupported operations → NaN
```

**Solution**:
- Option A: Use first buffer type from list (quick fix)
- Option B: Per-tensor buffer selection (proper fix)

**Files**:
- `src/llama.cpp:4938` - Layer buffer setup
- `src/llama.cpp:5000-8000` - Tensor loading (if Option B)

**Progress Update** (2025-10-03 Late Evening):
✅ **COMPLETED**:
- Added `buft_layer_list` to model structure
- Implemented `make_cpu_buft_list` and `select_weight_buft`
- Created `create_tensor_for_layer` helper function
- Buffer list infrastructure fully operational
- Build succeeds, baseline (no --amx) works

⏳ **IN PROGRESS**:
- Need to update ~100-150 tensor creation calls to use `create_tensor_for_layer`
- Focus on weight matrices (attn_q, attn_k, ffn_gate, etc.)

**Next Actions**:
1. Update QWEN3MOE architecture weight tensors (~30 calls)
2. Test with --amx flag
3. Verify 3 buffers created (~32GB)
4. Repeat for other architectures

**See**: AMX_OPTION_B_STATUS.md for detailed implementation guide

---

## 🟡 HIGH PRIORITY ISSUES

### Issue #2: Buffer Count Mismatch (ACTIVE)
**Priority**: P1 - High
**Status**: 🟡 Investigating
**Component**: Buffer allocation

**Problem**:
- Upstream creates 3 buffers (~32GB total)
- Our fork creates 2 buffers (~1GB total)
- Missing buffer types in allocation

**Evidence**:
```
Upstream with --no-host:
- CPU_REPACK: 14904 MB
- AMX: 729 MB
- CPU_Mapped: 16217 MB
Total: ~32 GB

Our fork with --amx:
- CPU: 410 MB
- AMX: 656 MB
Total: ~1 GB
```

**Root Cause**:
- Related to Issue #1
- Single buffer type per layer prevents multiple buffers
- Most tensors fall back to mmap (not counted in buffer size)

**Solution**:
- Same as Issue #1 - implement buffer lists
- Will automatically create proper buffers

---

## 🟢 RESOLVED ISSUES

### Issue #3: F32 Tensors in AMX Buffers ✅ PARTIALLY FIXED
**Priority**: P0 - BLOCKING
**Status**: 🟢 Partially Resolved
**Component**: Buffer allocation

**Problem**:
- `buft_input` (embeddings) used AMX buffer type
- `buft_output` (output layer) used AMX buffer type
- F32 operations in AMX buffer → NaN

**Evidence**:
```
BEFORE FIX:
Input first 8 values: nan nan nan nan nan nan nan nan

AFTER FIX:
Input first 8 values: 0.000011 0.000130 0.000172 -0.000004
```

**Solution Implemented**:
```cpp
// Line 4934, 4978, 5002 in src/llama.cpp
model.buft_input = llama_default_buffer_type_cpu(need_host_buffer, false /* no AMX */);
model.buft_output = llama_default_buffer_type_cpu(need_host_buffer, false /* no AMX */);
```

**Status**: Fixed for input/output tensors, but layer tensors still affected by Issue #1

---

### Issue #4: Multi-threading Barrier No-op ✅ FIXED
**Priority**: P0 - BLOCKING
**Status**: 🟢 Resolved
**Component**: Multi-threading

**Problem**:
- `ggml_barrier()` was empty no-op function
- 32 threads had no synchronization
- Race conditions in AMX operations

**Solution**:
- Implemented proper atomic-based barrier
- Uses `n_barrier` and `n_barrier_passed` atomics
- Added CPU pause in spin loop

**File**: `ggml/src/ggml.c:4592-4622`

---

### Issue #5: Backend Association Missing ✅ FIXED
**Priority**: P0 - BLOCKING
**Status**: 🟢 Resolved
**Component**: Backend system

**Problem**:
- AMX buffer type had no `is_host` function
- `ggml_backend_synchronize()` crashed (SEGFAULT)
- Backend couldn't associate AMX buffers with CPU

**Solution**:
- Added `ggml_backend_amx_buffer_type_is_host()` returning `true`
- AMX buffers now associated with CPU backend

**File**: `ggml/src/ggml-cpu/amx/amx.cpp:277-280`

---

### Issue #6: Tensor Type Filtering Incorrect ✅ FIXED
**Priority**: P1 - High
**Status**: 🟢 Resolved
**Component**: Tensor traits

**Problem**:
- All tensors in AMX buffers got AMX traits (including F32)
- AMX kernels tried to process F32 tensors
- "Unsupported quantized data type: 0" errors

**Solution**:
- Added type check in `init_tensor`: only Q4_0/Q4_1 get AMX traits
- Added type check in `compute_forward`: only Q4_0/Q4_1 use AMX kernels

**File**: `ggml/src/ggml-cpu/amx/amx.cpp:67-69, 33`

---

### Issue #7: Struct Layout Mismatch ✅ FIXED
**Priority**: P0 - BLOCKING
**Status**: 🟢 Resolved
**Component**: Type definitions

**Problem**:
- `ggml.c` uses `struct ggml_compute_params` with `shared*` pointer
- `ggml-cpu-impl.h` defines same struct with `threadpool*` pointer
- Memory corruption from wrong pointer offset

**Solution**:
- Added correct struct definition to `common.h` with `shared*`
- Added include guards to prevent redefinition
- Fixed barrier call: `ggml_barrier((struct ggml_threadpool *)params->shared)`

**Files**:
- `ggml/src/ggml-cpu/amx/common.h:7-15`
- `ggml/src/ggml-cpu/ggml-cpu-impl.h:18-30`
- `ggml/src/ggml-cpu/amx/mmq.cpp:2446`

---

### Issue #8: Missing Header Files ✅ FIXED
**Priority**: P2 - Medium
**Status**: 🟢 Resolved
**Component**: Build system

**Problem**:
- `mmq.cpp` missing `simd-mappings.h`, `quants.h`
- Wrong function definitions or macros

**Solution**:
- Added `#include "../simd-mappings.h"`
- Added `#include "../quants.h"`
- Used correct relative paths

**File**: `ggml/src/ggml-cpu/amx/mmq.cpp:9-12`

---

## 📋 BACKLOG (Future Work)

### Issue #9: Performance Not Measured
**Priority**: P3 - Low
**Status**: ⏸️ Blocked on Issue #1
**Component**: Performance

**Description**:
- Need to benchmark AMX vs non-AMX
- Need to measure prompt processing speedup
- Need to measure token generation speedup

**Blocked By**: Issue #1 (need working AMX first)

---

### Issue #10: GPU Hybrid Mode Untested
**Priority**: P3 - Low
**Status**: ⏸️ Blocked on Issue #1
**Component**: GPU integration

**Description**:
- Need to test `-ngl 20` (some layers on GPU)
- Verify CPU layers use AMX correctly
- Verify GPU<->CPU transfers work

**Blocked By**: Issue #1 (need working AMX first)

---

### Issue #11: NUMA Optimization Missing
**Priority**: P4 - Optional
**Status**: ⏸️ Future work
**Component**: NUMA support

**Description**:
- Upstream has NUMA mirror buffer support
- Replicates AMX buffers across NUMA nodes
- Improves multi-socket performance

**Complexity**: High (requires porting upstream NUMA code)

---

## Issue Summary

**Total**: 11 issues tracked
- 🔴 Blocking: 1 active
- 🟡 High Priority: 1 active
- 🟢 Resolved: 6 fixed
- 📋 Backlog: 3 future

**Current Focus**: Issue #1 (tensor loading buffer selection)

**Next Session Goal**: Resolve Issue #1 and Issue #2

---

## Issue History

**2025-10-03 Evening**:
- Created issue tracker
- Resolved Issues #3, #4, #5, #6, #7, #8
- Identified Issue #1 as root cause of remaining NaN
- Identified Issue #2 as symptom of Issue #1

**2025-10-03 Late Evening**:
- Implemented buffer list infrastructure
- Partially resolved Issue #3 (input/output tensors)
- Issue #1 remains blocking (layer tensors)
- Ready for next session to resolve Issue #1

---

**END OF ISSUE TRACKER**
