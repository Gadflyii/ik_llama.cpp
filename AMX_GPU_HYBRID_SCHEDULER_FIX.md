# AMX+GPU Hybrid Mode: Scheduler Weight Copy Bug

**Date**: 2025-10-06
**Status**: 🔴 BUG IDENTIFIED - Implementation Plan Ready
**Priority**: HIGH - Blocks AMX+GPU hybrid mode with long prompts

---

## Executive Summary

**Problem**: AMX+GPU hybrid mode crashes during long prompt processing (200+ tokens) when scheduler attempts to copy weight tensors from AMX buffers to CUDA.

**Root Cause**: Backend scheduler incorrectly tries to copy `blk.0.attn_q.weight` (and likely other weight tensors) from AMX buffer to CUDA buffer during inference. AMX weights are in VNNI format which cannot be read directly, causing segfault when fallback path calls `get_tensor`.

**Impact**:
- ✅ Short prompts (<10 tokens) work fine: 61.24 t/s with 1000 token generation
- ❌ Long prompts (200+ tokens) crash during generation
- ❌ Makes AMX+GPU hybrid mode unusable for production

**Solution**: Modify backend scheduler to never copy weight tensors between backends. Weights are read-only and should remain on their allocated backend. Only intermediate activations should move between CPU/GPU.

---

## Bug Details

### Crash Location
```
Thread 1 "llama-cli" received signal SIGABRT, Aborted.
#5  ggml_backend_amx_buffer_get_tensor() - Line 144: GGML_ASSERT failed
#6  ggml_backend_tensor_copy()
#7  ggml_backend_sched_graph_compute_async()
#8  llama_decode()
```

### Failed Tensor Copy
```
ERROR: Attempt to read AMX-formatted tensor:
  tensor name: blk.0.attn_q.weight
  tensor type: 2 (Q4_0 with AMX VNNI format)
  tensor size: 4718592 bytes
This indicates a buffer is trying to copy from AMX without proper cpy_tensor support
```

### Why It Fails

1. **AMX buffer has VNNI-formatted weights**
   - Q4_0 weights repacked in VNNI layout for AMX acceleration
   - Cannot be read directly - data is in special format
   - `get_tensor` function intentionally asserts on AMX-formatted data

2. **CUDA buffer's `cpy_tensor` doesn't handle AMX sources**
   ```cpp
   // ggml-cuda.cu:626
   static bool ggml_backend_cuda_buffer_cpy_tensor(...) {
       if (ggml_backend_buffer_is_cuda(src->buffer)) {
           // Handle CUDA-to-CUDA copies
           return true;
       }
       return false;  // ← Returns false for AMX sources
   }
   ```

3. **Fallback path tries to read from AMX**
   ```cpp
   // ggml-backend.cpp:360-367
   if (!ggml_backend_buffer_copy_tensor(src, dst)) {
       // Fallback: read from src using get_tensor
       ggml_backend_tensor_get(src, data, 0, nbytes);  // ← CRASHES
   }
   ```

### Why Long Prompts Trigger It

- **Short prompts**: Fit in single batch, minimal cross-backend interaction
- **Long prompts**: Require multiple decode passes, increasing scheduler complexity
- **Hypothesis**: Split computation or batching logic triggers weight copy for some operation that spans CPU/GPU boundary

---

## Investigation Findings

### How Upstream Handles It

**Upstream (llama.cpp)**: Also has same buffer implementations!
- AMX buffer: `cpy_tensor = nullptr`, `get_tensor = nullptr`
- CUDA buffer: Only handles CUDA-to-CUDA copies
- **Key difference**: Device registry + smarter buffer priority

**Buffer Priority Order**:
- **Upstream**: ACCEL → GPU host → CPU extra (AMX) → CPU
- **Fork**: AMX → CUDA_Host → CPU

**Why upstream works**: GPU host buffer comes FIRST, so weights that might be accessed by GPU operations go there instead of AMX, avoiding the copy issue entirely.

**Why we can't just reorder**: Fork lacks device registry infrastructure. Simple reordering causes all weights to try allocating in GPU host buffer → OOM (out of memory).

### Three Solution Options Analyzed

| Option | Complexity | Performance | Verdict |
|--------|-----------|-------------|---------|
| 1. Port device registry | Very High | Best (long-term) | ❌ Too invasive |
| 2. Decompress VNNI in `get_tensor` | Medium-High | **Worst** | ❌ Defeats AMX purpose |
| 3. **Fix scheduler to not copy weights** | **Low-Medium** | **Best** | ✅ **CHOSEN** |

**Why Option 3 is correct**:
- **Architectural correctness**: Weights should NEVER be copied between backends during inference
- **Performance**: Zero overhead - prevents unnecessary operation
- **Minimal changes**: Focused fix in scheduler/split logic
- **Root cause**: Scheduler bug incorrectly marking weights as needing cross-backend transfer

---

## Implementation Plan

### Phase 1: Identify Weight Copy Trigger (Research)

**Goal**: Find exactly which operation is requesting the weight copy

**Tasks**:
1. Add detailed logging to scheduler's tensor copy decision logic
2. Run test with long prompt and capture full operation graph
3. Identify which graph node/operation triggers `blk.0.attn_q.weight` copy
4. Determine if it's:
   - Split-device computation (operation spans CPU+GPU)
   - Incorrect buffer routing decision
   - MUL_MAT_ID (expert selection) touching both backends
   - Batch processing logic error

**Files to examine**:
```
/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp
  - ggml_backend_tensor_copy()  (line 356-377)
  - ggml_backend_sched_graph_compute_async()
  - Backend split logic

/home/ron/src/ik_llama.cpp/ggml/src/ggml.c
  - Graph execution
  - Operation dependencies
```

**Expected output**:
```
DEBUG: Operation [ggml_op] requires tensor 'blk.0.attn_q.weight'
DEBUG: Tensor buffer: AMX, Operation backend: CUDA
DEBUG: Scheduling copy from AMX to CUDA  ← THIS IS THE BUG
```

### Phase 2: Implement Fix (Code Changes)

**Goal**: Prevent scheduler from copying weight tensors

**Approach A: Filter weight tensors from copy operations**
```cpp
// In ggml_backend_sched_graph_compute_async() or similar
bool should_copy_tensor(const ggml_tensor * tensor) {
    // Never copy weight tensors - they're read-only and backend-resident
    if (tensor->flags & GGML_TENSOR_FLAG_WEIGHT) {
        return false;
    }

    // Never copy tensors that are part of model parameters
    if (is_model_tensor(tensor)) {
        return false;
    }

    // Only copy intermediate activations
    return true;
}
```

**Approach B: Fix operation scheduling to use correct backend**
```cpp
// Ensure operations use the backend where their weights reside
ggml_backend_t select_op_backend(const ggml_op * op) {
    // If operation uses model weights, schedule on weight's backend
    for (each input tensor in op) {
        if (is_weight_tensor(input)) {
            return get_backend_for_buffer(input->buffer);
        }
    }

    // Otherwise, use configured backend split
    return get_backend_for_layer(op->layer_id);
}
```

**Approach C: Add AMX→CUDA copy support (fallback)**
```cpp
// Only if Approach A/B don't work
// In CUDA buffer's cpy_tensor:
static bool ggml_backend_cuda_buffer_cpy_tensor(...) {
    // Existing CUDA-to-CUDA logic
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ...
    }

    // NEW: Handle host buffers (including AMX-incompatible)
    if (ggml_backend_buffer_is_host(src->buffer) || is_amx_buffer(src->buffer)) {
        // For AMX buffers, this will fail and fall back to slower path
        // But at least won't crash - will use get_tensor on host-visible data
        return false;  // Let fallback handle it
    }

    return false;
}
```

**Files to modify**:
```
/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp
  - Backend scheduler logic
  - Tensor copy decision making
  - Operation→backend assignment

Possibly:
/home/ron/src/ik_llama.cpp/ggml/src/ggml-cuda.cu
  - CUDA buffer cpy_tensor (if Approach C needed)
```

### Phase 3: Testing & Validation

**Test Cases**:

1. **Short prompt (baseline)**:
   ```bash
   numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
     -m /path/to/model.gguf \
     -t 64 -b 2048 -c 4096 -n 100 -ngl 15 --amx -fa \
     -p "Test"
   ```
   Expected: ✅ Works (already does)

2. **Long prompt (200 tokens)**:
   ```bash
   numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
     -m /path/to/model.gguf \
     -t 64 -b 2048 -c 4096 -n 1000 -ngl 15 --amx -fa \
     -p "The history of artificial intelligence..." (200+ tokens)
   ```
   Expected: ✅ Should work after fix

3. **Very long prompt (500+ tokens)**:
   Expected: ✅ Should work

4. **CPU-only mode (verify no regression)**:
   ```bash
   ./build/bin/llama-cli -m /path/to/model.gguf -t 64 -n 100 --amx -p "Test"
   ```
   Expected: ✅ Still works

5. **GPU-only mode (verify no regression)**:
   ```bash
   ./build/bin/llama-cli -m /path/to/model.gguf -t 64 -n 100 -ngl 99 -p "Test"
   ```
   Expected: ✅ Still works

**Success Criteria**:
- [ ] No crashes with long prompts
- [ ] No weight tensors copied during inference
- [ ] Performance maintained (61+ t/s)
- [ ] All test cases pass
- [ ] No regressions in CPU-only or GPU-only modes

---

## Current Status

### Completed (This Session)
1. ✅ Fixed NULL `cpy_tensor` crash - implemented basic AMX buffer cpy_tensor
2. ✅ Added debug logging to identify which tensor is being copied
3. ✅ Investigated upstream, ktransformers, Intel Extension for PyTorch approaches
4. ✅ Analyzed all three solution options
5. ✅ Determined Option 3 (fix scheduler) is the correct approach
6. ✅ Identified that buffer reordering alone won't work without device registry
7. ✅ Created comprehensive implementation plan

### In Progress
- 🟡 Phase 1: Identify weight copy trigger

### Blocked/TODO
- ⏳ Phase 2: Implement scheduler fix
- ⏳ Phase 3: Testing & validation

---

## Technical Notes

### Why Weights Should Never Be Copied

1. **Read-only**: Model weights don't change during inference
2. **Backend-resident**: Allocated once at load time on appropriate backend
3. **Performance**: Copying 16GB of weights is expensive and pointless
4. **Architecture**: Only activations flow between backends, not parameters

### Why Fork's Buffer Priority Causes Issues

**Fork's approach**:
- Prioritize AMX first for maximum performance
- Works great for CPU-only mode
- Causes issues when GPU is involved because:
  - Weights land in AMX buffers (good for CPU)
  - But some operations might touch GPU
  - Scheduler incorrectly tries to copy weights to GPU (bad)

**Upstream's approach**:
- Prioritize GPU host buffer when GPU present
- Weights land in GPU-accessible host memory
- AMX reserved for CPU-only scenarios
- Device registry intelligently routes operations

**Ideal solution** (long-term):
- Device registry + capability queries
- Per-tensor intelligent buffer selection
- Weights go to optimal buffer based on which backends will access them
- Fork doesn't have this infrastructure yet

### AMX Buffer Architecture

```
AMX Buffer:
  - Stores Q4_0/Q4_1 weights in VNNI format
  - VNNI = Vector Neural Network Instructions format
  - Layout optimized for Intel AMX tile operations
  - Cannot be read directly - requires decompression
  - get_tensor = nullptr (intentionally)
  - cpy_tensor = handles host→AMX only

CUDA Buffer:
  - Stores tensors on GPU device memory
  - cpy_tensor = handles CUDA→CUDA only
  - Cannot directly copy from AMX (VNNI format incompatible)

CUDA_Host Buffer:
  - Pinned host memory accessible by both CPU and GPU
  - Used for efficient CPU↔GPU transfers
  - Standard format (not VNNI)
  - Can be read/written by both backends
```

---

## Files Modified This Session

### 1. `/home/ron/src/ik_llama.cpp/ggml/src/ggml-cpu/amx/amx.cpp`

**Lines 133-149**: Added debug logging and improved `get_tensor`:
```cpp
static void GGML_CALL ggml_backend_amx_buffer_get_tensor(...) {
    if (qtype_has_amx_kernels(tensor->type)) {
        fprintf(stderr, "\n\nERROR: Attempt to read AMX-formatted tensor:\n");
        fprintf(stderr, "  tensor name: %s\n", tensor->name);
        fprintf(stderr, "  tensor type: %d\n", tensor->type);
        // ... detailed error message
        GGML_ASSERT(false && "Cannot directly read AMX-formatted tensor data");
    }
    memcpy(data, (const char *)tensor->data + offset, size);
}
```

**Lines 151-196**: Implemented `cpy_tensor` for AMX buffers:
```cpp
static bool GGML_CALL ggml_backend_amx_buffer_cpy_tensor(...) {
    // Handle copy FROM host buffer TO AMX buffer
    if (ggml_backend_buffer_is_host(src->buffer)) {
        // Includes NUMA mirror buffer support
        if (qtype_has_amx_kernels(dst->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }

    // Handle copy FROM AMX buffer TO host buffer
    if (ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ASSERT(!qtype_has_amx_kernels(src->type) && "Cannot copy AMX-formatted tensor to host");
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    // AMX-to-AMX or AMX-to-other-device copies not supported
    return false;
}
```

**Line 196**: Updated buffer interface:
```cpp
static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    ...
    /* .get_tensor      = */ ggml_backend_amx_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_amx_buffer_cpy_tensor,  // ← Was nullptr
    ...
};
```

### 2. Committed Changes

```bash
commit fb91d1c5
Author: Ron Henderson <ronhendersontx@gmail.com>
Date:   Sun Oct 6 13:56:59 2025 -0500

    Fix AMX+GPU segfault by implementing cpy_tensor for AMX buffers

    Fixed NULL function pointer crash during GPU+AMX hybrid inference by
    implementing ggml_backend_amx_buffer_cpy_tensor and
    ggml_backend_amx_buffer_get_tensor functions.
```

---

## References

### Upstream Implementation
- **Repo**: `/home/ron/src/llama.cpp`
- **Branch**: main (includes AMX support)
- **Key commit**: 121a130b8 - "feat(amx): add --amx toggle; prefer CPU 'extra' with GPU host+mmap when enabled"
- **Buffer priority**: ACCEL → GPU host → CPU extra (AMX) → CPU
- **Uses**: Device registry for intelligent buffer selection

### Related Issues
- Upstream issue #XXXX: AMX+GPU hybrid mode
- Fork issue: AMX GPU hybrid investigation (docs/amx_gpu_hybrid_investigation.md)

### Documentation
- AMX_SESSION_CURRENT.md - Previous session (buffer list architecture)
- AMX_IMPLEMENTATIONS_COMPREHENSIVE_COMPARISON.md - AMX comparison study
- AMX_BUFFER_SYSTEM_PORT.md - Buffer system architecture
- This file: AMX_GPU_HYBRID_SCHEDULER_FIX.md

---

## Next Session Instructions

### Step 1: Create Backup
```bash
BACKUP_DIR="/tmp/amx-scheduler-fix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical files
cp /home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp "$BACKUP_DIR/"
cp /home/ron/src/ik_llama.cpp/ggml/src/ggml.c "$BACKUP_DIR/"
cp /home/ron/src/ik_llama.cpp/ggml/src/ggml-cuda.cu "$BACKUP_DIR/" 2>/dev/null || true

# Save current HEAD
cd /home/ron/src/ik_llama.cpp
git log -1 --format="%H %s" > "$BACKUP_DIR/git-head.txt"
git diff > "$BACKUP_DIR/git-diff.txt"

echo "Backup created in: $BACKUP_DIR"
```

### Step 2: Add Diagnostic Logging
See implementation plan Phase 1 above.

### Step 3: Run Test & Capture Logs
```bash
cd /home/ron/src/ik_llama.cpp

# Rebuild with debug logging
cmake --build build -j 64

# Run test and capture full output
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 100 -ngl 15 --amx -fa \
  -p "The history of artificial intelligence..." \
  2>&1 | tee /tmp/amx_scheduler_debug.log
```

### Step 4: Analyze & Implement Fix
Based on diagnostic output, implement appropriate fix from Phase 2.

### Step 5: Test & Validate
Run all test cases from Phase 3.

---

**Last Updated**: 2025-10-06 14:30
**Next Priority**: Add diagnostic logging to identify weight copy trigger
**Blocking Issue**: Scheduler copying weight tensors between backends (architectural bug)
**Estimated Effort**: 4-8 hours (2-3 hours research, 2-3 hours implementation, 2 hours testing)
