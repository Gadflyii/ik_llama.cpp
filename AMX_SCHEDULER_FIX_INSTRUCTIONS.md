# AMX+GPU Scheduler Fix - Step-by-Step Implementation Instructions

**Goal**: Fix backend scheduler to prevent copying weight tensors between backends
**Approach**: Modify scheduler logic to keep operations on the backend where their weights reside
**Expected Outcome**: Long prompts work with AMX+GPU hybrid mode (currently crash after prompt processing)

---

## Pre-Implementation Checklist

Before starting, confirm:
- [ ] Current status: Short prompts work (61.24 t/s), long prompts crash
- [ ] Last commit: fb91d1c5 "Fix AMX+GPU segfault by implementing cpy_tensor"
- [ ] Clean working directory (no uncommitted changes except docs)
- [ ] Backup location prepared: `/tmp/amx-scheduler-fix-YYYYMMDD-HHMMSS/`

---

## Phase 1: Add Diagnostic Logging (Research)

### Goal
Identify exactly which operation is triggering the weight tensor copy from AMX to CUDA.

### Step 1.1: Backup Current State

```bash
# Create timestamped backup directory
BACKUP_DIR="/tmp/amx-scheduler-fix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup files we'll modify
cp /home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp "$BACKUP_DIR/ggml-backend.cpp.backup"
cp /home/ron/src/ik_llama.cpp/ggml/src/ggml.c "$BACKUP_DIR/ggml.c.backup" 2>/dev/null || true

# Save git state
cd /home/ron/src/ik_llama.cpp
git status > "$BACKUP_DIR/git-status.txt"
git log -1 --format="%H %s %ai" > "$BACKUP_DIR/git-head.txt"
git diff > "$BACKUP_DIR/git-diff-before.txt"

echo "✅ Backup created in: $BACKUP_DIR"
echo "$BACKUP_DIR" > /tmp/amx-scheduler-fix-backup-location.txt
```

### Step 1.2: Add Logging to `ggml_backend_tensor_copy`

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`

**Location**: Function `ggml_backend_tensor_copy` (around line 356-377)

**Add before the copy logic**:

```cpp
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    // ===== NEW DIAGNOSTIC LOGGING =====
    #ifdef GGML_DEBUG_TENSOR_COPY
    const char * src_buf_name = src->buffer ? ggml_backend_buffer_name(src->buffer) : "NULL";
    const char * dst_buf_name = dst->buffer ? ggml_backend_buffer_name(dst->buffer) : "NULL";

    fprintf(stderr, "\n[TENSOR_COPY_DEBUG] Copying tensor:\n");
    fprintf(stderr, "  Name: %s\n", src->name ? src->name : "(unnamed)");
    fprintf(stderr, "  Src buffer: %s\n", src_buf_name);
    fprintf(stderr, "  Dst buffer: %s\n", dst_buf_name);
    fprintf(stderr, "  Size: %zu bytes\n", ggml_nbytes(src));
    fprintf(stderr, "  Op: %s\n", ggml_op_name(src->op));

    // Check if this is a weight tensor
    bool is_weight = (src->flags & GGML_TENSOR_FLAG_PARAM) ||
                     (strstr(src->name, ".weight") != NULL) ||
                     (strstr(src->name, ".bias") != NULL);

    if (is_weight) {
        fprintf(stderr, "  ⚠️  WARNING: This appears to be a WEIGHT tensor!\n");
        fprintf(stderr, "      Weights should not be copied during inference.\n");
    }
    #endif
    // ===== END DIAGNOSTIC LOGGING =====

    // ... existing copy logic ...
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        #ifdef GGML_DEBUG_TENSOR_COPY
        fprintf(stderr, "  → Using slow fallback path (get_tensor + set_tensor)\n");
        #endif

        #ifndef NDEBUG
        fprintf(stderr, "%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
        #endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}
```

### Step 1.3: Enable Debug Logging in CMake

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/CMakeLists.txt`

**Find**: `add_compile_definitions` or similar section

**Add**:
```cmake
# Add debug flag for tensor copy logging
target_compile_definitions(ggml PRIVATE GGML_DEBUG_TENSOR_COPY)
```

### Step 1.4: Rebuild with Debug Logging

```bash
cd /home/ron/src/ik_llama.cpp

# Clean rebuild to ensure debug flag takes effect
cmake --build build --target clean
cmake --build build -j 64

# Verify build succeeded
ls -lh build/bin/llama-cli
```

### Step 1.5: Run Test with Long Prompt

```bash
cd /home/ron/src/ik_llama.cpp

# Create test script
cat > /tmp/amx_scheduler_diag_test.sh << 'EOF'
#!/bin/bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 100 -ngl 15 --amx -fa \
  -p "The history of artificial intelligence began in the 1950s when Alan Turing proposed the Turing Test as a measure of machine intelligence. Early AI research focused on symbolic reasoning and expert systems. In the 1980s, neural networks gained prominence, leading to breakthroughs in machine learning. The 21st century saw the rise of deep learning, enabling AI to excel in tasks like image recognition, natural language processing, and game playing. Modern AI systems use transformer architectures and large language models trained on massive datasets. Today, AI is transforming industries from healthcare to transportation, raising important questions about ethics, privacy, and the future of human-AI collaboration. Researchers continue to push the boundaries of what machines can achieve."
EOF

chmod +x /tmp/amx_scheduler_diag_test.sh

# Run and capture all output
/tmp/amx_scheduler_diag_test.sh 2>&1 | tee "$BACKUP_DIR/diagnostic_run.log"

# Also save just the tensor copy debug info
grep -A 10 "TENSOR_COPY_DEBUG" "$BACKUP_DIR/diagnostic_run.log" > "$BACKUP_DIR/tensor_copies.log"
```

### Step 1.6: Analyze Results

**Look for**:
1. Any tensor copies from "AMX" to "CUDA" or "CUDA0"
2. Specifically look for `blk.0.attn_q.weight` or similar weight tensors
3. Note which operation (`Op:` field) is triggering the copy

**Expected findings**:
```
[TENSOR_COPY_DEBUG] Copying tensor:
  Name: blk.0.attn_q.weight
  Src buffer: AMX
  Dst buffer: CUDA0
  Size: 4718592 bytes
  Op: MUL_MAT (or MUL_MAT_ID)
  ⚠️  WARNING: This appears to be a WEIGHT tensor!
```

**Save findings**:
```bash
# Document what operation is causing the issue
cat >> "$BACKUP_DIR/analysis.txt" << EOF
=== Diagnostic Results ===
Date: $(date)

Tensor causing issue: [FILL IN]
Source buffer: [FILL IN]
Destination buffer: [FILL IN]
Operation type: [FILL IN]

Root cause: [Describe why this operation is trying to copy the weight]

Recommended fix approach: [A, B, or C from Phase 2]
EOF
```

---

## Phase 2: Implement Fix (Based on Phase 1 Findings)

### Approach A: Filter Weight Tensors from Copy Operations

**Use if**: Scheduler is indiscriminately copying all tensors including weights

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`

**Add helper function** (around line 350, before `ggml_backend_tensor_copy`):

```cpp
// Check if tensor is a model weight/parameter (should not be copied during inference)
static bool ggml_tensor_is_weight(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->name) {
        return false;
    }

    // Check tensor flags
    if (tensor->flags & GGML_TENSOR_FLAG_PARAM) {
        return true;
    }

    // Check name patterns
    const char * name = tensor->name;
    if (strstr(name, ".weight") != NULL ||
        strstr(name, ".bias") != NULL ||
        strstr(name, "token_embd") != NULL ||
        strstr(name, "output.weight") != NULL) {
        return true;
    }

    return false;
}
```

**Modify `ggml_backend_tensor_copy`**:

```cpp
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    // === NEW: Prevent weight tensor copies ===
    if (ggml_tensor_is_weight(src)) {
        fprintf(stderr, "ERROR: Attempted to copy weight tensor '%s' between backends\n", src->name);
        fprintf(stderr, "       Weights should remain on their allocated backend.\n");
        fprintf(stderr, "       This indicates a scheduler bug.\n");
        GGML_ASSERT(false && "Weight tensors should not be copied during inference");
        return;  // In release builds, just skip the copy
    }
    // === END NEW CODE ===

    // ... existing copy logic ...
}
```

### Approach B: Fix Operation Backend Assignment

**Use if**: Operations are being scheduled on wrong backend (where weights don't reside)

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp` or related scheduler file

**Find**: Backend assignment logic in scheduler (function that decides which backend runs each operation)

**Modify** to check weight locations:

```cpp
// Pseudo-code - actual implementation depends on scheduler structure
ggml_backend_t select_backend_for_op(const struct ggml_tensor * op) {
    // Check if operation uses model weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * src = op->src[i];
        if (src && ggml_tensor_is_weight(src)) {
            // Operation uses weights - must run on backend where weights reside
            ggml_backend_t weight_backend = get_backend_for_buffer(src->buffer);
            if (weight_backend) {
                return weight_backend;  // Use weight's backend
            }
        }
    }

    // No weights involved - use default backend selection
    return select_default_backend(op);
}
```

### Approach C: Add AMX-Aware Copy Path (Fallback)

**Use if**: Approaches A and B don't fully solve it, or as additional safety

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-cuda.cu`

**Modify CUDA buffer's `cpy_tensor`** (around line 626):

```cpp
static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // Existing CUDA-to-CUDA logic
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }

    // === NEW: Handle host/AMX buffers ===
    // For AMX buffers specifically, we cannot directly read VNNI format
    // Return false to let fallback handle it (which will fail with better error)
    if (!ggml_backend_buffer_is_host(src->buffer)) {
        // Source is not CUDA and not host - likely AMX or other special buffer
        fprintf(stderr, "WARNING: CUDA cpy_tensor cannot handle non-host source buffer: %s\n",
                ggml_backend_buffer_name(src->buffer));
        return false;
    }
    // === END NEW CODE ===

    return false;

    GGML_UNUSED(buffer);
}
```

---

## Phase 3: Test & Validate

### Test 1: Long Prompt (Primary Fix Validation)

```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 1000 -ngl 15 --amx -fa \
  -p "The history of artificial intelligence..." \
  2>&1 | tee "$BACKUP_DIR/test_long_prompt.log"
```

**Expected**: ✅ Completes successfully, generates 1000 tokens

### Test 2: Short Prompt (Regression Check)

```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 100 -ngl 15 --amx -fa \
  -p "Test" \
  2>&1 | tee "$BACKUP_DIR/test_short_prompt.log"
```

**Expected**: ✅ Still works (no regression)

### Test 3: CPU-Only Mode (Regression Check)

```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 100 --amx \
  -p "Test prompt for CPU-only" \
  2>&1 | tee "$BACKUP_DIR/test_cpu_only.log"
```

**Expected**: ✅ Works correctly

### Test 4: GPU-Only Mode (Regression Check)

```bash
./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 100 -ngl 99 \
  -p "Test prompt for GPU-only" \
  2>&1 | tee "$BACKUP_DIR/test_gpu_only.log"
```

**Expected**: ✅ Works correctly (AMX not used)

### Test 5: Very Long Prompt (Stress Test)

```bash
# Generate a 500+ token prompt
LONG_PROMPT="The history of artificial intelligence is a fascinating journey that spans multiple decades and encompasses numerous breakthroughs..." # (continue for 500+ tokens)

numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 500 -ngl 15 --amx -fa \
  -p "$LONG_PROMPT" \
  2>&1 | tee "$BACKUP_DIR/test_very_long_prompt.log"
```

**Expected**: ✅ Works correctly

---

## Phase 4: Performance Validation

### Benchmark: Prompt Processing

```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-bench \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -ngl 15 --amx -p 512 -n 0 \
  2>&1 | tee "$BACKUP_DIR/bench_prompt.log"
```

**Expected**: Prompt processing speed should be ~same as before fix

### Benchmark: Token Generation

```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-bench \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -ngl 15 --amx -p 512 -n 128 \
  2>&1 | tee "$BACKUP_DIR/bench_generation.log"
```

**Expected**: ~61 t/s (same as before fix)

---

## Phase 5: Commit & Document

### Step 5.1: Remove Debug Logging

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/CMakeLists.txt`

**Remove**: `target_compile_definitions(ggml PRIVATE GGML_DEBUG_TENSOR_COPY)`

**File**: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`

**Remove or comment out**: All `#ifdef GGML_DEBUG_TENSOR_COPY` blocks

### Step 5.2: Final Rebuild & Test

```bash
cmake --build build --target clean
cmake --build build -j 64

# Quick validation test
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 100 -ngl 15 --amx -fa \
  -p "The history of artificial intelligence..." | head -50
```

### Step 5.3: Commit Changes

```bash
cd /home/ron/src/ik_llama.cpp

# Check what changed
git status
git diff

# Stage changes
git add ggml/src/ggml-backend.cpp
git add ggml/src/ggml-cuda.cu  # If modified

# Commit with detailed message
git commit -m "$(cat <<'EOF'
Fix AMX+GPU hybrid mode crash with long prompts

Fixed backend scheduler incorrectly copying weight tensors from AMX to CUDA
during long prompt processing. Weights are read-only model parameters that
should remain on their allocated backend.

**Root cause**:
- Scheduler was attempting to copy blk.0.attn_q.weight from AMX to CUDA
- AMX weights are in VNNI format which cannot be read directly
- Fallback path called get_tensor which asserted on AMX-formatted data

**Solution implemented**: [Approach A/B/C - describe which one]
- [Describe specific changes made]
- Prevents weight tensor copies during inference
- Operations now execute on backend where weights reside

**Testing**:
- ✅ Long prompts (200+ tokens) now work correctly
- ✅ Token generation speed maintained (~61 t/s)
- ✅ No regressions in CPU-only or GPU-only modes
- ✅ Short prompts still work as before

**Performance**:
- Prompt processing: [X ms/token]
- Token generation: [Y tokens/s]
- No performance degradation from fix

**Files modified**:
- ggml/src/ggml-backend.cpp: [describe changes]
- [other files if applicable]

Fixes: Long prompt crash in AMX+GPU hybrid mode
Related: AMX_GPU_HYBRID_SCHEDULER_FIX.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# Verify commit
git log -1 --stat
```

### Step 5.4: Update Documentation

**File**: `/home/ron/src/ik_llama.cpp/AMX_SESSION_CURRENT.md`

Update with:
- Current status: ✅ AMX+GPU hybrid mode fully working
- What was fixed this session
- Performance results
- Next priorities (if any)

**File**: `/home/ron/src/ik_llama.cpp/AMX_GPU_HYBRID_SCHEDULER_FIX.md`

Update "Current Status" section with results.

---

## Success Criteria

Mark complete when ALL of the following are true:

- [ ] Long prompts (200+ tokens) work with AMX+GPU hybrid mode
- [ ] No crashes during generation
- [ ] Performance maintained (~61 t/s)
- [ ] No weight tensors copied during inference (verified in logs)
- [ ] All 5 test cases pass
- [ ] No regressions in other modes
- [ ] Code committed with comprehensive message
- [ ] Documentation updated
- [ ] Backup preserved with all test results

---

## Troubleshooting

### Issue: Still crashing after Approach A

**Try**: Approach B - fix backend assignment instead of filtering copies

### Issue: Performance degraded

**Check**: Are activations being copied correctly?
**Solution**: Make sure fix only prevents WEIGHT copies, not activation copies

### Issue: Build fails

**Check**: Syntax errors in added code
**Solution**: Review code carefully, check matching braces/semicolons

### Issue: Different tensor being copied

**Solution**: Add that tensor name pattern to `ggml_tensor_is_weight()` function

---

## Backup Location

All backups and test results stored in:
```
/tmp/amx-scheduler-fix-YYYYMMDD-HHMMSS/
```

Location saved to:
```
/tmp/amx-scheduler-fix-backup-location.txt
```

---

**Document Version**: 1.0
**Created**: 2025-10-06
**Last Updated**: 2025-10-06 14:45
**Status**: Ready for implementation
