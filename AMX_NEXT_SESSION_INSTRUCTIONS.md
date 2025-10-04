# AMX Implementation - Next Session Instructions
**Created**: 2025-10-03 23:58
**For**: Next continuation session

---

## Session Startup Instructions

### 1. Read These Documents First
```
/home/ron/src/ik_llama.cpp/AMX_SESSION_CURRENT.md
/home/ron/src/ik_llama.cpp/AMX_BUFFER_SYSTEM_PORT.md
```

### 2. Current State Summary

**✅ What's Working**:
- Extra buffer types infrastructure in ggml-backend.cpp
- AMX registered as first extra buffer type
- Buffer list creation functions (`make_cpu_buft_list`, `select_weight_buft`)
- `buft_input` and `buft_output` fixed to NOT use AMX (prevents F32 NaN)

**❌ What's Broken**:
- `buft_layer` still uses single buffer type (not buffer list)
- All layer tensors try to use same buffer type
- No per-tensor buffer selection during allocation

**🎯 Root Cause**:
- F32 tensors were being allocated in AMX buffers
- AMX doesn't support F32 operations → NaN
- Need buffer list with fallback: AMX → CPU

---

## Immediate Actions (First 30 Minutes)

### Step 1: Test Current State

**Build**:
```bash
cd /home/ron/src/ik_llama.cpp/build
cmake --build . --target llama-cli -j 32
```

**Test**:
```bash
cd /home/ron/src/ik_llama.cpp

numactl --cpunodebind=2 --membind=2 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 32 -n 10 -p "Test" --amx 2>&1 | tee /tmp/amx_current_state.log
```

**Check**:
```bash
# Look for buffer allocation
grep "buffer size\|buffer type" /tmp/amx_current_state.log

# Look for NaN errors
grep -i "nan\|Oops" /tmp/amx_current_state.log

# Expected: Still get NaN, but later in execution (not in first matmul inputs)
# Expected: See AMX buffer allocated
```

**Purpose**: Verify that `buft_input` fix prevents immediate NaN in layer norm

### Step 2: Analyze Buffer Allocation

**Question to Answer**:
- How many buffers are created?
- What are their sizes?
- Does it match upstream (should be ~32GB total, not 1GB)?

**Expected Current Behavior**:
- 2 buffers: CPU (~410 MB) + AMX (~656 MB)
- Total ~1GB (rest is mmap'd)
- Still getting NaN (but later in execution)

**Desired Behavior** (after fixes):
- 3 buffers: AMX + CPU_REPACK? + CPU_Mapped
- Total ~32GB
- No NaN

---

## Main Implementation Task

### Goal: Update Tensor Loading to Use Buffer Lists

**Files to Modify**:
1. `src/llama.cpp` - Lines around 4925-4945 (layer buffer setup)
2. `src/llama.cpp` - Lines around 5000-8000 (tensor loading in `llm_load_tensors`)

### Approach: Two Options

**Option A: Simple (Recommended First)**

Keep `model.buft_layer` as single `ggml_backend_buffer_type_t` but select it intelligently:

```cpp
// Around line 4938 in llama.cpp:
for (int i = 0; i < i_gpu_start; ++i) {
    // OLD CODE (WRONG):
    // model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, use_amx_buffer);

    // NEW CODE:
    buft_list_t buft_list = make_cpu_buft_list(use_amx, false);  // use_extra_bufts=use_amx, no_host=false

    // Use first buffer type (AMX if available, else CPU)
    model.buft_layer[i] = buft_list.empty() ? ggml_backend_cpu_buffer_type() : buft_list[0];
}
```

**Pros**:
- Minimal changes (~10 lines)
- Doesn't change model structure
- Quick to implement and test

**Cons**:
- All tensors in layer still use same buffer type
- Won't fully solve the issue if layer has mixed F32/Q4_0 tensors

**Option B: Full Implementation (Proper Solution)**

Change `model.buft_layer` to store buffer list and select per-tensor:

1. **Change Model Structure** (`src/llama-impl.h`):
```cpp
struct llama_model {
    // OLD:
    // std::vector<llama_layer_buft> buft_layer;

    // NEW:
    std::vector<buft_list_t> buft_layer;  // Each layer has list of buffer types
};
```

2. **Update Layer Setup** (`src/llama.cpp` ~line 4938):
```cpp
for (int i = 0; i < i_gpu_start; ++i) {
    model.buft_layer[i] = make_cpu_buft_list(use_amx, false);
}
```

3. **Update Tensor Loading** (`src/llama.cpp` in `llm_load_tensors`):

Find where tensors are allocated (search for `ggml_backend_buffer_type_t` in tensor loading code).

Add per-tensor selection:
```cpp
// For each tensor being loaded:
int layer_idx = get_layer_for_tensor(tensor);  // Determine which layer
if (layer_idx >= 0 && layer_idx < model.buft_layer.size()) {
    // Select best buffer type from list
    ggml_backend_buffer_type_t buft = select_weight_buft(
        tensor,
        GGML_OP_MUL_MAT,  // or determine actual op
        model.buft_layer[layer_idx]
    );

    // Allocate tensor in selected buffer
    // (existing allocation code, just use 'buft')
}
```

**Pros**:
- Proper solution matching upstream architecture
- Per-tensor buffer selection ensures Q4_0→AMX, F32→CPU
- Fully solves the buffer allocation issue

**Cons**:
- More changes (~100-200 lines)
- Requires understanding tensor loading code
- More testing needed

### Recommended Approach

**Phase 1**: Try Option A first (10 minutes)
- Quick test to see if it helps
- If it works partially, document findings

**Phase 2**: Implement Option B (proper solution)
- Study upstream `llama-model.cpp` tensor loading
- Port the buffer selection logic
- Test thoroughly

---

## Implementation Steps (Detailed)

### Step 1: Update Layer Buffer Setup

**File**: `src/llama.cpp`
**Location**: Around line 4925-4945

**Current Code**:
```cpp
bool need_host_buffer = (n_gpu_layers > 0);
bool use_amx_buffer = use_amx && !need_host_buffer;

// assign cpu layers
for (int i = 0; i < i_gpu_start; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, use_amx_buffer);
}
```

**New Code** (Option A - Simple):
```cpp
bool need_host_buffer = (n_gpu_layers > 0);
bool use_extra_bufts = use_amx;  // Enable AMX via extra buffer types
bool no_host = use_amx && !need_host_buffer;  // Skip host buffer for CPU-only AMX

// assign cpu layers
for (int i = 0; i < i_gpu_start; ++i) {
    if (use_extra_bufts) {
        buft_list_t buft_list = make_cpu_buft_list(use_extra_bufts, no_host);
        model.buft_layer[i] = buft_list.empty() ? ggml_backend_cpu_buffer_type() : buft_list[0];
    } else {
        model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, false);
    }
}
```

### Step 2: Test Option A

**Build and Test**:
```bash
cd /home/ron/src/ik_llama.cpp/build
cmake --build . --target llama-cli -j 32

numactl --cpunodebind=2 --membind=2 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 32 -n 20 -p "Test" --amx 2>&1 | tee /tmp/amx_option_a.log
```

**Check Results**:
```bash
# Buffer allocation
grep "buffer size\|buffer type" /tmp/amx_option_a.log

# NaN errors
grep -i "nan\|Oops" /tmp/amx_option_a.log

# Success check
tail -30 /tmp/amx_option_a.log
```

**Expected Outcomes**:

**If Better**:
- Might see 3 buffers instead of 2
- NaN might be delayed or reduced
- Some token generation might work

**If Same**:
- Still 2 buffers, still NaN
- Means we need Option B (per-tensor selection)

**If Worse**:
- Something broke in the changes
- Revert and debug

### Step 3: Implement Option B (If Needed)

**Study Upstream First**:
```bash
# Look at how upstream does it
grep -A 20 "select_weight_buft" /home/ron/src/llama.cpp/src/llama-model.cpp

# Look at tensor loading
grep -B 5 -A 10 "buft_layer" /home/ron/src/llama.cpp/src/llama-model.cpp | less
```

**Change Model Structure**:

Find `struct llama_model` definition and check if `buft_layer` can be changed to `std::vector<buft_list_t>`.

Our fork might have different structure than upstream. Check carefully.

**Update Tensor Allocation**:

Find in `llm_load_tensors` where tensors are allocated to buffers.

Look for patterns like:
```cpp
ggml_backend_buffer_type_t buft = model.buft_layer[some_index];
```

Replace with:
```cpp
ggml_backend_buffer_type_t buft = select_weight_buft(tensor, op, model.buft_layer[some_index]);
```

---

## Testing Checklist

After each change:

- [ ] Build succeeds without errors
- [ ] Check buffer allocation logs
- [ ] Count number of buffers created
- [ ] Check total buffer size
- [ ] Look for NaN errors
- [ ] Try to generate at least 10 tokens
- [ ] Compare output to non-AMX baseline

---

## Success Criteria

**Minimum Success**:
- No NaN errors
- Model generates valid tokens
- AMX buffer is created and used

**Full Success**:
- 3 buffers allocated (AMX, REPACK?, CPU_Mapped)
- Total buffer size ~32GB (not 1GB)
- Q4_0/Q4_1 tensors use AMX buffer
- F32 tensors use CPU buffer
- Token generation matches non-AMX output
- Performance improvement measurable

---

## Debugging If Things Break

**Common Issues**:

1. **Build Errors**:
   - Check that `buft_list_t` is defined (line 1786 in llama.cpp)
   - Check that `make_cpu_buft_list` is declared static (line 1805)
   - Check header includes

2. **Runtime Crash**:
   - Check `buft_list` is not empty before accessing `[0]`
   - Check `ggml_backend_cpu_get_extra_bufts()` returns valid pointer
   - Add debug logging

3. **Still Getting NaN**:
   - Means tensor loading still needs update (Option B required)
   - Check which tensors are in which buffers
   - Add logging to `select_weight_buft` to see what it's selecting

4. **Wrong Number of Buffers**:
   - Check `use_extra_bufts` is set correctly
   - Check `make_cpu_buft_list` is being called
   - Add logging to see buffer type list contents

---

## Debug Logging to Add

If debugging needed, add this to see buffer selection:

```cpp
// In make_cpu_buft_list:
LLAMA_LOG_INFO("%s: Creating buft_list with use_extra_bufts=%d, no_host=%d\n",
               __func__, use_extra_bufts, no_host);

// After adding each buffer type:
LLAMA_LOG_INFO("%s: Added buffer type: %s\n",
               __func__, ggml_backend_buft_name(buft));

// In select_weight_buft:
LLAMA_LOG_INFO("%s: Selecting buffer for tensor %s (type=%s)\n",
               __func__, tensor->name, ggml_type_name(tensor->type));

// After selection:
LLAMA_LOG_INFO("%s: Selected buffer type: %s\n",
               __func__, ggml_backend_buft_name(selected_buft));
```

---

## Quick Reference: Key File Locations

**Infrastructure (Already Done)**:
- `ggml/src/ggml-backend.cpp:733-772` - Extra buffer types
- `ggml/include/ggml-backend.h:128,131` - Declarations
- `src/llama.cpp:1785-1836` - Buffer list functions

**Need to Modify**:
- `src/llama.cpp:4925-4945` - Layer buffer setup ⚠️ PRIORITY
- `src/llama.cpp:5000-8000` - Tensor loading (if Option B) ⚠️ IF NEEDED

**Fixed Earlier**:
- `src/llama.cpp:4934` - buft_input (embeddings)
- `src/llama.cpp:4978,5002` - buft_output (output layer)

---

## Questions to Answer This Session

1. **Does Option A help at all?**
   - Even partial improvement suggests we're on right track

2. **How many buffers are actually created?**
   - Should move from 2 → 3

3. **What's the total buffer size?**
   - Should move from ~1GB → ~32GB

4. **Does NaN still occur? If so, where?**
   - Track down exactly which operation produces NaN
   - Might need Option B for full fix

5. **Can we generate any valid tokens?**
   - Even one valid token is progress

---

## Communication with User

User wants documentation kept updated. After this session:

1. Update `AMX_SESSION_CURRENT.md` with:
   - What was implemented
   - Test results
   - What worked / what didn't
   - Next steps

2. Update `AMX_BUFFER_SYSTEM_PORT.md` if architecture understanding changes

3. Create issue list if new problems found

4. Keep build/test commands documented

---

**END OF INSTRUCTIONS**

Start new session by:
1. Reading AMX_SESSION_CURRENT.md
2. Reading this file
3. Testing current state (Step 1 above)
4. Implementing Option A (Step 1 in Implementation)
5. Testing and documenting results
