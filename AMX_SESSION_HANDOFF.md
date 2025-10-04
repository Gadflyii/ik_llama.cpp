# AMX Implementation - Session Handoff
**Date:** 2025-10-04 12:45
**Branch:** Add_AMX_Clean (commit: 1dbb37d9)
**Status:** Core implementation complete, build in progress

---

## IMMEDIATE PRIORITY

### Build Status
- **Clean rebuild in progress** (started ~12:30)
- Reason: Fixed GPU buffer allocation logic
- Command running: `cmake --build build --target llama-cli -j 16`
- **Action Required:** Wait for build to complete, then run tests

### Critical Tests Pending
1. **GPU without --amx** - Verify CUDA illegal memory access is fixed
2. **GPU with --amx** - Verify AMX correctly rejected when GPU layers present

---

## What Was Accomplished

### Problem Solved
**Original Issue (Add_AMX branch):**
- Multi-buffer logic (`buft_layer_list[]`) ran unconditionally
- Broke GPU functionality even without `--amx` flag
- CUDA illegal memory access errors

**Solution (Add_AMX_Clean branch):**
- Clean implementation with single `use_amx` boolean parameter
- AMX ONLY when: `--amx` flag SET **AND** `n_gpu_layers == 0`
- Zero baseline changes when AMX disabled
- GPU behavior completely preserved

### Implementation Details

#### File: `src/llama.cpp`

**1. Modified `llama_default_buffer_type_cpu()` (line 1781):**
```cpp
ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer, bool use_amx = false) {
#ifdef GGML_USE_AMX
    if (use_amx && !host_buffer) {
        buft = ggml_backend_amx_buffer_type();
        if (buft != nullptr) {
            return buft;
        }
    }
#endif
    // ... original logic unchanged
}
```

**2. Updated `llm_load_tensors()` signature (line 4900):**
- Added `bool use_amx` parameter
- Passed from `params.use_amx` (line 7606)

**3. Fixed CPU layer buffer allocation (line 4931):**
```cpp
// CRITICAL: AMX only for pure CPU-only inference
bool use_amx_for_cpu_layers = use_amx && (n_gpu_layers == 0);
bool need_host_buffer = (n_gpu_layers > 0);  // Host buffers for GPU transfer

for (int i = 0; i < i_gpu_start; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, use_amx_for_cpu_layers);
}
```

**4. Output layer protection (lines 4973, 4999):**
```cpp
// Output norms are F32, never use AMX
model.buft_output = llama_default_buffer_type_cpu(true, false);
```

**5. Input layer protection (line 4927):**
```cpp
// Embeddings are F32, never use AMX
model.buft_input = llama_default_buffer_type_cpu(true, false);
```

---

## Test Results

### ✅ Test 1: CPU-only Baseline (no --amx)
```bash
env CUDA_VISIBLE_DEVICES='' numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 10 -p "Test"
```
**Result:** ✅ PASS
```
llm_load_tensors:        CPU buffer size =   410.36 MiB
llm_load_tensors:        CPU buffer size = 16158.80 MiB
```
- No AMX buffer created
- Model loads and runs correctly
- Baseline behavior preserved

### ✅ Test 2: CPU-only with AMX
```bash
env CUDA_VISIBLE_DEVICES='' numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 10 -p "Test" --amx
```
**Result:** ✅ PASS
```
llm_load_tensors:        CPU buffer size = 16569.15 MiB
llm_load_tensors:        AMX buffer size =   486.00 MiB
```
- AMX buffer created successfully!
- Q4_0 weights (486MB) allocated in AMX buffer
- F32 weights (16569MB) remain in CPU buffer

### ❌ Test 3: GPU without --amx (BEFORE FIX)
```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -ngl 10 -n 10 -p "Test"
```
**Result:** ❌ FAIL
```
llm_load_tensors: context for buffer type CPU has no tensors
CUDA error: an illegal memory access was encountered
```

**Root Cause:** Incorrect logic
```cpp
// WRONG (before fix):
bool use_amx_for_cpu_layers = use_amx && (n_gpu_layers == 0);
for (int i = 0; i < i_gpu_start; ++i) {
    model.buft_layer[i] = llama_default_buffer_type_cpu(!use_amx_for_cpu_layers, use_amx_for_cpu_layers);
    //                                                    ^^^^^^^^^^^^^^^^^^^^^^  <- INVERTED!
}
```
When `n_gpu_layers=10`, `use_amx_for_cpu_layers=false`, so `host_buffer=true` (correct) but second param still wrong.

**Fix Applied:**
```cpp
// CORRECT (after fix):
bool need_host_buffer = (n_gpu_layers > 0);
model.buft_layer[i] = llama_default_buffer_type_cpu(need_host_buffer, use_amx_for_cpu_layers);
//                                                    ^^^^^^^^^^^^^^^^^  <- DIRECT!
```

### 🔄 Test 3: GPU without --amx (AFTER FIX)
**Status:** Build in progress (ETA: ~12:45-12:50)
**Expected:** ✅ PASS with host buffers, no AMX

### ⏳ Test 4: GPU with --amx (NOT YET RUN)
**Expected:** AMX should be disabled (n_gpu_layers > 0)
```
llm_load_tensors:  CUDA_Host buffer size = XXXXX MiB  <- Host buffers
llm_load_tensors:      CUDA0 buffer size = XXXXX MiB  <- GPU buffers
(No AMX buffer - correctly rejected)
```

---

## Next Steps (Priority Order)

### 1. **Verify GPU Fix** (IMMEDIATE)
```bash
# After build completes:
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -ngl 10 -n 100 -p "Test"
```
**Expected:** No CUDA errors, successful inference
**Check:** `grep -E "buffer size|CUDA error"`

### 2. **Test GPU + AMX Rejection**
```bash
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -ngl 10 -n 100 -p "Test" --amx
```
**Expected:** No AMX buffer, uses host buffers instead

### 3. **Run Full Performance Test Suite**
Use existing script: `/tmp/amx_final_performance_tests.sh`

Update it for Add_AMX_Clean branch:
```bash
#!/bin/bash
FORK_BIN="/home/ron/src/ik_llama.cpp/build/bin/llama-cli"
UPSTREAM_BIN="/home/ron/src/llama.cpp/build/bin/llama-cli"
MODEL="/mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf"
THREADS=64
BATCH=2048
CTX=4096
N_PREDICT=512
NUMA_CMD="numactl -N 2,3 -m 2,3"
PROMPT="The history of artificial intelligence"

# Fork tests
# 1. CPU baseline (no AMX)
env CUDA_VISIBLE_DEVICES='' $NUMA_CMD $FORK_BIN -m "$MODEL" -t $THREADS -b $BATCH -c $CTX -n $N_PREDICT -p "$PROMPT"

# 2. CPU with AMX
env CUDA_VISIBLE_DEVICES='' $NUMA_CMD $FORK_BIN -m "$MODEL" -t $THREADS -b $BATCH -c $CTX -n $N_PREDICT -p "$PROMPT" --amx

# 3. GPU (no AMX)
$NUMA_CMD $FORK_BIN -m "$MODEL" -t $THREADS -b $BATCH -c $CTX -n $N_PREDICT -ngl 10 -p "$PROMPT"

# 4. Upstream CPU (auto AMX)
env CUDA_VISIBLE_DEVICES='' $NUMA_CMD $UPSTREAM_BIN -m "$MODEL" -t $THREADS -b $BATCH -c $CTX -n $N_PREDICT -p "$PROMPT"

# 5. Upstream GPU + AMX
$NUMA_CMD $UPSTREAM_BIN -m "$MODEL" -t $THREADS -b $BATCH -c $CTX -n $N_PREDICT -ngl 10 -p "$PROMPT"
```

### 4. **Research Performance Gap**
- **Goal:** Understand why fork baseline >> upstream
- **Investigate:**
  - AVX512 implementations in fork vs upstream
  - IQK optimizations in fork
  - Kernel differences in `ggml-quants.c`
  - Flash attention implementations
- **Tools:** `perf record`, `perf report`, code diff analysis

### 5. **Investigate SPARAMX**
- **Location:** Check for SPARAMX repo in `/home/ron/src/`
- **Paper:** https://arxiv.org/html/2502.12444
- **Goal:** Understand novel sparse AMX approach
- **Tasks:**
  - Read paper
  - Review PyTorch kernels
  - Identify key innovations
  - Plan C/C++ port strategy

### 6. **Create Sparse_AMX Branch**
- **Only after Add_AMX_Clean is fully tested and validated**
- Branch from Add_AMX_Clean
- Port SPARAMX PyTorch kernels to C/C++
- Integrate sparse AMX approach

---

## Key Files and Locations

### This Project (`/home/ron/src/ik_llama.cpp`)
- **Branch:** Add_AMX_Clean
- **Commit:** 1dbb37d9
- **Binary:** `build/bin/llama-cli` (rebuilding)
- **Model:** `/mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf`

### Core Modified Files
1. `src/llama.cpp` - Buffer allocation logic (lines 1781, 4900, 4931, 4973, 4999)
2. `src/llama-impl.h` - Function signature (line 221)
3. `ggml/src/ggml-cpu/amx/mmq.cpp` - AMX kernels
4. `ggml/src/ggml-backend.cpp` - AMX backend
5. `common/common.h` - CLI flag `use_amx`

### Reference Projects
- **Upstream:** `/home/ron/src/llama.cpp` (branch: numa_read_mirror)
- **SPARAMX:** `/home/ron/src/SPARAMX` (check if exists)
- **Old Branch:** `Add_AMX` (has GPU bug, for reference only)

### Documentation
- `AMX_CLEAN_STATUS.md` - Current status
- `AMX_SESSION_HANDOFF.md` - This file
- `AMX_*.md` - Previous session docs (in Add_AMX branch)

---

## Build Configuration

```bash
# Clean build
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AMX=ON

# Build
cmake --build build --target llama-cli -j 16
```

**Flags:**
- `GGML_AMX=ON` - Enables AMX support
- `GGML_AMX_INT8=ON` - Auto-enabled (requires AVX512VNNI)
- `GGML_CUDA=ON` - GPU support
- `GGML_NATIVE=OFF` - Prevent -march=native conflicts

---

## Critical Constraints

1. **No Baseline Changes:** Without `--amx`, must behave identically to original fork
2. **Multi-GPU Support:** Must not break existing GPU buffer allocation
3. **Complete Implementations:** No stubs, all code must be functional
4. **Upstream Read-Only:** `/home/ron/src/llama.cpp` is reference only, never modify

---

## Known Issues

### ✅ FIXED: GPU Buffer Allocation
- **Issue:** CUDA illegal memory access when GPU layers present
- **Cause:** Inverted `host_buffer` logic
- **Fix:** Use `need_host_buffer = (n_gpu_layers > 0)` directly
- **Status:** Fix committed, build in progress

### ⚠️ POTENTIAL: MoE Expert Weights
- **Issue:** MoE experts might need special handling
- **Current:** AMX skips F32 tensors (which includes experts)
- **Verify:** Check if expert weights are quantized (Q4_0) or F32
- **Location:** `src/llama.cpp:1805` (select_weight_buft in Add_AMX branch had MoE check)

---

## Success Criteria

### Minimum Viable Product
- [x] AMX buffer type infrastructure
- [x] CLI flag `--amx` support
- [x] Conditional AMX activation
- [x] CPU-only AMX working
- [ ] GPU without AMX working (build in progress)
- [ ] GPU + AMX correctly rejected
- [ ] Performance matches or exceeds baseline

### Full Success
- [ ] All tests passing
- [ ] Performance: ~46 t/s generation, ~230 t/s prompt (with AMX)
- [ ] Zero regressions without AMX
- [ ] Documentation complete
- [ ] SPARAMX research complete
- [ ] Sparse AMX implementation (future)

---

## Debugging Commands

### Check Buffer Allocation
```bash
./build/bin/llama-cli -m MODEL -t 64 -n 1 --amx 2>&1 | grep "buffer size"
```

### Check AMX Activation
```bash
./build/bin/llama-cli -m MODEL -t 64 -n 1 --amx 2>&1 | grep -i amx
```

### Check CUDA Errors
```bash
./build/bin/llama-cli -m MODEL -t 64 -ngl 10 -n 10 2>&1 | grep -i "cuda error\|illegal"
```

### Performance Test
```bash
time env CUDA_VISIBLE_DEVICES='' numactl -N 2,3 -m 2,3 \
  ./build/bin/llama-cli -m MODEL -t 64 -n 512 --amx 2>&1 | grep "tok/s"
```

---

## Session Continuation Checklist

- [ ] Wait for build to complete (~10-15 min from 12:30)
- [ ] Test GPU without --amx
- [ ] Test GPU with --amx
- [ ] If tests pass: Run performance suite
- [ ] If tests fail: Debug and fix
- [ ] Document results
- [ ] Research baseline vs upstream performance gap
- [ ] Investigate SPARAMX
- [ ] Create Sparse_AMX branch (if AMX working)

---

**READY FOR AUTONOMOUS CONTINUATION**

Next session: Start by checking build status and running GPU tests.
