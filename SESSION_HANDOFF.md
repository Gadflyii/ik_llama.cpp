# AMX Implementation - Session Handoff Document

**Date:** 2025-10-02
**Repository:** `/home/ron/src/ik_llama.cpp` (fork)
**Upstream Reference:** `/home/ron/src/llama.cpp` (branch: `numa_read_mirror`)
**Current Status:** Backend buffer type implemented, integration pending

---

## CRITICAL INSTRUCTIONS FOR NEXT SESSION

### Development Rules
1. **NO STUBS** - Every function must be complete and functional
2. **NO INCOMPLETE IMPLEMENTATIONS** - Code must compile and work correctly
3. **DO NOT MODIFY UPSTREAM** - The upstream repo at `/home/ron/src/llama.cpp` on branch `numa_read_mirror` is READ-ONLY reference
4. **DO NOT REBUILD UPSTREAM** - Use it only as implementation reference

### Priority Goals
1. **HIGHEST PRIORITY:** Complete Q4_0 and Q8_0 support with all required kernels
2. **HIGHEST PRIORITY:** All block types and kernels must be fully functional
3. **LOWEST PRIORITY:** Additional quantization types beyond Q4_0 and Q8_0

### Implementation Strategy
- **Reference upstream implementation:** `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/` for patterns
- **Reference Intel documentation:** AMX intrinsics guide and examples
- **Reference GitHub examples:** Search for AMX implementations
- **Continuously update:** [AMX_STATUS.md](AMX_STATUS.md) with detailed tasks and progress

---

## Current State Analysis

### What Works in Upstream (numa_read_mirror branch)
Located at: `/home/ron/src/llama.cpp`

1. ✅ **Backend Buffer Type System**
   - `ggml_backend_cpu_repack_buffer_type()` in [ggml-cpu/repack.cpp:1872-2052](../llama.cpp/ggml/src/ggml-cpu/repack.cpp)
   - `init_tensor` callback stores tensor_traits in `tensor->extra`
   - `set_tensor` callback unpacks weights ONCE during model load
   - Automatic selection during model loading

2. ✅ **Quantization Support**
   - Q4_0, Q4_1, Q8_0 fully supported
   - Q4_K, Q5_K, Q6_K, Q2_K, Q3_K supported
   - IQ4_NL, IQ4_XS supported
   - Each has dedicated repack function

3. ✅ **AMX Tile Operations**
   - Full tile-based matrix multiplication
   - Proper VNNI data layout handling
   - Stride calculation: `TILE_N × VNNI_BLK = 64`
   - Implementation in [ggml-cpu/amx/mmq.cpp](../llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp)

4. ✅ **GEMM and GEMV Kernels**
   - Separate kernels for different quantization types
   - Optimized for both prompt (GEMM) and token generation (GEMV)
   - Fallback to AVX-512 when tiles not optimal

5. ✅ **Integration**
   - Automatically selected based on tensor properties
   - CPU backend detects and dispatches to repack buffer
   - Works with NUMA mirror mode

### What Works in Our Fork (ik_llama.cpp)
Located at: `/home/ron/src/ik_llama.cpp`

1. ✅ **Backend Buffer Type Infrastructure**
   - `ggml_backend_amx_repack_buffer_type()` in [ggml/src/ggml-backend.cpp:788-919](ggml/src/ggml-backend.cpp)
   - `init_tensor` callback stores repack function pointer
   - `set_tensor` callback calls repack function
   - `get_alloc_size` returns correct repacked size
   - **STATUS:** Implemented but NOT integrated into model loading

2. ✅ **Q4_0 Repacking**
   - `ggml_repack_q4_0_to_q4_0x8()` in [ggml/src/ggml-amx-repack.c:40-81](ggml/src/ggml-amx-repack.c)
   - Unpacks nibbles to full INT8 format
   - Groups 8 blocks together
   - **STATUS:** Working, tested

3. ✅ **F32 → Q8_0x8 Quantization**
   - `ggml_quantize_f32_to_q8_0x8()` in [ggml/src/ggml-amx-repack.c:97-149](ggml/src/ggml-amx-repack.c)
   - Per-block absmax quantization
   - **STATUS:** Working, tested

4. ✅ **Scalar INT8 GEMV Kernel**
   - `ggml_amx_gemv_q4_0_8x8_q8_0()` in [ggml/src/ggml-amx-repack.c:196-219](ggml/src/ggml-amx-repack.c)
   - Produces CORRECT output (verified)
   - **STATUS:** Working but SLOW (~41 t/s vs 37.79 t/s baseline)

5. ✅ **Integration Wrapper**
   - `ggml_amx_mul_mat_q4_0_f32()` in [ggml/src/ggml-amx-impl.c:359-407](ggml/src/ggml-amx-impl.c)
   - Expects already-repacked weights
   - **STATUS:** Working but buffer type not integrated yet

### What's Missing in Our Fork

1. ❌ **Buffer Type Integration**
   - Buffer type exists but model loading doesn't use it
   - Still allocating with regular CPU buffer type
   - Need to hook into llama.cpp model loading code

2. ❌ **Q8_0 Repacking**
   - No `ggml_repack_q8_0_to_q8_0x8()` function
   - Upstream has this at [ggml-cpu/repack.cpp:138-182](../llama.cpp/ggml/src/ggml-cpu/repack.cpp)

3. ❌ **AVX-512 VNNI Vectorization**
   - Only scalar loops (slow)
   - Need `_mm512_dpbusd_epi32()` for INT8×INT8
   - Upstream uses this extensively

4. ❌ **AMX Tile Operations**
   - Tile infrastructure exists but not active
   - Need proper VNNI layout and stride
   - Upstream reference: [ggml-cpu/amx/mmq.cpp:2066-2073](../llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp)

5. ❌ **GEMM Kernel**
   - Only GEMV exists (one row at a time)
   - Need batch processing for prompt
   - Upstream has separate GEMM kernels

6. ❌ **Automatic Dispatch**
   - Upstream automatically selects repack buffer based on tensor
   - We need similar logic in model loading

---

## File Locations

### Our Fork (`/home/ron/src/ik_llama.cpp`)

**Core AMX Implementation:**
- `ggml/src/ggml-amx.c` - Runtime detection, initialization
- `ggml/src/ggml-amx.h` - Public API
- `ggml/src/ggml-amx-impl.c` - Integration wrapper
- `ggml/src/ggml-amx-repack.c` - Repacking, quantization, kernels
- `ggml/src/ggml-amx-repack.h` - Data structures, declarations
- `ggml/src/ggml-backend.cpp:788-919` - AMX buffer type
- `ggml/include/ggml-backend.h:118` - Buffer type API

**Build System:**
- `ggml/src/CMakeLists.txt` - Build configuration

**Documentation:**
- `AMX_STATUS.md` - Current status and tasks

### Upstream Reference (`/home/ron/src/llama.cpp` - READ ONLY)

**Branch:** `numa_read_mirror`

**Key References:**
- `ggml/src/ggml-cpu/repack.cpp` - Complete repack buffer implementation
- `ggml/src/ggml-cpu/amx/mmq.cpp` - AMX tile operations
- `ggml/src/ggml-cpu/amx/common.h` - AMX common utilities
- `ggml/src/ggml-backend.cpp` - Backend system
- `src/llama.cpp` - Model loading (find buffer type selection)

---

## Detailed Task List

### PHASE 1: Complete Q4_0 Support (HIGHEST PRIORITY)

#### Task 1.1: Integrate Buffer Type into Model Loading
**Goal:** Make model loading actually use `ggml_backend_amx_repack_buffer_type()`

**Steps:**
1. Study upstream's buffer type selection in `/home/ron/src/llama.cpp/src/llama.cpp`
   - Search for `ggml_backend_cpu_repack_buffer_type()`
   - Find where buffer types are selected during model load
   - Understand the selection criteria

2. Implement similar logic in our fork
   - Add condition: if `--amx` flag AND tensor type is Q4_0
   - Return `ggml_backend_amx_repack_buffer_type()` instead of `ggml_backend_cpu_buffer_type()`

3. Add debug logging
   - Print when AMX buffer type is selected
   - Print in `set_tensor` callback when repacking happens
   - Verify it happens ONCE during load, NOT during inference

4. Test and verify
```bash
# Should see repacking during load
GGML_LOG_LEVEL=debug ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 16 -c 256 -n 8 -p "test" --amx 2>&1 | grep -i "repack\|set_tensor"
```

**Expected:**
- See `set_tensor` messages during model load
- NO repacking during token generation
- Performance same as baseline (~37 t/s) - no regression

**Files to modify:**
- `src/llama.cpp` or equivalent model loading code
- Possibly `ggml/src/ggml-backend.cpp` if dispatch logic needs changes

**Completion criteria:**
- [ ] Buffer type selected automatically when AMX enabled
- [ ] Weights unpacked ONCE during model load
- [ ] No repacking during inference
- [ ] No performance regression

---

#### Task 1.2: Add AVX-512 VNNI Vectorization to GEMV
**Goal:** Speed up scalar INT8×INT8 dot products using `_mm512_dpbusd_epi32()`

**Reference upstream:**
- Search `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/` for `_mm512_dpbusd_epi32`
- Look at how they vectorize INT8 dot products

**Implementation in:** `ggml/src/ggml-amx-repack.c`

**Current scalar code (lines 196-219):**
```c
for (int k = 0; k < QK4_0; k++) {
    sumi += (int32_t)x_block->qs[j * QK4_0 + k] *
            (int32_t)y_block->qs[j * QK8_0 + k];
}
```

**Replace with AVX-512 VNNI:**
```c
#if defined(__AVX512VNNI__)
__m512i acc = _mm512_setzero_si512();
for (int k = 0; k < QK4_0; k += 64) {
    __m512i vx = _mm512_loadu_si512(&x_block->qs[j * QK4_0 + k]);
    __m512i vy = _mm512_loadu_si512(&y_block->qs[j * QK8_0 + k]);
    acc = _mm512_dpbusd_epi32(acc, vx, vy);
}
int32_t sumi = _mm512_reduce_add_epi32(acc);
#else
// fallback to scalar
#endif
```

**Test:**
```bash
# Should be 3-4x faster than scalar
./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 16 -n 64 -p "test" --amx
```

**Expected:** ~60-80 t/s token generation

**Completion criteria:**
- [ ] AVX-512 VNNI code compiles
- [ ] Output matches scalar version (correctness test)
- [ ] Performance 3-4x better than scalar
- [ ] Fallback to scalar when AVX-512 not available

---

#### Task 1.3: Implement AMX Tile Operations for Q4_0
**Goal:** Replace scalar/vectorized with actual AMX tiles

**Prerequisites:**
- Task 1.1 complete (buffer type integrated)
- Task 1.2 complete (vectorized version as reference)

**Reference upstream:**
- `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp` lines 2066-2073
- Study tile configuration, VNNI layout, stride calculation

**Key challenges:**
1. **VNNI Layout:** Data must be arranged in VNNI blocks (4 INT8 elements)
2. **Stride Parameter:** `_tile_loadd()` needs stride = `TILE_N × VNNI_BLK = 64`
3. **Tile Config:** Proper `tileconfig` structure for 16×16×32 tiles

**Implementation approach:**
1. Start with simplest case: 16×16×32 tiles
2. Implement for single block first
3. Add tile configuration setup
4. Implement tile load, multiply, store
5. Test against vectorized output (must match exactly)
6. Only then optimize for larger blocks

**Code location:** `ggml/src/ggml-amx-repack.c`

**Completion criteria:**
- [ ] Tile operations produce identical output to vectorized
- [ ] Performance matches or exceeds upstream (~46 t/s)
- [ ] Fallback to vectorized when tiles not available
- [ ] No crashes or undefined behavior

---

#### Task 1.4: Implement GEMM Kernel for Q4_0
**Goal:** Batch-process multiple output rows for faster prompt processing

**Current:** `ggml_amx_gemv_q4_0_8x8_q8_0()` processes one row at a time

**New:** `ggml_amx_gemm_q4_0_8x8_q8_0()` processes N rows in batch

**Reference upstream:**
- Search for GEMM vs GEMV in `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/`
- Understand batching strategy

**Implementation:**
1. Keep GEMV for N=1 (single row)
2. Add GEMM for N>1 (multiple rows)
3. Use same tile operations but process multiple outputs
4. Test correctness by comparing to sequential GEMV calls

**Target performance:** ~230 t/s prompt processing

**Completion criteria:**
- [ ] GEMM produces identical output to N×GEMV
- [ ] Prompt processing ~2x faster than baseline
- [ ] Automatic selection between GEMV and GEMM based on N

---

### PHASE 2: Add Q8_0 Support (HIGHEST PRIORITY)

#### Task 2.1: Implement Q8_0 Repacking
**Goal:** Support Q8_0 quantized weights

**Reference upstream:**
- `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/repack.cpp` lines 138-182
- Q8_0 is already in INT8 format, just needs regrouping

**Implementation in:** `ggml/src/ggml-amx-repack.c`

**Add function:**
```c
void ggml_repack_q8_0_to_q8_0x8(
    const block_q8_0 * src,
    block_q8_0x8_unpacked * dst,
    int64_t n_blocks);
```

**Add to buffer type:**
- Update `ggml_amx_get_repack_fn()` to handle GGML_TYPE_Q8_0
- Update `ggml_amx_get_repacked_size()` for Q8_0
- Add repack callback in `ggml-backend.cpp`

**Completion criteria:**
- [ ] Q8_0 repacking function implemented
- [ ] Buffer type supports Q8_0
- [ ] Tested with Q8_0 model

---

#### Task 2.2: Add Q8_0 GEMV Kernel
**Goal:** Support Q8_0×F32 matrix multiplication

**Implementation:** Similar to Q4_0 kernel but simpler (no nibble unpacking)

**Completion criteria:**
- [ ] Q8_0 GEMV kernel working
- [ ] Vectorized with AVX-512 VNNI
- [ ] Tile operations implemented
- [ ] Performance matches upstream

---

#### Task 2.3: Add Q8_0 GEMM Kernel
**Goal:** Batch processing for Q8_0 prompt evaluation

**Completion criteria:**
- [ ] Q8_0 GEMM kernel working
- [ ] Performance matches upstream

---

### PHASE 3: Testing and Validation

#### Task 3.1: Correctness Testing
```bash
# Test Q4_0
./build/bin/llama-cli -m model_q4_0.gguf -s 42 -n 100 -p "Once upon" > baseline.txt
./build/bin/llama-cli -m model_q4_0.gguf -s 42 -n 100 -p "Once upon" --amx > amx.txt
diff baseline.txt amx.txt  # Must be identical

# Test Q8_0
./build/bin/llama-cli -m model_q8_0.gguf -s 42 -n 100 -p "Once upon" > baseline.txt
./build/bin/llama-cli -m model_q8_0.gguf -s 42 -n 100 -p "Once upon" --amx > amx.txt
diff baseline.txt amx.txt  # Must be identical
```

**Completion criteria:**
- [ ] Outputs identical with/without AMX
- [ ] Works across different model sizes
- [ ] No crashes or undefined behavior

---

#### Task 3.2: Performance Testing
```bash
# Benchmark Q4_0
./build/bin/llama-cli -m model_q4_0.gguf -t 16 -n 256 -p "test"
./build/bin/llama-cli -m model_q4_0.gguf -t 16 -n 256 -p "test" --amx

# Benchmark Q8_0
./build/bin/llama-cli -m model_q8_0.gguf -t 16 -n 256 -p "test"
./build/bin/llama-cli -m model_q8_0.gguf -t 16 -n 256 -p "test" --amx
```

**Target metrics:**
- Token generation: ~46 t/s (+24.8% vs baseline)
- Prompt processing: ~230 t/s (+119% vs baseline)

**Completion criteria:**
- [ ] Q4_0 performance matches upstream
- [ ] Q8_0 performance matches upstream
- [ ] No performance regressions

---

### PHASE 4: Additional Quantizations (LOWEST PRIORITY)

**Only start after Q4_0 and Q8_0 are COMPLETELY FUNCTIONAL**

- Q4_1 support
- Q4_K, Q5_K, Q6_K support
- Q2_K, Q3_K support
- IQ4_NL, IQ4_XS support

---

## Build Commands

```bash
# Clean build
cd /home/ron/src/ik_llama.cpp
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_OPENMP=ON -DGGML_NATIVE=OFF
cmake --build build --target llama-cli -j 16

# Quick rebuild after changes
cmake --build build --target llama-cli -j 16

# Test
./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 16 -c 256 -n 8 -p "test" --amx
```

---

## Key Data Structures

### Block Formats (Unpacked)
```c
// Q4_0 unpacked format (272 bytes)
typedef struct {
    ggml_fp16_t d[8];       // 8 scale factors
    int8_t qs[32 * 8];      // 256 INT8 values (from nibbles)
} block_q4_0x8_unpacked;

// Q8_0 unpacked format (272 bytes)
typedef struct {
    ggml_fp16_t d[8];       // 8 scale factors
    int8_t qs[32 * 8];      // 256 INT8 values
} block_q8_0x8_unpacked;
```

### Buffer Type Callbacks
```c
// init_tensor: Store repack function in tensor->extra
static void ggml_backend_amx_repack_buffer_init_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

// set_tensor: Call repack function ONCE during model load
static void ggml_backend_amx_repack_buffer_set_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    const void * data, size_t offset, size_t size);

// get_alloc_size: Return repacked size
static size_t ggml_backend_amx_repack_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
```

---

## Reference Documentation

### Intel AMX
- **Intrinsics Guide:** https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Key intrinsics:**
  - `_tile_loadconfig()` - Configure tiles
  - `_tile_loadd()` - Load tile with stride
  - `_tile_dpbssd()` - INT8×INT8 matrix multiply
  - `_tile_stored()` - Store tile result
  - `_tile_release()` - Release tiles

### AVX-512 VNNI
- **Key intrinsic:** `_mm512_dpbusd_epi32()`
- **Layout:** 4 INT8 elements per VNNI block
- **Stride:** For tiles, stride = TILE_N × VNNI_BLK = 64 bytes

### GitHub Examples
- Search: "AMX INT8 matrix multiplication"
- Search: "AVX512 VNNI INT8 dot product"
- Look for working code examples to reference

---

## Progress Tracking

**Update [AMX_STATUS.md](AMX_STATUS.md) after each task with:**
- Task completion status
- Performance numbers
- Any issues encountered
- Next steps

**Format:**
```markdown
### Task X.Y: [Name] - [Status]
**Completed:** YYYY-MM-DD
**Performance:** XX t/s token gen, XX t/s prompt
**Issues:** [any issues]
**Verification:** [test command and results]
```

---

## Success Criteria

### Minimum Viable Product (Q4_0 + Q8_0)
- [x] Buffer type infrastructure complete
- [ ] Buffer type integrated into model loading
- [ ] Q4_0 GEMV with AVX-512 VNNI
- [ ] Q4_0 GEMV with AMX tiles
- [ ] Q4_0 GEMM with AMX tiles
- [ ] Q8_0 GEMV with AVX-512 VNNI
- [ ] Q8_0 GEMV with AMX tiles
- [ ] Q8_0 GEMM with AMX tiles
- [ ] All tests passing (correctness)
- [ ] Performance matches upstream

### Performance Targets
- Token generation: ~46 t/s (+24.8%)
- Prompt processing: ~230 t/s (+119%)
- No regressions vs baseline

---

## IMPORTANT REMINDERS

1. ✅ **Always compile test** - Every change must compile successfully
2. ✅ **Always correctness test** - Output must match baseline/upstream
3. ✅ **Always reference upstream** - Don't guess, look at working code
4. ✅ **NO STUBS** - Complete implementations only
5. ✅ **Update AMX_STATUS.md** - Keep documentation current
6. ❌ **DON'T TOUCH UPSTREAM** - It's read-only reference

---

**Ready to continue with Task 1.1: Integrate Buffer Type into Model Loading**
