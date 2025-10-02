# AMX Implementation Status for ik_llama.cpp

## Summary
Intel AMX (Advanced Matrix Extensions) infrastructure has been successfully added to the ik_llama.cpp fork. **AMX tile operations are now functional and verified working.** Runtime control, initialization, build system, weight repacking, and dispatch hooks are all complete. Integration with the full inference pipeline is in progress.

**Latest Update (2025-10-02)**: AMX tile matrix multiply operations are now confirmed working via test function. Dispatch hooks have been added to ggml.c and are being triggered during inference. Weight repacking functions for Q4_0, Q4_1, and Q8_0 are implemented.

## ✅ Completed Components

### 1. Runtime Control System
- **Files**: `ggml/src/ggml-amx.c`, `ggml/src/ggml-amx.h`
- **Status**: ✅ Fully Functional
- **Features**:
  - `ggml_amx_set_enabled(bool)` - Enable/disable AMX at runtime
  - `ggml_amx_is_enabled()` - Query AMX status
  - Default: **DISABLED** (opt-in with `--amx` flag)
  - Global runtime flag prevents AMX usage when disabled

### 2. AMX Initialization
- **Status**: ✅ Fully Functional
- **Features**:
  - AMX-INT8 initialization (for quantized types: Q4_0, Q8_0, IQK types)
  - AMX-BF16 initialization (for FP16/BF16 types)
  - Linux syscall `ARCH_REQ_XCOMP_PERM` for XFEATURE_XTILEDATA
  - Proper error handling and logging
- **Test Results**:
  ```
  [AMX] Runtime AMX acceleration ENABLED
  [AMX-INT8] Successfully initialized
  [AMX-BF16] Successfully initialized
  ```

### 3. Command-Line Interface
- **Files**: `common/common.h`, `common/common.cpp`
- **Status**: ✅ Fully Functional
- **Features**:
  - `--amx` flag to enable AMX acceleration
  - Help text integration
  - Proper initialization in `llama_init_from_gpt_params()`
- **Usage**:
  ```bash
  # With AMX enabled
  ./llama-cli --amx -m model.gguf -p "prompt"

  # Without AMX (default)
  ./llama-cli -m model.gguf -p "prompt"
  ```

### 4. Build System Integration
- **Files**:
  - `ggml/CMakeLists.txt` - AMX build options
  - `ggml/src/CMakeLists.txt` - Compiler flags and source files
  - `common/CMakeLists.txt` - AMX flags for common library
- **Status**: ✅ Fully Functional
- **CMake Options**:
  - `GGML_AMX=ON` - Enable AMX support
  - `GGML_AMX_INT8=ON` - Enable AMX-INT8 (quantized types)
  - `GGML_AMX_BF16=ON` - Enable AMX-BF16 (floating point)
- **Compiler Flags Applied**:
  - `-mamx-tile` - Tile configuration support
  - `-mamx-int8` - INT8 matrix operations
  - `-mamx-bf16` - BF16 matrix operations
  - `-mavx512vnni` - Required for AMX-INT8

### 5. Bug Fixes
- **IQK Flash Attention**: Fixed C++ name mangling issue in `iqk_flash_attn.cpp`
  - Added `extern "C"` linkage
  - Fixed missing `sinks` parameter
- **Preprocessor Symbols**: Fixed AMX detection in common library
  - Added `-mavx512vnni` flag to common library build
  - Ensures `__AMX_INT8__` and `__AVX512VNNI__` are defined

## ⏳ Incomplete Components

### 1. Complete AMX Matrix Multiplication Kernels
- **Status**: ⚠️ Partially Implemented
- **Completed**:
  - ✅ Tile operations verified working (`_tile_dpbssd` tested)
  - ✅ Weight repacking for Q4_0, Q4_1, Q8_0
  - ✅ Dispatch hooks in ggml.c
- **Remaining Work**:
  - Full GEMV (matrix-vector) kernel implementation with quantized inputs
  - GEMM (matrix-matrix) for batched operations
  - Handle multi-threaded execution properly
  - IQK quantization kernels (IQ2_K through IQ6_K)
  - Trellis quantization kernels
  - Q4_K, Q5_K, Q6_K, IQ4_XS support (upstream has these)
- **Current State**: Proof-of-concept implementation exists, needs completion for full inference pipeline

### 2. Weight Repacking Infrastructure
- **Status**: ✅ Implemented for Q4_0, Q4_1, Q8_0
- **Completed Functions**:
  - `pack_B_q4_0()` - Repack Q4_0 to VNNI format
  - `pack_B_q4_1()` - Repack Q4_1 to VNNI format with mins
  - `pack_B_q8_0()` - Repack Q8_0 with s8s8 compensation
  - `ggml_amx_pack_weights()` - Public API
  - `ggml_amx_get_packed_size()` - Buffer size calculation
- **Remaining Work**:
  - On-the-fly repacking during inference (currently not called)
  - Repacking for Q4_K, Q5_K, Q6_K, IQ4_XS
  - Buffer management and caching strategy
  - Integration with model loading (optional optimization)

### 3. Integration with Inference Pipeline
- **Status**: ⚠️ Partially Implemented
- **Completed**:
  - ✅ Dispatch hooks added to ggml.c before iqk_mul_mat
  - ✅ Runtime detection working (`ggml_amx_is_enabled()`)
  - ✅ Type detection (`ggml_amx_can_handle()`)
- **Remaining Work**:
  - Complete the matmul kernel to actually perform computation
  - Handle quantization flow (src1 F32 -> Q8_0 conversion)
  - Multi-threaded support (currently single-thread only)
  - Batch processing support
  - Edge case handling (non-aligned dimensions)

### 4. Tile Configuration Management
- **Status**: ✅ Implemented
- **Completed**:
  - ✅ Thread-local tile configuration (`ggml_amx_tile_config_init()`)
  - ✅ Tile register allocation (TMM0-TMM7)
  - ✅ INT8 configuration (palette 1, 16x16 tiles)
  - ✅ Tile release (`_tile_release()`)
- **Verified**: Hardware test confirms correct configuration

### 5. Quantization Functions
- **Status**: ❌ Not Implemented (Stubs Only)
- **Required Work**:
  - AMX-accelerated quantization for IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K
  - AMX-accelerated quantization for Q4_0, Q8_0
  - Trellis quantization variants
- **Note**: May not be necessary if quantization is done offline

## 🏗️ Implementation Roadmap

### Phase 1: Basic Q4_0 AMX Kernel (2-3 days)
1. Study upstream `tinygemm_kernel_amx` for Q4_0 pattern
2. Implement weight repacking for Q4_0
3. Implement basic Q4_0 matrix-vector multiply (GEMV)
4. Test with Q4_0 model and verify AMX usage
5. Benchmark against AVX512 baseline

### Phase 2: Additional Quantization Types (1-2 weeks)
1. Implement Q8_0 kernel
2. Implement IQ2_K, IQ3_K, IQ4_K kernels
3. Implement IQ5_K, IQ6_K kernels
4. Implement Trellis variants
5. Test each quantization type

### Phase 3: Optimization (1 week)
1. Optimize repacking strategy
2. Implement matrix-matrix multiply (GEMM) for batch operations
3. Tune tile configuration for best performance
4. Implement BF16 kernels if needed
5. Comprehensive benchmarking

### Phase 4: Integration & Testing (3-5 days)
1. Integrate with iqk_mul_mat dispatch system
2. Test with various model sizes and quantizations
3. Performance validation across different workloads
4. Edge case handling
5. Documentation and examples

## 📊 Current Test Results

### Initialization Test (✅ Passing)
```bash
$ ./llama-cli --amx -m model.gguf -p "test" -n 1 -ngl 0
[AMX] Runtime AMX acceleration ENABLED
[AMX-INT8] Successfully initialized
[AMX-INT8] Running tile operation test...
[AMX] Test tile multiply result: C[0]=64 (expected 64)
[AMX-BF16] Successfully initialized
```
**Result**: ✅ AMX hardware is functional and tile operations work correctly

### AMX Tile Operations Test (✅ Passing)
The test function performs a 16x16 INT8 matrix multiply using AMX tiles:
- Loads two 16x64 INT8 matrices into tile registers
- Executes `_tile_dpbssd()` for matrix multiplication
- Stores result and verifies correctness
- **Expected**: C[0] = 64 (16 * 1 * 1 * 64 iterations)
- **Actual**: C[0] = 64 ✅

**Result**: AMX tile matrix multiply hardware operations are working correctly!

### Dispatch Hook Test (✅ Triggered - Single Thread)
```bash
$ ./llama-cli --amx -t 1 -m model.gguf -p "test" -n 1 -ngl 0
[AMX-DEBUG] src0_type=2 Q4_0=2, src1_type=0, dst_type=0 F32=0, ith=0 nth=1
[AMX] Using AMX path for Q4_0xF32: K=2048, N=1
[AMX] Using AMX path for Q4_0xF32: K=4096, N=1
[AMX] Using AMX path for Q4_0xF32: K=2048, N=1
```
**Result**: ✅ Dispatch hooks are triggered and detect Q4_0 matrices during inference

### Inference Test (⚠️ Partial - Dispatch Works, Kernel Incomplete)
```bash
$ ./llama-cli --amx -t 1 -m model.gguf -p "prompt" -n 10 -ngl 0
# Dispatches to AMX path successfully
# But falls through because GEMV kernel needs completion
# Currently uses iqk_mul_mat as fallback
```

**Monitoring Result**: AMX tile operations confirmed working in test, but full inference pipeline integration needs kernel completion

## 🔧 Build Instructions

```bash
cd ~/src/ik_llama.cpp
mkdir -p build && cd build

# Configure with AMX support
cmake .. \
  -DGGML_AMX=ON \
  -DGGML_AMX_INT8=ON \
  -DGGML_AMX_BF16=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Test
./bin/llama-cli --help | grep -A1 amx
./bin/llama-cli --amx -m /path/to/model.gguf -p "test" -n 10 -ngl 0
```

## 📝 Technical Notes

### AMX Tile Dimensions
- **TILE_M**: 16 rows
- **TILE_N**: 16 columns
- **TILE_K_INT8**: 64 elements (for INT8 operations)
- **TILE_K_BF16**: 32 elements (for BF16 operations)

### Tile Registers (TMM0-TMM7)
- TMM0, TMM1: B matrix tiles
- TMM2, TMM3: A matrix tiles
- TMM4-TMM7: C accumulator tiles (INT32 results)

### CPU Requirements
- Intel Xeon 4th Gen (Sapphire Rapids) or newer
- Linux kernel 5.16+ for proper AMX support
- AMX instructions: `amx_tile`, `amx_int8`, `amx_bf16` in `/proc/cpuinfo`

### Key Intrinsics
- `_tile_loadconfig()` - Configure tile dimensions
- `_tile_loadd()` - Load data into tile register
- `_tile_dpbssd()` - INT8 tile matrix multiply-accumulate
- `_tile_dpbf16ps()` - BF16 tile matrix multiply-accumulate
- `_tile_stored()` - Store tile register to memory
- `_tile_release()` - Release tile resources

## 🎯 Next Steps

**Immediate Priority**:
1. Implement basic Q4_0 weight repacking function
2. Implement Q4_0 AMX GEMV kernel using tile operations
3. Test and verify AMX tile usage in monitoring tools

**Medium Priority**:
1. Extend to other quantization types (Q8_0, IQK types)
2. Optimize repacking strategy and buffer management
3. Benchmark performance improvements

**Long-term**:
1. Full IQK quantization support
2. BF16 operations for FP16 types
3. GEMM optimization for batched operations
4. Comprehensive testing and validation

## 📚 References

- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-amx-overview.html)
- [Upstream llama.cpp AMX implementation](~/src/llama.cpp/ggml/src/ggml-cpu/amx/)
- [ik_llama.cpp IQK implementation](~/src/ik_llama.cpp/ggml/src/iqk/)

## 🐛 Known Issues

1. ✅ FIXED: IQK flash attention linker error (C++ name mangling)
2. ✅ FIXED: Common library missing AMX compiler flags
3. ⏳ TODO: AMX kernels not implemented (stubs only)
4. ⏳ TODO: Weight repacking infrastructure needed
5. ⏳ TODO: Integration with iqk_mul_mat dispatch system

---

**Last Updated**: 2025-10-02 18:00 UTC
**Status**: AMX Tiles Functional ✅ - Integration In Progress ⚠️

## 🎯 Current State Summary

**What's Working**:
- ✅ AMX hardware initialization and permission handling
- ✅ AMX tile operations (`_tile_dpbssd` verified with hardware test)
- ✅ Runtime control via `--amx` flag
- ✅ Weight repacking functions for Q4_0, Q4_1, Q8_0
- ✅ Dispatch hooks in ggml.c (triggered during inference)
- ✅ Build system integration with proper compiler flags

**What Remains**:
- ⚠️ Complete GEMV/GEMM kernel implementation for full inference
- ⚠️ Handle F32 → Q8_0 quantization in AMX path
- ⚠️ Multi-threaded execution support
- ⏳ Additional quantization types (Q4_K, Q5_K, Q6_K, IQK types)

**Significance**: This is a major milestone - AMX hardware operations are now confirmed working. The remaining work is completing the kernel to handle the full quantization and computation flow during inference.

### 6. AMX Tile Operations - VERIFIED WORKING ✅
- **Files**: `ggml/src/ggml-amx-kernel.c`
- **Status**: ✅ Functional - Hardware Verified
- **Test Results**:
  ```
  [AMX-INT8] Running tile operation test...
  [AMX] Test tile multiply result: C[0]=64 (expected 64)
  ```
- **Confirmed Working Operations**:
  - `_tile_loadconfig()` - Tile configuration
  - `_tile_loadd()` - Load data into tile registers
  - `_tile_dpbssd()` - INT8 tile matrix multiply-accumulate
  - `_tile_stored()` - Store tile results to memory
  - `_tile_release()` - Release tile resources
- **Hardware Test**: 16x16 INT8 matrix multiply executes correctly on AMX hardware

### 7. Weight Repacking Functions
- **Files**: `ggml/src/ggml-amx-impl.c`
- **Status**: ✅ Implemented (Q4_0, Q4_1, Q8_0)
- **Functions**:
  - `pack_B_q4_0()` - Repack Q4_0 weights for AMX VNNI format
  - `pack_B_q4_1()` - Repack Q4_1 weights for AMX VNNI format
  - `pack_B_q8_0()` - Repack Q8_0 weights with compensation
  - `ggml_amx_pack_weights()` - Public API for weight packing
  - `ggml_amx_get_packed_size()` - Calculate buffer size needed
  - `ggml_amx_can_handle()` - Check if quantization type is supported
- **Packed Layout**: Transposed VNNI format optimized for `_tile_dpbssd` operations

### 8. Dispatch Integration
- **Files**: `ggml/src/ggml.c` (lines 15585-15612)
- **Status**: ✅ Hooks Added and Triggered
- **Behavior**: 
  - AMX path is checked before `iqk_mul_mat` when `--amx` flag is set
  - Currently requires single-threaded execution (`-t 1`)
  - Detects Q4_0 matrices during matmul operations
- **Test Output**:
  ```
  [AMX-DEBUG] src0_type=2 Q4_0=2, src1_type=0, dst_type=0 F32=0, ith=0 nth=1
  [AMX] Using AMX path for Q4_0xF32: K=2048, N=1
  [AMX] Using AMX path for Q4_0xF32: K=4096, N=1
  ```
