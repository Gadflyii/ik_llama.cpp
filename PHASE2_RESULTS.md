# Phase 2: Performance Testing Results

## Test Configuration
- Model: Qwen3-30B-A3B Q4_0 (16.18 GiB)
- Hardware: 2x Xeon (128 cores), RTX 5090
- Test: 100 token generation, batch=1024, ctx=1024

## Performance Results

### CPU-Only Tests (numactl -N 2,3 -m 2,3, 64 threads)
| Config | Prompt Eval | Token Gen | vs Baseline |
|--------|-------------|-----------|-------------|
| Baseline CPU | 328.02 t/s | 63.28 t/s | - |
| AMX CPU | 210.12 t/s | 52.99 t/s | **-16.3%** |

**CRITICAL FINDING: AMX is SLOWER than baseline**
- Token generation: 52.99 vs 63.28 t/s (-16.3%)
- Prompt processing: 210.12 vs 328.02 t/s (-36%)

### GPU Tests (10 layers offloaded)
- Baseline GPU: FAILED - CUDA illegal memory access
- AMX+GPU: 45.83 t/s (but CUDA was disabled by numactl)

**Issue**: numactl NUMA pinning prevents CUDA access
```
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
```

## Buffer Allocation Verification

### Baseline CPU (no --amx)
- CPU buffer: 16569.15 MiB ✓
- No AMX buffers ✓

### AMX CPU (--amx)
- CPU buffer: 16158.23 MiB
- AMX buffer: 486.00 MiB ✓ 
- Tiles allocated ✓

### AMX+GPU (--amx -ngl 10)
- CUDA_Host buffer: 12832.99 MiB
- AMX buffer: 384.75 MiB ✓
- CUDA0 buffer: 3351.42 MiB ✓

## Build System Fix
**Problem**: AMX compiler flags (-mamx-tile, -mavx512vnni) were polluting IQK Flash Attention code

**Solution**: Applied per-file flags only:
```cmake
set_source_files_properties(
    ggml-cpu/amx/amx.cpp
    ggml-cpu/amx/mmq.cpp
    ggml-cpu-traits.cpp
    PROPERTIES COMPILE_FLAGS "${AMX_COMPILE_FLAGS_STR}"
)
```

Clean build now succeeds ✓

## Next Steps (Phase 2 continued)

1. **Investigate AMX Performance**
   - Why is AMX 16% slower than baseline?
   - Check tile configuration is optimal
   - Verify AMX kernels are actually being used
   - Compare memory bandwidth usage
   - Test without NUMA pinning

2. **Fix GPU Testing**
   - Run GPU tests without numactl
   - Verify CUDA + AMX hybrid performance
   - Test with multiple GPU configurations

3. **Upstream Comparison**
   - Compare with llama.cpp AMX implementation
   - Identify performance optimization opportunities

## Status
- ✅ Phase 1 Complete: AMX isolation, GPU hybrid mode
- 🔄 Phase 2 In Progress: Performance investigation needed
- ⏳ Phase 3: Pending
- ⏳ Phase 4: Pending  
- ⏳ Phase 5: Pending

## CRITICAL ISSUES DISCOVERED

### Issue 1: CUDA Illegal Memory Access (RTX 5090 / CC 12.0)
```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_compute_forward at /home/ron/src/ik_llama.cpp/ggml/src/ggml-cuda.cu:3328
```

**Impact**: GPU offloading completely broken in ik_llama.cpp fork
**Root Cause**: Likely compute capability 12.0 (RTX 5090) incompatibility
**Status**: Blocks all GPU+AMX hybrid testing

### Issue 2: AMX Performance Regression
**Finding**: AMX is 16-36% SLOWER than baseline CPU
- Token gen: 52.99 vs 63.28 t/s (-16.3%)
- Prompt eval: 210.12 vs 328.02 t/s (-36%)

**Possible Causes**:
1. Suboptimal tile configuration
2. Memory bandwidth bottleneck
3. Kernel selection issues
4. Thread synchronization overhead

### Issue 3: Build System (RESOLVED ✓)
**Problem**: AMX flags polluting IQK FA code
**Solution**: Per-file COMPILE_FLAGS in CMake
**Status**: Clean build working

## Recommendations

### Immediate Actions
1. **Fix CUDA bug** - Critical blocker for hybrid mode
   - Check ggml-cuda.cu line 3328 (GET_ROWS operation)
   - Review CC 12.0 specific issues
   - Compare with upstream llama.cpp CUDA implementation

2. **Investigate AMX Performance**
   - Profile tile usage vs AVX512 fallback
   - Check memory access patterns
   - Compare kernel selection with upstream
   - Verify thread pool configuration

3. **Retest After Fixes**
   - Baseline GPU must work before AMX+GPU testing
   - AMX CPU must match or exceed baseline before proceeding

### Phase 2 Status: BLOCKED
- ✅ AMX isolation complete
- ✅ Build system fixed  
- ❌ Performance regression found
- ❌ GPU testing blocked by CUDA bug
- ⏸️  Cannot proceed to Phase 3 until resolved

## Next Session Priorities
1. Debug and fix CUDA illegal memory access
2. Profile and optimize AMX performance
3. Re-run comprehensive benchmarks
4. Document performance improvements
5. Then proceed to Phase 3
