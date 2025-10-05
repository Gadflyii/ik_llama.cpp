# Phase 2 Blockers - Urgent Issues

## Status: BLOCKED - Cannot Proceed to Phase 3

Two critical issues prevent Phase 2 completion:

## Blocker 1: CUDA Illegal Memory Access (RTX 5090 / CC 12.0)

### Error
```
ggml_cuda_compute_forward: GET_ROWS failed
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_compute_forward at ggml/src/ggml-cuda.cu:3328
```

### Impact
- **GPU offloading completely broken** in ik_llama.cpp
- Cannot test AMX+GPU hybrid mode
- Blocks all multi-device testing
- Affects baseline GPU performance measurement

### Investigation
- Issue #514 reported same error with RTX 5090
- Issue marked as closed by maintainer (ikawrakow)
- Error persists in current Add_AMX_Clean branch
- Main branch may have fix - needs sync investigation

### Next Steps
1. Check if main branch has RTX 5090 fix
2. Bisect to find fixing commit
3. Cherry-pick fix to Add_AMX_Clean
4. Verify GPU offloading works
5. Retest AMX+GPU hybrid mode

## Blocker 2: AMX Performance Regression (-16% to -36%)

### Findings
| Metric | Baseline CPU | AMX CPU | Difference |
|--------|--------------|---------|------------|
| Token Generation | 63.28 t/s | 52.99 t/s | **-16.3%** |
| Prompt Processing | 328.02 t/s | 210.12 t/s | **-36.0%** |

### Evidence
- AMX buffers allocated correctly (486 MiB)
- Tile operations present in code (_tile_loadd, etc.)
- Model type Q4_0 compatible with AMX
- No obvious implementation errors

### Possible Root Causes
1. **Suboptimal Tile Configuration**
   - Check TILE_M, TILE_N, TILE_K parameters
   - Verify tile register usage
   - Compare with upstream llama.cpp AMX

2. **Memory Bandwidth Bottleneck**
   - AMX may be memory-bound vs compute-bound
   - Check if mem bandwidth saturated
   - Profile cache miss rates

3. **Kernel Selection Issues**
   - Verify AMX kernels actually being used
   - Check for AVX512 fallback paths
   - Add instrumentation to confirm kernel dispatch

4. **Thread Pool Configuration**
   - Check thread affinity
   - Verify NUMA awareness
   - Test different thread counts

### Next Steps
1. Add debug logging to confirm AMX kernel usage
2. Profile with perf/vtune to identify bottleneck
3. Compare implementation with upstream
4. Test varying tile sizes
5. Benchmark memory bandwidth usage
6. Consider upstream merge if implementation differs significantly

## Resolution Strategy

### Phase 2 Cannot Continue Until:
- ✅ CMake build system fixed (DONE)
- ❌ CUDA bug resolved (BLOCKER 1)
- ❌ AMX performance >= baseline (BLOCKER 2)

### Timeline Impact
- Phase 3: Delayed until Phase 2 complete
- Phase 4: Delayed
- Phase 5: Delayed

### Recommendation
**PRIORITY**: Fix CUDA bug first (enables GPU testing), then optimize AMX performance.

Alternative: If AMX optimization proves complex, consider:
1. Merge upstream llama.cpp AMX implementation
2. Adapt to ik_llama.cpp buffer system
3. Leverage proven performant implementation
