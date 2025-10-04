# Performance Test Results - ik_llama.cpp AMX Implementation

**Date**: 2025-10-02 15:35 CDT
**Build**: c17d8425 (3902)
**Model**: Qwen3-30B Q4_0 (16.18 GiB, 30.53B params)
**Hardware**: Dual-socket Intel Xeon with AMX (NUMA nodes 2,3)

---

## Test Results Summary

| Configuration | Threads | NUMA Nodes | Prompt (pp512) | Token Gen (tg32) |
|--------------|---------|------------|----------------|------------------|
| **Test 1**   | 16      | Node 2     | **246.76 t/s** | **40.90 t/s**    |
| **Test 2**   | 32      | Node 2     | **405.06 t/s** | **57.04 t/s**    |
| **Test 3**   | 64      | Nodes 2,3  | **565.47 t/s** | **53.75 t/s**    |

---

## Detailed Results

### Test 1: 16 Threads (NUMA Node 2)

```
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      16 |         pp512 |    246.76 ± 3.36 |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      16 |          tg32 |     40.90 ± 0.42 |
```

- **Prompt processing**: 246.76 tokens/sec (±1.4%)
- **Token generation**: 40.90 tokens/sec (±1.0%)

### Test 2: 32 Threads (NUMA Node 2)

```
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      32 |         pp512 |    405.06 ± 8.42 |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      32 |          tg32 |     57.04 ± 0.43 |
```

- **Prompt processing**: 405.06 tokens/sec (±2.1%)
- **Token generation**: 57.04 tokens/sec (±0.8%)
- **Scaling vs 16T**: 1.64x prompt, 1.39x token gen

### Test 3: 64 Threads (NUMA Nodes 2,3)

```
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      64 |         pp512 |  565.47 ± 106.89 |
| qwen3moe ?B Q4_0               |  16.18 GiB |    30.53 B | CPU        |      64 |          tg32 |     53.75 ± 2.45 |
```

- **Prompt processing**: 565.47 tokens/sec (±18.9%)
- **Token generation**: 53.75 tokens/sec (±4.6%)
- **Scaling vs 32T**: 1.40x prompt, 0.94x token gen
- **Note**: Higher variance due to cross-socket communication

---

## Performance Analysis

### Scaling Efficiency

| Metric | 16T→32T | 32T→64T | 16T→64T |
|--------|---------|---------|---------|
| **Prompt** | 1.64x (82% eff) | 1.40x (70% eff) | 2.29x (57% eff) |
| **Token Gen** | 1.39x (70% eff) | 0.94x (47% eff) | 1.31x (33% eff) |

- **Best configuration for prompt**: 64 threads across 2 NUMA nodes (565 t/s)
- **Best configuration for token gen**: 32 threads on single NUMA node (57 t/s)
- **Balanced configuration**: 32 threads on single node (good for both)

### Observations

1. **Prompt Processing**:
   - Excellent scaling from 16→32 threads (82% efficiency)
   - Good scaling to 64 threads despite cross-socket overhead
   - Peak: 565 t/s with 64 threads

2. **Token Generation**:
   - Best performance at 32 threads (57 t/s)
   - Slight degradation at 64 threads due to synchronization overhead
   - Cross-socket penalty visible in token generation

3. **NUMA Effects**:
   - Single-node (32T) provides best token gen performance
   - Cross-socket (64T) benefits prompt processing more
   - Higher variance with cross-socket configuration

---

## Comparison with Previous Results

### Previous AMX Work (from conversation summary)
- Token generation: +24.8% speedup
- Prompt processing: -4.8% (slower than baseline)

### Current Implementation (infrastructure only)
- ✅ Infrastructure builds successfully
- ⚠️ **AMX tile operations not yet active** (using scalar fallbacks)
- ⚠️ **Repacking not yet integrated** with model loading

### Expected After Full Integration

Based on upstream llama.cpp results and current baseline:

| Metric | Current | Expected (Full AMX) | Improvement |
|--------|---------|---------------------|-------------|
| Prompt (32T) | 405 t/s | ~887 t/s | +119% |
| Token Gen (32T) | 57 t/s | ~71 t/s | +24.8% |

**Rationale**:
- Upstream achieved +119% prompt speedup with weight repacking + AMX
- Previous work achieved +24.8% token gen with basic AMX
- Current build has infrastructure but not active AMX tiles

---

## Next Steps to Activate AMX

1. **Enable AMX tile operations** in kernels
   - Replace scalar loops with `_tile_dpbssd` calls
   - Currently in: `ggml-amx-kernel.c`

2. **Integrate repacking with model load**
   - Detect Q4_0 quantization
   - Allocate repacked buffer (~11.6 GB)
   - Call `ggml_amx_repack_parallel()`

3. **Update dispatch logic**
   - Route Q4_0 matmuls to AMX kernels
   - Use repacked weights for inference

4. **Expected improvement**:
   - Prompt: 405 → 887 t/s (+119%)
   - Token gen: 57 → 71 t/s (+24.8%)

---

## Hardware Configuration

- **CPU**: Dual-socket Intel Xeon (Sapphire Rapids or later)
- **NUMA**: 4 nodes (nodes 2,3 used for testing)
- **Memory**: Sufficient for 30B model + repacked weights
- **AMX Support**: ✅ Detected and available
- **VNNI Support**: ✅ Hardware-accelerated

---

**Test Command**:
```bash
numactl -N 2,3 -m 2,3 /home/ron/src/ik_llama.cpp/build/bin/llama-bench \
    -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
    -t 64 \
    -p 512 \
    -n 32 \
    --numa numactl
```

---

**Status**: ✅ Baseline performance established
**Next**: Activate AMX tile operations for full speedup
