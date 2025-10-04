# AMX Implementations - Comprehensive Comparison
**Date**: 2025-10-03
**Purpose**: Identify missing fundamental piece causing NaN in ik_llama.cpp AMX integration

---

## Executive Summary

**Goal**: Compare 4 working AMX implementations to find what we're missing in ik_llama.cpp

**Implementations**:
1. ✅ **Upstream llama.cpp** (`/home/ron/src/llama.cpp` - numa_read_mirror branch) - PROVEN WORKING
2. **SparseAMX** (`/home/ron/src/SparAMX`) - Standalone library
3. **K-Transformers** (`/home/ron/src/ktransformers`) - Python framework with C++ AMX kernels
4. **Intel Extensions for PyTorch** (`/home/ron/src/intel-extension-for-pytorch`) - Official Intel implementation

**Current Problem**: Our ik_llama.cpp AMX port produces NaN despite:
- ✅ Correct struct layout (params->shared)
- ✅ Correct headers included
- ✅ Multi-threading barrier working
- ✅ Backend association working
- ✅ Tensor type filtering working
- ❌ NaN in ffn_moe_weights_sum-0

---

## Implementation 1: Upstream llama.cpp (numa_read_mirror)

### Location
`/home/ron/src/llama.cpp/ggml/src/ggml-cpu/amx/`

### Status
✅ **PROVEN WORKING** - Tested with Qwen3-30B Q4_0 model, generates correct output

### Key Files
