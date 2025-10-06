# AMX + GPU Hybrid Mode Investigation Log

## Investigation Goal
Fix V weights failing in AMX format when GPU is present. Q and K weights work fine, but adding V causes all Q/K/V projections to fail.

## Timeline of Investigation

### 2025-10-05: Initial Problem Identification

**Symptom:** With `--amx -ngl >0`, system crashes or hangs with NaN

**Initial tests:**
- `-ngl 1`: NaN in ffn_moe_weights_sum-0
- `-ngl 20`: Same NaN error
- `-ngl 30`: Same NaN error

**Finding:** `is_host` setting was initially `true`, changed to `false` to match upstream

### Discovery: Incremental Weight Testing

**Systematic test approach:**
1. Only attn_output in AMX → ✅ Works
2. attn_output + attn_q → ✅ Works
3. attn_output + attn_q + attn_k → ✅ Works
4. attn_output + attn_q + attn_k + attn_v → ❌ NaN!

**Conclusion:** V weights specifically cause the failure

**Key observation:** Q and K have same dimensions as V (2048×512 for GQA), so not a dimension issue

### Critical Discovery: AMX Computation Pattern

**Test with Q+K in AMX (V excluded):**
```
AMX computing: Qcur-0 (many times)
AMX computing: Kcur-0 (many times)
Result: SUCCESS - no NaN
```

**Test with Q+K+V all in AMX:**
```
(No AMX computing logs at all for Q/K/V)
Only: AMX computing kqv_out-0
Result: FAILURE - NaN in MoE router
```

**CRITICAL FINDING:** When V weights are added to AMX buffers, AMX stops computing ALL Q/K/V projections. The regular CPU backend then tries to read VNNI-formatted weights directly, producing corrupt output.

### Root Cause Analysis

**What we know:**
1. V weights CAN be in VNNI format (upstream proves this)
2. When V is in AMX buffer, something prevents AMX from being selected for Q/K/V operations
3. Q and K work fine when V is absent
4. The issue is scheduler-related, not kernel-related

**Hypotheses:**
1. V's `tensor->extra` field not being set to AMX traits when V present?
2. Operation scheduling logic has a check that fails only when V is in AMX buffer?
3. Some property of V tensor differs from Q/K that affects trait assignment?
4. Buffer initialization order issue?

### Comparison with Upstream

**Upstream behavior (with `--no-host -ngl 1`):**
```
AMX computes: Qcur-0 ✅
AMX computes: Kcur-0 ✅
AMX computes: Vcur-0 ✅
Result: Works perfectly
```

**Fork behavior (with `--amx -ngl 1`):**
- Q+K in AMX: AMX computes Qcur, Kcur ✅
- Q+K+V in AMX: AMX computes NOTHING ❌

**Key difference:** Upstream has device registry infrastructure. Fork uses simpler tensor traits via `extra` field.

### Code Differences

**Upstream `amx.cpp`:**
- Has `.device` field in buffer_type struct
- Uses `ggml_backend_reg_dev_get()`
- Part of device registry system (added Oct 2024)

**Fork `amx.cpp`:**
- No `.device` field (doesn't exist in fork's struct)
- Comment acknowledges this difference
- Uses `is_host = false` to associate with CPU backend

### Attempted Fixes

**Fix 1:** Changed `is_host` from `true` to `false`
- Result: Prevented CUDA crash, but exposed NaN issue

**Fix 2:** Disabled AMX when GPU present (`&& !gpu_present`)
- Result: Works but violates requirement (AMX must work with GPU)
- Rejected as workaround

**Fix 3:** Exclude V weights from AMX buffers
- Result: Works perfectly with `-ngl 20`
- Status: Temporary workaround, not acceptable as final solution

### Current Workaround (Temporary)

In `src/llama.cpp` line 1810:
```cpp
const bool allow_amx_for_this_tensor = is_attn_output || is_attn_q || is_attn_k;
// V excluded - workaround only
```

This allows testing other functionality but doesn't solve the root problem.

## ROOT CAUSE IDENTIFIED AND FIXED ✅

### Problem Summary
When V weights were in AMX buffers with GPU present, the CPU backend's `supports_buft()` function was rejecting AMX buffers because AMX has `is_host=false`. This caused the scheduler to route Q/K/V projection operations to other backends instead of CPU+AMX.

### Root Cause Chain
1. AMX buffer type sets `is_host=false` (correct - prevents CUDA from direct memory access)
2. CPU backend's `ggml_backend_cpu_supports_buft()` only accepted buffers where `is_host=true`
3. Scheduler calls `ggml_backend_sched_buffer_supported()` which checks if CPU backend supports AMX buffer type
4. CPU backend says "no" → scheduler routes operations to other backends
5. Operations never reach AMX dispatch code → VNNI-formatted weights read by wrong backend → NaN

### Solution Implemented
Two fixes in `/home/ron/src/ik_llama.cpp/ggml/src/ggml-backend.cpp`:

**Fix 1: CPU backend supports AMX buffer type (line 979-995)**
```cpp
GGML_CALL static bool ggml_backend_cpu_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // CPU backend supports host buffers and also AMX buffers (which are CPU-based but not "host")
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }

#ifdef GGML_USE_AMX
    // AMX buffers are CPU-based even though is_host=false
    if (buft == ggml_backend_amx_buffer_type()) {
        return true;
    }
#endif

    return false;
}
```

**Fix 2: CPU backend checks buffer type constraints (line 959-971)**
```cpp
        case GGML_OP_MUL_MAT:
            // For operations with src[0] in AMX buffer, check AMX-specific constraints
            // AMX requires src[1] to be F32 and in host buffer, which regular CPU doesn't care about
            if (op->src[0] && op->src[0]->buffer && op->src[0]->buffer->buft) {
                // Let buffer type decide if it can handle this specific operation
                // This calls AMX's extra_buffer_type::supports_op() which checks:
                // - src[1] is F32
                // - src[1] is in host buffer (not AMX buffer)
                // - Dimensions are contiguous 2D
                return ggml_backend_buft_supports_op(op->src[0]->buffer->buft, op);
            }
            return true;
```

### Test Results
All scenarios now working:
- ✅ CPU-only mode: Q+K+V all computed by AMX
- ✅ `-ngl 1`: Q+K+V all computed by AMX
- ✅ `-ngl 10`: Q+K+V all computed by AMX
- ✅ No NaN errors
- ✅ Coherent text generation

## Reference Commands

### Debug Builds
```bash
cmake --build build --target llama-cli -j 16
```

### Test Configurations
```bash
# Q+K only (works)
./build/bin/llama-cli -m [model] --amx -ngl 1 -n 10 -p "test"

# Q+K+V (fails)
# Modify line 1810 to include is_attn_v, rebuild, test

# Upstream reference (works)
cd /home/ron/src/llama.cpp
./build/bin/llama-cli -m [model] --no-host -ngl 1 -n 10 -p "test"
```

### Add Debug Logging
```cpp
// In compute_forward (amx.cpp:29)
if (strstr(op->name, "cur-0")) {
    fprintf(stderr, "AMX computing: %s\n", op->name);
}

// In init_tensor (amx.cpp:68)
fprintf(stderr, "AMX init_tensor: %s, extra=%p\n",
        tensor->name, tensor->extra);
```

## Resources

- Upstream AMX: `/home/ron/src/llama.cpp/ggml/src/ggml-cpu/amx/`
- ktransformers: `/home/ron/src/ktransformers/`
- Fork AMX: `/home/ron/src/ik_llama.cpp/ggml/src/ggml-cpu/amx/`
- Test model: `/mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf`

## Success Criteria

- [ ] Understand WHY V's presence stops AMX from computing
- [ ] Implement proper fix (not workaround)
- [ ] All Q/K/V weights in AMX buffers with GPU
- [ ] Tests pass: -ngl 1, 10, 20, 30
- [ ] AMX verified to be computing Q/K/V projections
- [ ] Performance benchmarked and documented
- [ ] Feature parity with upstream confirmed
