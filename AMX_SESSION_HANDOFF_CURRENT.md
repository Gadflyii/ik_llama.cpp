# AMX Implementation - Session Handoff & Current Status

**⚠️ READ THIS FIRST - Context Summary for New Sessions**

**Last Updated**: 2025-10-06 14:50
**Current Branch**: `Add_AMX_Clean`
**Working Directory**: `/home/ron/src/ik_llama.cpp`
**Latest Commit**: `fb91d1c5` - "Fix AMX+GPU segfault by implementing cpy_tensor for AMX buffers"

---

## 🎯 Current Priority: Fix AMX+GPU Hybrid Mode Scheduler Bug

**Status**: 🟡 Implementation plan ready, awaiting execution
**Issue**: Long prompts (200+ tokens) crash when using AMX+GPU hybrid mode (`-ngl 15 --amx`)
**Root Cause**: Scheduler incorrectly copies weight tensors from AMX to CUDA (weights should never move)

### Quick Status
- ✅ **CPU-only AMX**: Fully working
- ✅ **GPU-only**: Fully working
- ✅ **AMX+GPU short prompts** (<10 tokens): Working at 61.24 t/s
- ❌ **AMX+GPU long prompts** (200+ tokens): Crashes during generation

### Next Steps
**📋 Follow instructions in**: `AMX_SCHEDULER_FIX_INSTRUCTIONS.md`

**Phase 1** (Research): Add diagnostic logging to identify which operation triggers weight copy
**Phase 2** (Fix): Modify scheduler to prevent weight tensor copies
**Phase 3** (Test): Validate all modes still work, no regressions

**Estimated effort**: 4-8 hours total

---

## 📚 Essential Documentation (Read in Order)

### 1. **AMX_SCHEDULER_FIX_INSTRUCTIONS.md** ← START HERE
Step-by-step implementation guide with:
- Pre-implementation checklist
- Backup procedures
- Code changes with exact locations
- Test procedures
- Success criteria

### 2. **AMX_GPU_HYBRID_SCHEDULER_FIX.md**
Technical deep-dive on the bug:
- Detailed crash analysis
- Why upstream doesn't have this issue
- Three solution approaches (chose #3)
- Architecture explanations

### 3. **AMX_SESSION_CURRENT.md**
Previous session context (buffer list architecture work)

---

## 🔧 Quick Reference

### Build & Test Commands

```bash
# Rebuild
cd /home/ron/src/ik_llama.cpp
cmake --build build -j 64

# Test CPU-only (works)
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 100 --amx -p "Test"

# Test AMX+GPU short prompt (works)
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -n 100 -ngl 15 --amx -fa -p "Test"

# Test AMX+GPU long prompt (CRASHES - need to fix)
numactl -N 2,3 -m 2,3 ./build/bin/llama-cli \
  -m /mnt/ssd2/AI/Qwen3_30B/Q4_0/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf \
  -t 64 -b 2048 -c 4096 -n 1000 -ngl 15 --amx -fa \
  -p "The history of artificial intelligence began in the 1950s..."
```

### Backup Location

Create timestamped backup before making changes:
```bash
BACKUP_DIR="/tmp/amx-scheduler-fix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
```

All previous backups in: `/tmp/amx-*`

---

## 🎭 What is Intel AMX?

**Intel Advanced Matrix Extensions (AMX)** - Hardware accelerator for matrix operations on Sapphire Rapids+ CPUs

- **Tile Registers**: 8 tile registers (tmm0-tmm7), each 1KB (16 rows × 64 bytes)
- **Operations**: TDPBF16PS (BF16), TDPBUUD/TDPBUSD (INT8), TDPBSUD/TDPBSSD (INT8)
- **VNNI Format**: Vector Neural Network Instructions format - special weight layout for AMX tiles
- **Performance**: 2-4x speedup for matrix operations on large batches (M>1)

### Why AMX in This Fork?

1. **CPU Performance**: Accelerates CPU layers in hybrid mode (some layers CPU, some GPU)
2. **Expert Routing**: Helps with MoE model expert selection (MUL_MAT_ID operations)
3. **Batch Processing**: Significant gains for M>1 batches (prompt processing)
4. **Single-token**: Uses AVX512-VNNI fallback (still faster than baseline)

---

## 🏗️ Architecture Overview

### Buffer Types (3-tier system)

```
┌─────────────────────────────────────────────────────────┐
│ Model Weights (16GB)                                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   AMX Buffer    │  │ CUDA_Host    │  │    CPU     │ │
│  │   (~334 MB)     │  │ (~11208 MB)  │  │  (mmap'd)  │ │
│  ├─────────────────┤  ├──────────────┤  ├────────────┤ │
│  │ Q4_0/Q4_1       │  │ Other quants │  │ F32 weights│ │
│  │ in VNNI format  │  │ Standard fmt │  │ (embeddings│ │
│  │                 │  │              │  │  norms, etc)│ │
│  │ CPU-only layers │  │ GPU transfers│  │            │ │
│  │ with AMX accel  │  │ & CPU fallbk │  │            │ │
│  └─────────────────┘  └──────────────┘  └────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Buffer Priority (Fork)

When `--amx` flag is set:
1. **AMX** - Q4_0/Q4_1 weights for CPU layers (VNNI format)
2. **CUDA_Host** - Pinned host memory for GPU transfers
3. **CPU** - Fallback for unsupported types (F32, etc.)

### Key Properties

**AMX Buffer**:
- `is_host() = false` - Not directly GPU-accessible (special format)
- `get_tensor = nullptr` - Cannot read VNNI data directly
- `cpy_tensor = [implemented]` - Handles host→AMX copies with VNNI conversion
- `set_tensor` - Converts weights to VNNI format on write

**CUDA_Host Buffer**:
- `is_host() = true` - Pinned host memory
- Accessible by both CPU and GPU
- Standard format (not VNNI)
- Used for efficient CPU↔GPU transfers

---

## 🐛 Known Issues & Limitations

### Issue #1: AMX+GPU Long Prompt Crash (CURRENT PRIORITY)
**Status**: 🔴 Fix in progress
**Symptom**: Segfault when generating after long prompt (200+ tokens)
**Cause**: Scheduler tries to copy `blk.0.attn_q.weight` from AMX to CUDA
**Solution**: See `AMX_SCHEDULER_FIX_INSTRUCTIONS.md`

### Issue #2: Buffer Reordering Causes OOM
**Status**: ⚠️ Architectural limitation
**Symptom**: Changing buffer priority to match upstream causes allocation failure
**Cause**: Fork lacks device registry infrastructure for smart buffer selection
**Workaround**: Keep current buffer priority, fix scheduler instead

### Issue #3: No NUMA Mirror Support Yet
**Status**: 🟡 Optional enhancement
**Impact**: Multi-socket systems don't get optimal performance
**Solution**: Port upstream's NUMA mirror buffer support (future work)

---

## 📊 Performance Expectations

### Current Performance (After All Fixes)

**CPU-only with AMX** (`--amx`):
- Prompt processing: 2-4x faster for M>1 batches
- Token generation: ~10-20% faster (AVX512-VNNI, not tiles)

**AMX+GPU Hybrid** (`-ngl 15 --amx`):
- Prompt processing: Similar to CPU-only AMX (for CPU layers)
- Token generation: ~61 t/s (GPU handles most layers)
- Best for models where some layers must stay on CPU

**GPU-only** (`-ngl 99`):
- Not affected by AMX (GPU handles everything)
- Baseline performance

---

## 🔍 Debugging Tips

### Enable Verbose Logging
```bash
# Model loading details
LLAMA_LOG_LEVEL=info ./build/bin/llama-cli ...

# CUDA operations
CUDA_LAUNCH_BLOCKING=1 ./build/bin/llama-cli ...
```

### Check Buffer Allocation
Look for these lines during model load:
```
llm_load_tensors:  CUDA_Host buffer size = 11207.91 MiB
llm_load_tensors:        AMX buffer size =   334.12 MiB
llm_load_tensors:      CUDA0 buffer size =  5027.13 MiB
```

### Monitor AMX Usage
```bash
# Run this while inference is running
sudo perf stat -a -e exe.amx_busy sleep 10
```

### GDB Debugging
```bash
gdb --args ./build/bin/llama-cli [args]
(gdb) run
# When it crashes:
(gdb) bt        # Backtrace
(gdb) frame 0   # Inspect crash location
(gdb) info registers
```

---

## 🎓 Implementation History

### Phase 1: Initial AMX Integration (Oct 2-3, 2025)
- ✅ Ported AMX kernels from upstream
- ✅ Fixed multi-threading barrier
- ✅ Fixed backend association (`is_host = true` workaround)
- ✅ Fixed tensor type filtering (only Q4_0/Q4_1 get AMX traits)
- ✅ CPU-only mode working

### Phase 2: Buffer System Architecture (Oct 3-4, 2025)
- ✅ Identified root cause: F32 tensors in AMX buffers → NaN
- ✅ Implemented buffer list infrastructure
- ✅ Fixed `buft_input` and `buft_output` to not use AMX
- ⚠️ Tensor loading still needs buffer list integration

### Phase 3: GPU Hybrid Mode (Oct 6, 2025)
- ✅ Fixed NULL `cpy_tensor` crash
- ✅ Implemented AMX buffer copy operations
- ✅ Short prompts working with AMX+GPU
- 🔴 Long prompts crash - scheduler bug (CURRENT PRIORITY)

---

## 🚀 After Scheduler Fix - Future Work

Once long prompt issue is resolved:

### Priority 1: Complete Buffer List Integration
- Finish tensor loading to use buffer lists
- Per-tensor buffer type selection
- Verify all tensor types go to correct buffers

### Priority 2: NUMA Optimization
- Port upstream's NUMA mirror support
- Replicate AMX buffers across nodes
- Test on dual-socket systems

### Priority 3: Performance Tuning
- Benchmark vs upstream
- Optimize tile configuration
- Profile and eliminate bottlenecks

### Priority 4: Documentation & Cleanup
- User guide for AMX flags
- Performance comparison charts
- Code cleanup and comments

---

## 📝 Session Workflow

### Starting a New Session

1. **Read this document first** (you're doing it! ✓)
2. **Check current priority** (scheduler fix)
3. **Review implementation instructions** (`AMX_SCHEDULER_FIX_INSTRUCTIONS.md`)
4. **Create backup** before making changes
5. **Follow step-by-step guide**
6. **Test thoroughly**
7. **Commit with detailed message**
8. **Update this document** with results

### Ending a Session

1. **Document progress** in this file
2. **List next steps** clearly
3. **Commit all changes**
4. **Save any test results** to `/tmp/amx-*`
5. **Update "Current Priority"** if changed

### If Context Runs Out Mid-Session

1. **Commit work in progress** (even if incomplete)
2. **Create detailed TODO** in this file
3. **Document exact next steps**
4. **Save all test logs** to backup directory
5. **Update "Last Updated"** timestamp

---

## 🎯 Success Metrics

### Minimum Viable Product (MVP)
- [ ] CPU-only AMX mode works ✅ **DONE**
- [ ] AMX+GPU hybrid works with short prompts ✅ **DONE**
- [ ] AMX+GPU hybrid works with long prompts ⏳ **IN PROGRESS**
- [ ] Performance matches or exceeds baseline ⏳ **Testing needed**
- [ ] No crashes, no NaN errors ⏳ **Testing needed**

### Complete Implementation
- [ ] Buffer list system fully integrated
- [ ] NUMA mirror support
- [ ] Comprehensive testing
- [ ] Performance documentation
- [ ] User guide

### Stretch Goals
- [ ] Dynamic batch size optimization
- [ ] Expert prediction caching for MoE
- [ ] Multi-GPU + AMX hybrid
- [ ] Upstream contribution

---

## 💡 Key Insights & Lessons Learned

### 1. Weights Should Never Move
**Lesson**: Weights are read-only model parameters that should stay on their allocated backend. Only intermediate activations should flow between backends during inference.

**Why it matters**: This is a fundamental architectural principle. The scheduler bug violates this, causing the crash.

### 2. Buffer Priority Matters
**Lesson**: The order of buffer types in the priority list determines where tensors land. Simple reordering can break allocation without smart selection logic.

**Why it matters**: Fork lacks device registry, so we can't just copy upstream's buffer order.

### 3. VNNI Format Is Opaque
**Lesson**: AMX VNNI format cannot be read directly. Any operation that needs to read weights must either:
- Not use AMX buffers, or
- Have special decompression logic

**Why it matters**: This is why `get_tensor = nullptr` and why the crash happens on fallback path.

### 4. Upstream Has Better Infrastructure
**Lesson**: Upstream's device registry enables per-tensor intelligent buffer selection. Fork's simpler architecture requires different solutions.

**Why it matters**: Can't just port upstream solutions directly - need to adapt to fork's architecture.

### 5. Test With Long Prompts
**Lesson**: Short prompts (single batch) don't trigger all code paths. Always test with long prompts to catch scheduler edge cases.

**Why it matters**: Bug only appears with 200+ token prompts that trigger multiple decode passes.

---

## 📞 Getting Help

### If Stuck
1. Read relevant documentation files (see "Essential Documentation" above)
2. Check git history: `git log --oneline --graph --all -20`
3. Look at upstream implementation: `/home/ron/src/llama.cpp`
4. Search for similar patterns in codebase
5. Add debug logging and investigate

### Common Pitfalls
- **Forgetting to rebuild**: Always `cmake --build build -j 64` after code changes
- **Wrong NUMA binding**: Use `numactl -N 2,3 -m 2,3` for optimal performance
- **Mixing AMX and non-AMX**: Use `--amx` consistently within a test
- **Not saving logs**: Always `tee` output to a file for later analysis

### Useful Commands
```bash
# Find function definition
grep -rn "function_name" ggml/src/

# Check buffer type implementation
grep -A 50 "ggml_backend.*_buffer_interface" ggml/src/

# See recent changes
git log --oneline --since="3 days ago"

# Restore from backup
cp /tmp/amx-scheduler-fix-YYYYMMDD-HHMMSS/file.cpp.backup path/to/file.cpp
```

---

## 🏁 Current Session Goals

**Primary Goal**: Fix AMX+GPU hybrid mode crash with long prompts

**Success Looks Like**:
- Long prompts work without crashes
- Performance maintained (~61 t/s)
- No regressions in other modes
- Code committed with comprehensive documentation

**Ready to Start?**
➡️ Open `AMX_SCHEDULER_FIX_INSTRUCTIONS.md` and begin Phase 1

---

**Document Status**: ✅ Complete and ready for session handoff
**Next Update**: After scheduler fix is implemented and tested
**Maintainer**: Update this file at end of each session
