# Master Task List - AMX Implementation Project
**Created:** 2025-10-04 16:35
**Branch:** Add_AMX_Clean
**Status:** Active Development

---

## Mission Statement

Implement Intel AMX acceleration in ik_llama.cpp fork with:
1. Zero baseline impact without --amx flag
2. Full GPU compatibility (multi-GPU support)
3. Performance exceeding baseline
4. Future integration of SPARAMX sparse kernels

---

## Phase 1: AMX Isolation & GPU Fix (CURRENT)

### Goal
Make AMX changes completely conditional - baseline must be unchanged without --amx flag.

### Tasks
- [ ] 1.1: Document current state and baseline behavior
- [ ] 1.2: Analyze all AMX-related code changes
- [ ] 1.3: Implement conditional AMX activation (--amx flag only)
- [ ] 1.4: Test: Baseline without --amx must match main branch exactly
- [ ] 1.5: Analyze main branch GPU buffer allocation
- [ ] 1.6: Fix GPU buffer system to match main branch
- [ ] 1.7: Test multi-GPU support
- [ ] 1.8: Verify zero regressions

### Success Criteria
- ✅ Without --amx: Identical to main branch
- ✅ With --amx: AMX buffers created for CPU-only
- ✅ GPU works perfectly with/without --amx
- ✅ Multi-GPU support maintained

---

## Phase 2: Performance Testing

### Goal
Verify AMX implementation meets/exceeds baseline performance.

### Tasks
- [ ] 2.1: Create comprehensive test suite
- [ ] 2.2: Baseline tests (main branch, no AMX)
- [ ] 2.3: AMX tests (CPU-only with --amx)
- [ ] 2.4: GPU tests (with/without --amx)
- [ ] 2.5: Compare results
- [ ] 2.6: Document performance gains

### Performance Target
- Generation: ~46 t/s with AMX (vs baseline)
- Prompt processing: ~230 t/s with AMX (vs baseline)
- Goal: AMX > baseline performance

---

## Phase 3: Fork vs Upstream Analysis

### Goal
Understand why ik_llama.cpp (main) >> upstream llama.cpp performance.

### Tasks
- [ ] 3.1: Benchmark fork vs upstream
- [ ] 3.2: Investigate AVX512 implementations
- [ ] 3.3: Analyze IQK optimizations
- [ ] 3.4: Compare kernel implementations
- [ ] 3.5: Document findings
- [ ] 3.6: Identify opportunities for AMX

### Investigation Areas
- AVX512 usage differences
- IQK flash attention kernels
- GEMM implementations
- Quantization kernels
- Memory access patterns

---

## Phase 4: AMX Feature Parity

### Goal
Ensure our AMX implementation has feature parity with upstream llama.cpp AMX.

### Tasks
- [ ] 4.1: Review upstream AMX implementation
- [ ] 4.2: Compare feature sets
- [ ] 4.3: Identify missing features
- [ ] 4.4: Implement missing features
- [ ] 4.5: Test compatibility
- [ ] 4.6: Document differences

---

## Phase 5: SPARAMX Integration

### Goal
Integrate novel sparse AMX kernels from SPARAMX research.

### Tasks
- [ ] 5.1: Read SPARAMX paper (https://arxiv.org/abs/2502.12444)
- [ ] 5.2: Clone/analyze SPARAMX repo
- [ ] 5.3: Study PyTorch kernel implementations
- [ ] 5.4: Create detailed port plan
- [ ] 5.5: Create Add_Sparse_AMX branch
- [ ] 5.6: Port kernels to C/C++
- [ ] 5.7: Integrate into fork
- [ ] 5.8: Test sparse model performance
- [ ] 5.9: Optimize and tune
- [ ] 5.10: Document implementation

### SPARAMX Focus
- Novel sparse AMX approach
- Kernel innovations
- Integration with existing AMX
- Performance on MoE models (Qwen3-30B)

---

## Operational Constraints

### Required Standards
- ✅ Complete implementations (no stubs)
- ✅ Proper, clean code
- ✅ Unlimited time/iterations
- ✅ Never stop until goals complete
- ✅ Constant documentation updates

### Repository Rules
- **READ ONLY:** main branch, upstream repos
- **ACTIVE:** Add_AMX_Clean (current work)
- **FUTURE:** Add_Sparse_AMX (Phase 5)
- **BACKUP:** ~/src/claudebackup/ik_llama.cpp/

### Resources Available
- Internet access
- Web search
- Local repos in ~/src/
- Can clone additional repos
- Can create venvs

---

## Current Status

**Active Phase:** Phase 1 (AMX Isolation)
**Current Task:** Documenting task structure
**Branch:** Add_AMX_Clean
**Last Updated:** 2025-10-04 16:35

---

## Key Files

### Project Directories
- **Main:** /home/ron/src/ik_llama.cpp
- **Backup:** ~/src/claudebackup/ik_llama.cpp/
- **Upstream:** /home/ron/src/llama.cpp (reference only)
- **SPARAMX:** ~/src/SPARAMX (to be cloned)

### Documentation
- MASTER_TASK_LIST.md (this file)
- SESSION_STATUS.md (current status)
- PHASE_*.md (phase-specific details)
- BUILD_LOG.md (build history)
- TEST_RESULTS.md (test outcomes)

---

## Next Actions

1. Create detailed Phase 1 plan
2. Analyze main branch baseline
3. Identify all AMX changes
4. Begin conditional implementation
5. Test and verify

**Status:** Ready to proceed autonomously
