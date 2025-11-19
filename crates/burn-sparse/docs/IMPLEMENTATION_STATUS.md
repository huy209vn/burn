# burn-sparse v2.0 - Implementation Status

**Last Updated:** 2025-11-19
**Total LOC:** ~3,700

## 📊 Overall Status: ~40% Complete (Phase 1 Foundation)

---

## ✅ **IMPLEMENTED** (Production Ready)

### Core Layer (~95% complete)
- [x] **SparseFormat** - All format enums (CSR, COO, CSC, BlockCSR, N:M)
- [x] **SparseMask** - Complete with gather/scatter, hamming distance, apply
- [x] **SparseTensor** - All format variants implemented
- [x] **CalibrationData** - Fully working with iterator
- [x] **ActivationStats** - Mean, std, L2 norm
- [x] **Error handling** - SparseError, SparseResult
- [x] **Validation** - Format validation for CSR, N:M constraints
- [x] **Format conversions** - Mask → CSR working and tested
- [x] **Utils** - topk, bottomk, percentile, reconstruction_error

**Test Coverage:** 30/31 tests passing (96.7%)
**Known Issues:** 1 test failure in `test_mask_construction` (nnz counting bug)

### Methods Layer - Static Pruning (~100% complete)
- [x] **Wanda** - Full implementation (330 LOC)
  - Activation-weighted magnitude pruning
  - L1/L2 norm modes
  - Calibration with samples
  - Score computation
  - Mask generation
  - **Tests:** All passing (5/5)

### Methods Layer - Iterative Pruning (~100% complete)
- [x] **DSnoT** - Full implementation (464 LOC)
  - Paper-exact algorithm (arXiv:2310.08915)
  - Variance-weighted mask refinement
  - Grow/prune score computation
  - Convergence detection
  - **Tests:** All passing (2/2)

---

## ⏳ **IN PROGRESS / PARTIAL**

### Format Conversions (~60% complete)
- [x] Mask → CSR (working)
- [ ] CSR → COO
- [ ] COO → CSR
- [ ] CSR → Mask
- [ ] BlockCSR conversions
- [ ] N:M conversions
- [ ] Conversion hub routing through CSR

---

## ❌ **NOT IMPLEMENTED** (Critical Path)

### Kernel Layer - CPU (Phase 1 blocker)
**Priority: HIGH - Required for Phase 1 completion**
- [ ] **CPU SpMM** - CSR sparse-dense matmul
- [ ] **CPU SddMM** - Sampled dense-dense (for DSnoT)
- [ ] **CPU format conversions** - Complete the hub
- [ ] **CPU validation** - Runtime checks in kernels

**Impact:** Cannot execute SparseTensor operations. Methods can generate masks but not train.

### Neural Network Layer (Phase 1.5)
**Priority: HIGH - Required for end-to-end training**
- [ ] **SparseLinear** - Standalone sparse linear layer
  - [ ] Forward pass (uses SpMM)
  - [ ] Backward pass (uses SddMM for weight grads)
  - [ ] Autodiff integration
  - [ ] `from_linear()` conversion API
  - [ ] `to_dense()` back-conversion
  - [ ] Mask update for dynamic sparsity
- [ ] **SparseConv2d** - Future, not v2.0 critical

**Impact:** No way to use sparse tensors in training. Methods exist but disconnected.

### Optimizer Layer (Phase 1.5)
**Priority: MEDIUM - Required for training convergence**
- [ ] **SparseOptimizer** - Wrapper for Burn optimizers
  - [ ] Sparse gradient handling
  - [ ] Momentum state (m, v) at active positions only
  - [ ] Mask update handling
  - [ ] Moment transfer on mask change
- [ ] **Sparse Adam** - Specialized implementation
- [ ] **Sparse SGD** - Specialized implementation

**Impact:** Cannot train sparse models properly. No momentum, no adaptive LR.

### Kernel Layer - CUDA (Phase 2)
**Priority: MEDIUM - Performance path**
- [ ] **cuSPARSE integration** - CSR SpMM wrapper
- [ ] **BlockCSR kernel** - Custom GPU kernel
- [ ] **N:M 2:4 kernel** - Tensor core acceleration
- [ ] **SddMM GPU** - Critical for DSnoT performance
- [ ] **Fused kernels** - SpMM + bias + activation

**Impact:** Stuck on CPU. No GPU acceleration. Cannot beat cuSPARSE.

### Kernel Layer - WGPU (Phase 2.5)
**Priority: LOW - Nice to have**
- [ ] **CSR SpMM** - Basic WebGPU compute shader
- [ ] **Fallback paths** - Graceful degradation

**Impact:** No web deployment. Desktop/server only.

### Methods Layer - Dynamic Sparsity (Phase 3)
**Priority: LOW - Research features**
- [ ] **RigL** - Dynamic sparse training
  - [ ] Gradient accumulation
  - [ ] Drop bottom-k active
  - [ ] Grow top-k pruned
  - [ ] Mask update integration
- [ ] **MEST** - Momentum-enabled sparse training
- [ ] **SET** - Sparse evolutionary training

**Impact:** Cannot do dynamic sparsity research. Static methods only.

### Documentation
**Priority: MEDIUM - User experience**
- [ ] Architecture docs (overview.md, core.md, kernels.md)
- [ ] Tutorial: Quickstart
- [ ] Tutorial: Wanda MNIST
- [ ] Tutorial: DSnoT refinement
- [ ] Tutorial: Custom method implementation
- [ ] API reference
- [ ] Performance benchmarks vs cuSPARSE
- [x] Design document (complete, 200KB)

---

## 🐛 **KNOWN ISSUES**

1. **test_mask_construction failing** - nnz count off by 1
   - Location: `src/core/sparse_tensor.rs:441`
   - Expected: 2, Got: 3
   - Impact: LOW - test bug, not production bug

2. **76 compiler warnings** - Mostly missing docs
   - Unused imports (3)
   - Missing module documentation (most)
   - Impact: LOW - code quality, not correctness

3. **Empty nn/ and optim/ directories** - Confusing
   - Should have at least README or stub files
   - Impact: LOW - confusing for contributors

---

## 🔄 **REDUNDANCY / CLEANUP NEEDED**

### Potential Code Smell: `primitives/` vs `core/`
**Location:** `src/primitives/` and `src/core/`

**Problem:**
Both exist. `lib.rs` has deprecated alias:
```rust
#[deprecated(since = "0.1.0", note = "Use `core` module instead")]
pub mod primitives {
    pub use crate::core::*;
}
```

**Files in primitives:**
- calibration.rs
- mask.rs
- stats.rs
- utils.rs
- mod.rs

**Files in core:**
- calibration.rs
- mask.rs
- stats.rs
- utils.rs
- + format.rs, sparse_tensor.rs, error.rs, validate.rs, convert.rs

**Resolution needed:**
- [ ] Delete `src/primitives/` entirely (deprecated legacy)
- [ ] Update examples if they reference primitives
- [ ] Remove deprecation shim from lib.rs

**Impact:** Maintenance burden, confusing directory structure.

---

## 📋 **PHASE ROADMAP**

### Phase 1: Core + CPU (8 weeks) - **40% DONE**
**Goal:** Working sparse infrastructure on CPU

Remaining:
- [ ] Fix test_mask_construction
- [ ] Delete primitives/ directory
- [ ] Implement CPU SpMM (CSR only)
- [ ] Implement CPU SddMM
- [ ] Complete format conversion hub
- [ ] Implement SparseLinear (basic)
- [ ] Write architecture docs
- [ ] Write quickstart tutorial
- [ ] End-to-end MNIST example (Wanda → train)

**Deliverable:** Can prune + train sparse models on CPU

### Phase 1.5: Autodiff + Optimizer (2 weeks)
- [ ] SparseLinear autodiff integration
- [ ] SparseOptimizer wrapper
- [ ] Momentum handling
- [ ] End-to-end training validation

**Deliverable:** Sparse training with proper gradients + optimizer

### Phase 2: CUDA Kernels (4 weeks)
- [ ] cuSPARSE CSR SpMM
- [ ] GPU SddMM
- [ ] BlockCSR custom kernel
- [ ] N:M 2:4 kernel (if time permits)
- [ ] Benchmarks vs cuSPARSE

**Deliverable:** GPU-accelerated sparse training, beat cuSPARSE

### Phase 2.5: CubeCL Migration (4 weeks)
**Critical decision point**

Options:
1. **Now:** Migrate before finishing CUDA
   - Pro: No duplicate work (CUDA + CubeCL)
   - Con: Delays GPU support, CubeCL learning curve

2. **After Phase 2:** Finish CUDA first
   - Pro: Working GPU path immediately
   - Con: Rewrite CUDA kernels in CubeCL later

3. **Never:** Stick with CUDA + WGPU separate
   - Pro: Simpler, proven
   - Con: Duplicate kernels, AMD/Metal support harder

**Recommendation:** Option 2 - finish CUDA first, then migrate
**Rationale:** Need working GPU soon. CubeCL can replace, not block.

### Phase 3: Dynamic Sparsity (2 weeks)
- [ ] RigL implementation
- [ ] MEST implementation
- [ ] Mask update integration with optimizer

**Deliverable:** Full dynamic sparse training

---

## 🎯 **IMMEDIATE NEXT STEPS** (Priority Order)

1. **Fix failing test** (30 min)
   - Debug test_mask_construction
   - Fix nnz counting logic

2. **Delete primitives/** (1 hour)
   - Remove deprecated directory
   - Update examples if needed
   - Clean deprecation shim

3. **Implement CPU SpMM - CSR only** (2-3 days)
   - Naive but correct implementation
   - Reference for correctness
   - Unblock SparseLinear

4. **Implement SparseLinear** (3-4 days)
   - Forward pass (calls SpMM)
   - Basic autodiff (may stub gradients initially)
   - from_linear() API
   - Tests

5. **End-to-end MNIST example** (2 days)
   - Wanda pruning → SparseLinear → train
   - Validate accuracy
   - Benchmark vs dense

6. **Write Phase 1 docs** (2 days)
   - Architecture overview
   - Quickstart tutorial
   - API reference

---

## 🔍 **API QUALITY ASSESSMENT**

### Good Decisions ✅
- Separation of SparseMask (algorithm) vs SparseTensor (execution)
- Format polymorphism through enum
- Panic on construction, Result on runtime
- Backend capability negotiation (KernelSupport)
- Calibration data abstraction

### Potential Issues ⚠️
1. **SparseTensor constructors unclear**
   - `from_mask()` vs `from_dense()` - when to use which?
   - Need builder pattern or clearer naming

2. **Conversion API split**
   - `SparseTensor::to_format()` vs `crate::core::convert::convert_format()`
   - Should be unified

3. **No clear "high-level" vs "low-level" API split**
   - Power users want direct SparseTensor
   - Casual users want SparseLinear
   - Need clear guidance

4. **Mask update ergonomics**
   - For RigL: `sparse_linear.update_mask(new_mask)` is manual
   - Should have integrated callback system?

### Recommendations
- [ ] Add builder pattern for SparseTensor
- [ ] Unify conversion APIs
- [ ] Add "Getting Started" with high-level API
- [ ] Add "Advanced" with low-level control

---

## 💭 **CubeCL DECISION FRAMEWORK**

### When to migrate?

**Migrate NOW if:**
- CUDA support can wait 4-6 weeks
- AMD ROCm is critical
- Team has CubeCL experience
- Want single kernel codebase

**Migrate AFTER Phase 2 if:**
- Need CUDA working ASAP
- CubeCL still maturing
- WGPU is low priority
- Okay with rewrite later

**Never migrate if:**
- CUDA-only target
- CubeCL too immature
- Team prefers handwritten kernels

### Current assessment:
**Recommendation: Migrate AFTER Phase 2**

Reasons:
1. Burn already uses CubeCL - consistency matters
2. AMD/Metal future-proofing
3. But: CPU + CUDA working first = momentum
4. CubeCL learning curve shouldn't block delivery

**Timeline:**
- Phase 1: CPU reference (8 weeks)
- Phase 2: CUDA production (4 weeks) ← **Deliver here**
- Phase 2.5: CubeCL migration (4 weeks) ← **Then modernize**

---

## 📦 **DELIVERABLES CHECKLIST**

### v0.1 (Phase 1 Complete)
- [ ] Core types (SparseMask, SparseTensor)
- [ ] Format conversions (Mask ↔ CSR ↔ COO)
- [ ] CPU SpMM, SddMM
- [ ] Wanda, DSnoT methods
- [ ] SparseLinear (basic)
- [ ] MNIST example
- [ ] Documentation (architecture + quickstart)

### v0.2 (Phase 2 Complete)
- [ ] CUDA SpMM (cuSPARSE)
- [ ] CUDA SddMM
- [ ] BlockCSR GPU kernel
- [ ] Benchmarks vs cuSPARSE
- [ ] SparseOptimizer
- [ ] Autodiff integration

### v0.3 (Phase 3 Complete)
- [ ] RigL, MEST dynamic methods
- [ ] Full optimizer integration
- [ ] Production-ready training

### v0.4 (CubeCL + Polish)
- [ ] CubeCL unified kernels
- [ ] WGPU support
- [ ] N:M 2:4 tensor cores
- [ ] Comprehensive benchmarks

---

## ❓ **OPEN QUESTIONS**

1. **Autodiff integration complexity?**
   - How hard is Burn's autodiff API to extend?
   - Can we register custom backward for SpMM?
   - Need prototype before committing to Phase 1.5

2. **cuSPARSE API stability?**
   - CUDA 12 changes?
   - Do we need cusparseSpMM or legacy APIs?

3. **CubeCL maturity?**
   - Production ready for sparse kernels?
   - Any known limitations?

4. **N:M hardware access?**
   - Do we have A100/H100 for testing?
   - Or stub for now?

---

## 🎓 **LESSONS LEARNED**

### What worked well:
- Design-first approach (200KB spec before coding)
- Separation of concerns (mask vs tensor)
- Test-driven (30 tests early)
- Clean error handling

### What needs improvement:
- Too many TODOs in design doc, not enough in code
- Empty directories confusing (nn/, optim/)
- Redundant primitives/ directory
- Insufficient examples early

### For Phase 2:
- Implement stubs with clear error messages, not empty dirs
- More examples DURING development, not after
- Benchmark from day 1, not "later"

---

## 📞 **CONTRIBUTION GUIDE**

Want to help? Pick from:

**Easy (good first issues):**
- Fix test_mask_construction
- Delete primitives/ directory
- Add missing module documentation
- Write examples/mnist_wanda.rs

**Medium:**
- Implement CPU SpMM (CSR)
- Complete format conversion hub
- Write architecture docs

**Hard:**
- Implement SparseLinear with autodiff
- CUDA cuSPARSE integration
- CubeCL kernel design

---

**Status:** Ready for Phase 1 completion sprint
**Next milestone:** v0.1 - CPU-complete sparse training
**ETA:** 4-6 weeks with focused effort
