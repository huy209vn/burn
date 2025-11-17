# burn-sparse v2.0 — SOTA Implementation Plan

**Philosophy**: Infrastructure-first. No MVPs. Production-grade from day one.

## Current State Analysis

### ✅ What Exists
- Core architecture: `SparseTensor`, `SparseMask`, `SparseFormat`
- Basic format stubs: CSR, COO, BlockCSR, N:M
- Validation framework: `validate.rs`
- Methods: Wanda (complete), DSnoT (complete)
- Calibration infrastructure: `CalibrationData`, `ActivationStats`
- Kernel API: `SparseKernel` trait, capability routing
- CPU kernel stubs

### ❌ What's Missing (Critical Path)
- **Format conversions** (all stubs - blocks everything)
- **CPU kernels** (SpMM, SddMM - needed for correctness validation)
- **nn/**: SparseLinear, autodiff integration
- **optim/**: Sparse optimizer adapter
- **CUDA kernels**: cuSPARSE integration, custom kernels
- **Dynamic methods**: RigL, MEST
- **Tests**: Property-based, correctness, e2e
- **Benchmarks**: vs cuSPARSE baseline

---

## Phase 1: Foundations (Week 1-2) — Make It Work

**Goal**: Core working on CPU. All formats convert correctly. Tests pass.

### 1.1 Format Conversions (CRITICAL - Day 1-3)
**Why first**: Everything blocks on this. Can't test SpMM without CSR. Can't validate without round-trips.

#### Files to implement:
```
src/core/convert.rs — Complete implementation
```

**Deliverables**:
- [x] Mask → CSR (CPU-side construction)
  - Gather (row, col, val) triplets
  - Sort by row then col
  - Build row_pointers via cumsum
  - Create col_indices, values tensors

- [x] CSR → Mask (expand to dense boolean)
  - Iterate row_pointers
  - Set mask[i, col_idx[j]] = true

- [x] CSR ↔ COO (bidirectional)
  - CSR→COO: expand row_pointers to row_indices
  - COO→CSR: sort by (row, col), build row_pointers

- [x] COO ↔ Mask (bidirectional)

- [x] Mask → BlockCSR
  - Pad to block alignment
  - Extract dense blocks where any nnz
  - Store block indices

- [x] Mask → N:M (with projection)
  - Iterate in M-sized groups
  - Project to nearest N:M pattern (greedy top-N)
  - Encode metadata (2 bits for 2:4)

- [x] Universal hub: All X ↔ CSR paths
  - CSR is canonical format
  - Other formats convert via CSR

**Test coverage**:
```rust
#[test] fn csr_mask_roundtrip() // Mask → CSR → Mask preserves
#[test] fn coo_csr_roundtrip()
#[test] fn nm_projection_valid() // 2:4 pattern enforced
#[test] fn blockcsr_alignment()  // Blocks properly aligned
```

**Success criteria**:
- All format pairs convert without panics
- Round-trips preserve values (ε < 1e-6)
- N:M projection satisfies constraint
- Property tests pass (1000 random tensors)

---

### 1.2 CPU Kernels — Reference Implementation (Day 4-6)

**Why**: Establishes correctness baseline. GPU kernels must match these.

#### Files:
```
src/kernel/cpu/spmm.rs    — SpMM implementations
src/kernel/cpu/sddmm.rs   — SddMM implementation
src/kernel/cpu/mod.rs     — CPU backend registration
```

**Deliverables**:

#### SpMM (Sparse @ Dense → Dense)
- **CSR SpMM** (naive, correct):
  ```rust
  fn spmm_csr(A_csr, B_dense) -> Y_dense {
      for row in 0..n_rows {
          for j in row_ptr[row]..row_ptr[row+1] {
              col = col_idx[j];
              val = values[j];
              for k in 0..B.cols {
                  Y[row, k] += val * B[col, k];
              }
          }
      }
  }
  ```
  - O(nnz × B.cols)
  - No SIMD, no blocking (reference only)

- **COO SpMM**:
  - Iterate (row, col, val) triplets
  - Accumulate into Y[row]
  - Requires sorted COO for cache locality

- **BlockCSR SpMM**:
  - Load block (B×B dense)
  - Matmul block @ B_dense slice
  - Accumulate into Y

- **N:M SpMM**:
  - Decode metadata → indices
  - Gather values
  - Standard sparse multiply

#### SddMM (Sampled Dense @ Dense → Sparse)
**Critical for DSnoT, RigL gradients**

```rust
fn sddmm(A_dense, B_dense, mask: SparseMask) -> C_sparse {
    // Compute (A @ B) only at mask positions
    for (row, col) in mask.active_positions() {
        C[row, col] = dot(A[row, :], B[:, col]);
    }
}
```

**Optimization**:
- Thread-parallel over active positions
- Cache-block B for locality
- Return in CSR format directly

**Test coverage**:
```rust
#[test] fn spmm_matches_dense()  // SpMM(A_sparse, B) == matmul(A_dense, B)
#[test] fn sddmm_matches_masked() // SddMM == (A@B) sampled
proptest! { spmm_correctness(shape, sparsity) }
```

**Success criteria**:
- SpMM matches dense reference (ε < 1e-5)
- SddMM computes only nnz values correctly
- All formats supported
- Tests pass

---

### 1.3 Neural Network Layer (Day 7-9)

**Why**: Users need SparseLinear to actually use this.

#### Files:
```
src/nn/linear.rs   — SparseLinear module
src/nn/convert.rs  — Dense ↔ Sparse conversion
src/nn/init.rs     — Sparse initialization
src/nn/mod.rs
```

**Deliverables**:

#### SparseLinear
```rust
#[derive(Module)]
pub struct SparseLinear<B: Backend> {
    pub weight: Param<SparseTensor<B>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub mask: SparseMask<B>,
    config: SparseLinearConfig,
}

impl<B: Backend> SparseLinear<B> {
    // Primary API: create from existing Linear
    pub fn from_linear(
        linear: Linear<B>,
        mask: SparseMask<B>,
        format: SparseFormat,
    ) -> Self;

    // Forward pass
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;

    // Convert back to dense
    pub fn to_dense(&self) -> Linear<B>;

    // Update mask (for RigL)
    pub fn update_mask(&mut self, new_mask: SparseMask<B>);
}
```

#### Autodiff Integration (CRITICAL)
**This is the hard part**. Two approaches:

**Approach A: Decompose to dense (simple, works now)**
```rust
fn forward_autodiff(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let w_dense = self.weight.to_dense(); // Materialize
    w_dense.matmul(x) + bias
}
```
- ✅ Works with existing autodiff
- ❌ Memory overhead (dense weight)
- ❌ Slow backward (dense gradients)

**Approach B: Custom backward (optimal, requires Burn internals)**
```rust
// Register custom backward for sparse matmul
// Forward: Y = W_sparse @ X
// Backward:
//   ∂L/∂W_sparse = SddMM(∂L/∂Y, X^T, mask)  // Sparse grad!
//   ∂L/∂X = W_sparse^T @ ∂L/∂Y
```
- ✅ Sparse gradients (memory efficient)
- ✅ Fast (SddMM is O(nnz))
- ❌ Requires hooking Burn's autodiff

**Decision**: Start with Approach A, implement Approach B in Phase 2 after studying `burn-autodiff` internals.

**Test coverage**:
```rust
#[test] fn sparse_linear_forward_matches_dense()
#[test] fn sparse_linear_backward_matches_dense() // At active positions
#[test] fn from_linear_preserves_weights()
```

**Success criteria**:
- Forward pass matches dense Linear
- Gradients flow correctly (dense decomposition OK for now)
- Can create from existing Linear + mask
- Can convert back to dense

---

### 1.4 Sparse Optimizer (Day 10-12)

**Why**: Need to handle sparse gradients, momentum for dynamic sparsity.

#### Files:
```
src/optim/adapter.rs  — SparseOptimizer wrapper
src/optim/state.rs    — Sparse state management
src/optim/transfer.rs — Momentum transfer (for mask updates)
src/optim/mod.rs
```

**Deliverables**:

#### SparseOptimizer
```rust
pub struct SparseOptimizer<B: AutodiffBackend, O: Optimizer<B>> {
    inner: O,
    sparse_states: HashMap<ParamId, SparseOptimizerState<B>>,
}

struct SparseOptimizerState<B: Backend> {
    mask: SparseMask<B>,
    moment1: Option<Tensor<B, 1>>, // [n_active] only
    moment2: Option<Tensor<B, 1>>,
}
```

**Functionality**:
1. **Sparse gradient handling**:
   - Intercept gradients for sparse params
   - Extract only active positions: `grad_active = mask.gather_active(grad)`
   - Update moments for active only
   - Scatter back: `grad_full = mask.scatter_active(grad_active)`

2. **Momentum transfer** (for RigL, DSnoT):
   ```rust
   pub fn update_mask(&mut self, param_id: ParamId, new_mask: SparseMask<B>) {
       let old_active = old_mask.active_indices;
       let new_active = new_mask.active_indices;
       let overlap = old_active ∩ new_active;

       // Transfer moments for overlapping weights
       m_new[overlap] = m_old[overlap];
       // Initialize new weights
       m_new[new \ overlap] = 0;
   }
   ```

**Test coverage**:
```rust
#[test] fn sparse_adam_step_matches_dense_at_active()
#[test] fn momentum_transfer_preserves_overlap()
#[test] fn newly_grown_weights_have_zero_momentum()
```

**Success criteria**:
- Optimizer updates only active positions
- Momentum transfer works correctly
- Works with Adam, SGD, AdamW

---

### 1.5 End-to-End Test (Day 13-14)

**Why**: Validates entire pipeline works.

#### Files:
```
tests/e2e_mnist.rs
tests/e2e_correctness.rs
```

**Test cases**:

1. **MNIST Wanda pruning**:
   ```rust
   // Train dense model
   let model = MnistModel::new();
   train(model, 5 epochs); // → 98% accuracy

   // Prune with Wanda
   let mask = Wanda::new(config).prune(&model.linear.weight, &calib_data);
   let sparse_linear = SparseLinear::from_linear(model.linear, mask, CSR);

   // Evaluate sparse model
   let accuracy = eval(sparse_model);
   assert!(accuracy > 90%); // Reasonable accuracy retention
   ```

2. **Gradient correctness**:
   ```rust
   // Forward-backward through SparseLinear must match Dense
   let loss_sparse = sparse_linear.forward(x).mse(target);
   let loss_dense = dense_linear.forward(x).mse(target);
   assert_close(loss_sparse, loss_dense);

   let grad_sparse = loss_sparse.backward();
   let grad_dense = loss_dense.backward();
   assert_close(grad_sparse[active], grad_dense[active]);
   ```

3. **Format conversion round-trips**:
   ```rust
   proptest! {
       fn all_format_roundtrips(tensor: Tensor, format: SparseFormat) {
           let sparse = SparseTensor::from_dense(tensor, format);
           let back = sparse.to_dense();
           assert_tensors_close(tensor, back, 1e-6);
       }
   }
   ```

**Success criteria**:
- MNIST Wanda pipeline runs end-to-end
- Sparse model accuracy within 5% of dense
- All property tests pass
- No panics, no NaNs

---

## Phase 2: Performance (Week 3-4) — Make It Fast

**Goal**: Beat cuSPARSE on CUDA. Optimized kernels. Production-ready.

### 2.1 CUDA Infrastructure (Day 1-3)

#### Setup cuSPARSE FFI
```
src/kernel/cuda/ffi.rs        — cuSPARSE bindings
src/kernel/cuda/cusparse.rs   — Wrapper functions
```

**Approach**: Use `cubecl` backend, FFI to cuSPARSE for SpMM.

```rust
// FFI declarations
extern "C" {
    fn cusparseSpMM(
        handle: cusparseHandle_t,
        opA: cusparseOperation_t,
        opB: cusparseOperation_t,
        alpha: *const f32,
        matA: cusparseSpMatDescr_t,
        matB: cusparseDnMatDescr_t,
        beta: *const f32,
        matC: cusparseDnMatDescr_t,
        // ...
    ) -> cusparseStatus_t;
}
```

**Wrapper**:
```rust
pub fn csr_spmm_cusparse<R: CubeRuntime>(
    values: &CubeTensor<R>,
    col_indices: &CubeTensor<R>,
    row_ptrs: &CubeTensor<R>,
    dense_b: &CubeTensor<R>,
    shape: [usize; 2],
) -> CubeTensor<R> {
    // Create cuSPARSE descriptors
    // Call cusparseSpMM
    // Return result
}
```

**Test**: CUDA SpMM must match CPU reference.

---

### 2.2 Custom CUDA Kernels (Day 4-8)

**Why**: Beat cuSPARSE at specific workloads.

#### BlockCSR Kernel (Tensor Core Optimized)
```
src/kernel/cuda/block_csr.cu
```

**Strategy**:
- 16×16 or 32×32 blocks
- Load entire block into shared memory
- Use tensor cores (WMMA/MMA instructions)
- Fused multiply-accumulate

**Target**: 1.5× faster than cuSPARSE CSR at 80% sparsity.

#### N:M (2:4) Kernel
```
src/kernel/cuda/nm_sparse.cu
```

**Strategy**:
- Use NVIDIA sparse tensor cores directly
- `mma.sp.sync` instructions
- Decode 2:4 metadata on-the-fly

**Target**: 2× speedup over dense on A100.

#### SddMM Kernel
```
src/kernel/cuda/sddmm.cu
```

**Why**: cuSPARSE doesn't have this. Critical for DSnoT/RigL.

**Strategy**:
- Thread per sparse position
- Coalesced reads from A, B
- Write to CSR output directly

**Target**: Faster than dense matmul + masking.

**Test all kernels**:
- Correctness vs CPU
- Performance vs cuSPARSE
- Numerical stability (FP16, FP32, BF16)

---

### 2.3 Benchmarking Suite (Day 9-10)

#### Files:
```
benches/spmm_cpu.rs
benches/spmm_cuda.rs
benches/format_conversion.rs
benches/e2e_training.rs
```

**Benchmark matrix**:
| Kernel | Format | Size | Sparsity | Target |
|--------|--------|------|----------|--------|
| SpMM | CSR | 1024² | 50% | 1.2× dense |
| SpMM | CSR | 1024² | 80% | 2.5× dense |
| SpMM | BlockCSR | 4096² | 80% | 1.5× cuSPARSE |
| SpMM | N:M 2:4 | 1024² | 50% | 2× dense |
| SddMM | Mask | 1024² | 80% | 3× dense+mask |

**Metrics**:
- Throughput (GFLOP/s)
- Memory bandwidth utilization
- vs cuSPARSE (speedup ratio)
- vs dense (speedup ratio)

**Success criteria**:
- CUDA CSR within 10% of cuSPARSE
- BlockCSR beats cuSPARSE at high sparsity
- N:M 2:4 achieves 1.8×+ on A100
- No regression vs CPU (correctness)

---

## Phase 3: Dynamic Sparsity (Week 5-6) — Complete Methods

**Goal**: RigL, MEST working. Full dynamic sparse training.

### 3.1 Dynamic Methods (Day 1-4)

#### RigL
```
src/methods/dynamic/rigl.rs
```

**Implementation**:
```rust
pub struct RigL<B: AutodiffBackend> {
    config: RigLConfig,
    mask: SparseMask<B>,
    grad_accumulator: Option<Tensor<B, 2>>,
    step_count: usize,
}

impl<B: AutodiffBackend> RigL<B> {
    pub fn update_mask(&mut self, gradients: &Tensor<B, 2>) -> SparseMask<B> {
        // Accumulate |∇W|
        self.grad_accumulator += gradients.abs();

        if self.step_count % self.config.update_freq == 0 {
            // Drop: bottom-k active by gradient
            let active_grads = self.mask.gather_active(&self.grad_accumulator);
            let k = (self.config.drop_fraction * n_active) as usize;
            let to_drop = bottomk_indices(&active_grads, k);

            // Grow: top-k pruned by gradient
            let pruned_grads = self.mask.gather_pruned(&self.grad_accumulator);
            let to_grow = topk_indices(&pruned_grads, k);

            // Swap
            self.mask.swap(to_drop, to_grow);
            self.grad_accumulator = None;
        }

        self.mask.clone()
    }
}
```

#### MEST (Magnitude-Entropy Sparse Training)
```
src/methods/dynamic/mest.rs
```

**Key difference from RigL**: Uses entropy of gradients, not just magnitude.

**Test**:
- MNIST RigL: train to 95%+ accuracy
- Mask changes during training
- Final sparsity matches target

---

### 3.2 Training Loop Integration (Day 5-6)

#### Example: MNIST RigL
```
examples/mnist_rigl.rs
```

```rust
// Initialize
let model = MnistModel::new();
let initial_mask = SparseMask::random(shape, 0.8); // 80% sparse
let mut sparse_linear = SparseLinear::from_mask(d_in, d_out, initial_mask, CSR);
let mut rigl = RigL::new(rigl_config, initial_mask);
let mut optimizer = SparseOptimizer::new(Adam::new(adam_config));

// Training loop
for epoch in 0..10 {
    for (step, batch) in dataloader.enumerate() {
        // Forward
        let output = model.forward(batch.input);
        let loss = criterion(output, batch.target);

        // Backward
        let grads = loss.backward();

        // Update mask (every 100 steps)
        if step % 100 == 0 {
            let weight_grad = grads.get(&sparse_linear.weight);
            let new_mask = rigl.update_mask(weight_grad);
            sparse_linear.update_mask(new_mask.clone());
            optimizer.update_mask(sparse_linear.weight.id(), new_mask);
        }

        // Optimizer step
        optimizer.step();
    }

    println!("Epoch {}: loss={}, sparsity={}", epoch, loss, sparse_linear.sparsity());
}
```

**Success criteria**:
- RigL converges to 95%+ accuracy
- Mask changes during training
- Final model is sparse and accurate

---

## Phase 4: Advanced Features (Week 7-8)

### 4.1 Additional Methods
- **SNIP**: Gradient-based one-shot pruning
- **GraSP**: Hessian-gradient product
- **SET**: Sparse Evolutionary Training (random grow/prune)
- **Lottery Ticket Hypothesis**: Iterative magnitude pruning + rewind

### 4.2 SparseConv2d
```
src/nn/conv.rs
```

**Challenge**: 4D tensors, spatial sparsity patterns.

**Approach**: Reshape to 2D (spatial → batch dim), use SparseLinear.

### 4.3 Mixed Precision
- FP16/BF16 sparse tensors
- Automatic mixed precision for sparse training
- Loss scaling

### 4.4 Distributed Training
- Sparse all-reduce
- Gradient synchronization with different masks
- Sharded sparse tensors

---

## Testing Strategy (Continuous)

### Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn format_conversion_preserves_values(
        shape in prop::array::uniform2(10..100),
        sparsity in 0.1f32..0.9,
    ) {
        let dense = random_tensor(shape);
        let sparse = SparseTensor::from_dense(&dense, CSR, 0.0);
        let back = sparse.to_dense();
        assert_tensors_close(&dense, &back, 1e-6);
    }

    #[test]
    fn spmm_matches_dense(
        n in 10..100, m in 10..100, k in 10..100,
        sparsity in 0.1..0.9,
    ) {
        let A_dense = random_tensor([n, m]);
        let B = random_tensor([m, k]);

        let A_sparse = SparseTensor::from_dense(&A_dense, CSR, sparsity);
        let Y_sparse = kernel::spmm(&A_sparse, &B);
        let Y_dense = A_dense.matmul(B);

        assert_tensors_close(&Y_sparse, &Y_dense, 1e-5);
    }
}
```

### Correctness Tests
- All format conversions round-trip correctly
- SpMM matches dense reference
- SddMM computes masked values correctly
- Gradients match dense at active positions

### Performance Tests
- CUDA kernels beat cuSPARSE baselines
- No memory leaks
- Numerical stability (no NaNs)

---

## Success Metrics

### Phase 1 (Foundations)
- [x] All format conversions work
- [x] CPU kernels match dense reference (ε < 1e-5)
- [x] SparseLinear works with autodiff
- [x] MNIST Wanda pipeline runs end-to-end
- [x] Property tests pass (1000+ random cases)

### Phase 2 (Performance)
- [x] CUDA CSR within 10% of cuSPARSE
- [x] BlockCSR beats cuSPARSE at 80%+ sparsity
- [x] N:M 2:4 achieves 2× on A100
- [x] SddMM faster than dense+mask

### Phase 3 (Dynamic)
- [x] RigL trains to 95%+ accuracy
- [x] MEST implementation complete
- [x] Mask updates work with optimizer

### Phase 4 (Advanced)
- [x] SparseConv2d working
- [x] Mixed precision support
- [x] All SOTA methods implemented

---

## Implementation Order (Strict Dependencies)

```
Week 1-2: Phase 1
├─ Day 1-3:   Format conversions (BLOCKS EVERYTHING)
├─ Day 4-6:   CPU kernels (SpMM, SddMM)
├─ Day 7-9:   SparseLinear + autodiff
├─ Day 10-12: SparseOptimizer
└─ Day 13-14: E2E tests

Week 3-4: Phase 2
├─ Day 1-3:  CUDA cuSPARSE integration
├─ Day 4-8:  Custom CUDA kernels (BlockCSR, N:M, SddMM)
└─ Day 9-10: Benchmarking suite

Week 5-6: Phase 3
├─ Day 1-4: RigL, MEST
└─ Day 5-6: Training integration

Week 7-8: Phase 4
├─ Additional methods (SNIP, GraSP, SET)
├─ SparseConv2d
└─ Advanced features
```

---

## Non-Negotiables

1. **No MVPs**: Every component is production-grade from the start
2. **Test before optimize**: CPU reference must be correct before CUDA
3. **Beat cuSPARSE**: Performance targets are mandatory, not aspirational
4. **Zero silent failures**: Panic on invalid construction, return Results on runtime failures
5. **Property tests**: Every operation has randomized property tests
6. **Documentation**: Every public API has examples and doctests

---

## Open Questions to Resolve

### Critical (Week 1)
1. **Autodiff integration**: Can we hook Burn's autodiff for custom sparse backward?
   - **Action**: Study `burn-autodiff` internals, prototype custom backward
   - **Fallback**: Use dense decomposition (Phase 1), optimize later

2. **Tensor indexing**: Does Burn support gather/scatter for sparse indices?
   - **Action**: Test `Tensor::select`, `Tensor::slice_assign`
   - **Fallback**: Use `to_data()` + manual indexing (slower)

### Important (Week 2)
3. **CUDA memory model**: How to manage pinned memory for sparse tensors?
   - **Action**: Check `cubecl` memory allocator APIs

4. **cuSPARSE version**: Which cuSPARSE version to target (11.x, 12.x)?
   - **Action**: Use latest stable (12.x)

### Future (Week 3+)
5. **WGPU sparse**: Worth implementing beyond basic CSR?
   - **Decision**: Basic CSR only, focus on CUDA

6. **Distributed training**: Defer to v0.4 or include in v1.0?
   - **Decision**: Defer, document as "future work"

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Burn autodiff doesn't support custom backward | High | Fallback to dense decomposition (Phase 1) |
| cuSPARSE API changes | Medium | Use FFI to stable API, version pinning |
| N:M kernel doesn't achieve 2× | Medium | Document as "hardware-dependent", focus on BlockCSR |
| Format conversion bugs | High | Property-based tests, extensive validation |
| Memory leaks in CUDA | High | Valgrind/compute-sanitizer in CI |

---

## Deliverables by Phase

### Phase 1 Output
- ✅ Working CPU implementation
- ✅ SparseLinear usable
- ✅ Wanda + DSnoT working
- ✅ All tests passing
- ✅ Example: MNIST pruning

### Phase 2 Output
- ✅ CUDA kernels beating cuSPARSE
- ✅ Benchmarks showing speedups
- ✅ Performance documentation

### Phase 3 Output
- ✅ RigL, MEST working
- ✅ Example: MNIST RigL training
- ✅ Dynamic sparsity docs

### Phase 4 Output
- ✅ All SOTA methods
- ✅ SparseConv2d
- ✅ Comprehensive docs
- ✅ Production-ready v1.0

---

**Total Timeline**: 8 weeks to production SOTA sparse training infrastructure.

**Philosophy**: Build it right the first time. No shortcuts. Beat cuSPARSE or go home.
