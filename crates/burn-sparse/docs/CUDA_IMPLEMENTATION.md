# burn-sparse CUDA Implementation Strategy

## Architecture Decision: Hybrid Approach

### Phase 1: cuSPARSE Baseline (Week 1)

**Use cuSPARSE via FFI for standard operations:**
- CSR SpMM → `cusparseSpmm()`
- CSR → Dense → `cusparseCsr2dense()`
- Format conversions → cuSPARSE utilities

**Why:**
- Battle-tested, heavily optimized
- Gives us immediate GPU performance
- Validates correctness
- Establishes baseline to beat

**Implementation:**
```rust
// burn-sparse/src/kernel/cuda/cusparse.rs
pub fn csr_spmm_cusparse<R: CubeRuntime>(
    values: &CubeTensor<R>,      // [nnz]
    col_indices: &CubeTensor<R>, // [nnz]
    row_ptrs: &CubeTensor<R>,    // [n_rows+1]
    dense_b: &CubeTensor<R>,     // [m, k]
    output: &mut CubeTensor<R>,  // [n, k]
) -> Result<(), SparseError> {
    unsafe {
        // Call cuSPARSE via FFI
        cusparseSpMM(
            handle,
            // ... parameters
        );
    }
}
```

### Phase 2: Custom Kernels (Weeks 2-3)

**Write custom CUDA kernels where we can beat cuSPARSE:**

#### 2.1 BlockCSR Kernel (Beat cuSPARSE at high sparsity)
```cuda
// burn-sparse/src/kernel/cuda/block_csr.cu
__global__ void block_csr_spmm_kernel(
    const float* __restrict__ blocks,      // [n_blocks, B*B]
    const int* __restrict__ block_col_idx, // [n_blocks]
    const int* __restrict__ block_row_ptr, // [n_block_rows+1]
    const float* __restrict__ dense_b,     // [m, k]
    float* __restrict__ output,            // [n, k]
    int n_block_rows,
    int block_size
) {
    // Each block handles one block-row
    int block_row = blockIdx.x;
    int tid = threadIdx.x;

    int start = block_row_ptr[block_row];
    int end = block_row_ptr[block_row + 1];

    for (int i = start; i < end; i++) {
        int block_col = block_col_idx[i];
        const float* block_ptr = blocks + i * block_size * block_size;

        // Load block into shared memory
        __shared__ float smem_block[16][16];
        // ... load with coalescing

        // Matrix-multiply block × dense_b[block_col*block_size : (block_col+1)*block_size, :]
        // Use wmma::mma_sync for tensor cores if block_size == 16
    }
}
```

**Target**: 20-30% faster than cuSPARSE for 80%+ sparsity

#### 2.2 SddMM Kernel (No cuSPARSE equivalent)
```cuda
// Sampled Dense-Dense MatMul: critical for backprop
__global__ void sddmm_kernel(
    const float* __restrict__ a,     // [n, m] dense
    const float* __restrict__ b,     // [m, k] dense
    const int* __restrict__ mask_row_idx, // [nnz]
    const int* __restrict__ mask_col_idx, // [nnz]
    float* __restrict__ output,      // [nnz] sparse output
    int nnz
) {
    // Thread-per-nonzero
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = mask_row_idx[idx];
    int col = mask_col_idx[idx];

    // Dot product: a[row, :] · b[:, col]
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        sum += a[row * m + i] * b[i * k + col];
    }
    output[idx] = sum;
}
```

**This is critical** - cuSPARSE has no SddMM, but we need it for backprop.

#### 2.3 N:M (2:4) Kernel (Use sparse tensor cores)
```cuda
// Use mma.sp.sync instructions (A100/H100 only)
__global__ void nm_24_spmm_kernel(...) {
    // Decode 2:4 metadata
    // Use sparse tensor core MMA instructions
    // 2× throughput vs dense
}
```

### Phase 3: cubecl Integration (Week 4)

**Use cubecl for non-critical operations:**
- Format validation
- Mask → CSR construction (if CPU-side is slow)
- Element-wise ops

**Why not cubecl everywhere?**
- SpMM is bandwidth-bound, needs hand-tuned memory access
- cuSPARSE is already optimal for standard CSR
- Custom kernels give us edge where needed

### Phase 4: Dispatch Layer

```rust
// burn-sparse/src/kernel/cuda/mod.rs
impl<R: CubeRuntime> SparseKernel<CubeBackend<R>> for CudaKernel {
    fn spmm(a: &SparseTensor<..>, b: &Tensor<..>) -> Result<Tensor<..>> {
        match a.format() {
            SparseFormat::CSR => {
                // Use cuSPARSE baseline
                csr_spmm_cusparse(a.data(), b)?
            }
            SparseFormat::BlockCSR { block_size: 16 } => {
                // Use custom tensor core kernel
                block_csr_spmm_kernel(a.data(), b)?
            }
            SparseFormat::NInM { n: 2, m: 4 } => {
                // Use sparse tensor core kernel
                nm_24_spmm_kernel(a.data(), b)?
            }
            _ => {
                // Convert to CSR, then use cuSPARSE
                let csr = a.to_format(SparseFormat::CSR)?;
                Self::spmm(&csr, b)?
            }
        }
    }
}
```

## Implementation Roadmap

### Week 1: cuSPARSE Baseline
- [ ] FFI bindings to cuSPARSE (cusparse-sys or manual)
- [ ] CSR SpMM via cusparseSpmm
- [ ] Memory management (allocate on GPU, transfer data)
- [ ] Benchmark vs dense matmul
- **Deliverable**: Working CSR SpMM on CUDA

### Week 2: Custom BlockCSR
- [ ] CUDA kernel for block-sparse matmul
- [ ] Tensor core integration (wmma API)
- [ ] Tune block size (16×16 best for FP16 tensor cores)
- [ ] Benchmark vs cuSPARSE
- **Deliverable**: BlockCSR beats cuSPARSE for 80%+ sparsity

### Week 3: SddMM + Autodiff
- [ ] SddMM CUDA kernel
- [ ] Integrate with Burn autodiff
- [ ] Sparse gradient flow
- **Deliverable**: Sparse backprop working

### Week 4: N:M Sparse Tensor Cores
- [ ] 2:4 metadata encoding
- [ ] mma.sp.sync kernel
- [ ] Projection: arbitrary mask → 2:4 pattern
- **Deliverable**: 2:4 runs at 2× dense throughput

## FFI Strategy

### Option A: cusparse-sys crate (if it exists)
Check if there's already a cuSPARSE binding:
```bash
cargo search cusparse
```

### Option B: Manual FFI
```rust
// burn-sparse/src/kernel/cuda/ffi.rs
#[link(name = "cusparse")]
extern "C" {
    fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
    fn cusparseSpMM(...) -> cusparseStatus_t;
    // ... other functions
}
```

### Option C: C++ Wrapper + bindgen
For complex kernels, write C++ wrapper, generate bindings:
```cpp
// sparse_kernels.cu
extern "C" void csr_spmm_wrapper(...) {
    // Call kernel
}
```

```rust
// build.rs
bindgen::Builder::default()
    .header("sparse_kernels.h")
    .generate()?;
```

## Next Immediate Steps

1. **Check for existing cuSPARSE bindings** (10 min)
2. **Create burn-sparse/src/kernel/cuda/** structure (10 min)
3. **Prototype CSR SpMM with cuSPARSE** (2 hours)
4. **Benchmark vs dense** (30 min)
5. **Validate correctness** (30 min)

If all goes well, we'll have working CUDA SpMM today.
