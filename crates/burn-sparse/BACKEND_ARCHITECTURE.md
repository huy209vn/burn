# burn-sparse Backend Architecture

> **Status**: Architecture defined, no real kernel implementations yet
> **Timeline**: 30-minute design sprint (architecture only)
> **Philosophy**: Infrastructure-first. Backend-agnostic. Leave kernel implementations for later.

## Overview

The backend system provides a trait-based interface for sparse operations across different compute backends (CPU, CUDA, WGPU). Each backend declares what it can do, and the dispatcher routes operations accordingly.

## Core Design Principles

1. **Format Polymorphism**: Support multiple sparse formats (CSR, COO, BlockCSR, N:M) without kernel rewrites
2. **Backend Independence**: Each backend implements what it can, declares capabilities
3. **Graceful Degradation**: Capability negotiation + fallback, never silent failures
4. **Default = Dense Fallback**: Everything works (slowly) out of the box

## Architecture

```text
User Code
    ↓
SparseDispatch (capability routing)
    ↓
SparseBackend trait
    ↓
┌─────────┬──────────┬──────────┐
│ NdArray │   CUDA   │   WGPU   │  (backend implementations)
└─────────┴──────────┴──────────┘
```

## The SparseBackend Trait

Located in: `src/backend/api.rs`

### Core Operations

```rust
pub trait SparseBackend: Sized {
    type B: Backend;

    // === Capability Query ===
    fn supports_spmm(format: SparseFormat) -> KernelSupport;
    fn supports_sddmm(format: SparseFormat) -> KernelSupport;
    fn supports_conversion(from: SparseFormat, to: SparseFormat) -> KernelSupport;

    // === Core Sparse Operations ===

    /// Sparse-Dense Matrix Multiply: Y = A_sparse @ B
    fn spmm(
        a: &SparseTensor<Self::B>,
        b: &Tensor<Self::B, 2>,
    ) -> Tensor<Self::B, 2>;

    /// Sampled Dense-Dense Matrix Multiply (for gradients)
    fn sddmm(
        a: &Tensor<Self::B, 2>,
        b: &Tensor<Self::B, 2>,
        mask: &SparseMask<Self::B>,
        format: SparseFormat,
    ) -> SparseTensor<Self::B>;

    // === Element-wise Operations ===
    fn sparse_add(a: &SparseTensor<Self::B>, b: &SparseTensor<Self::B>) -> SparseTensor<Self::B>;
    fn sparse_mul_scalar(a: &SparseTensor<Self::B>, scalar: f32) -> SparseTensor<Self::B>;
    fn sparse_mul(a: &SparseTensor<Self::B>, b: &SparseTensor<Self::B>) -> SparseTensor<Self::B>;

    // === Format Conversions ===
    fn to_format(a: &SparseTensor<Self::B>, target: SparseFormat) -> SparseTensor<Self::B>;

    // === Backend Info ===
    fn name() -> &'static str;
    fn is_gpu() -> bool { false }
}
```

### Default Implementations

**ALL operations have default implementations that convert to dense and back.**

This means:
- ✅ Code compiles and runs immediately
- ✅ Correctness guaranteed (dense ops are well-tested)
- ✅ Backends can add optimizations incrementally
- ⚠️ Performance is bad (defeats the purpose of sparsity)

Example default implementation:

```rust
fn spmm(a: &SparseTensor<Self::B>, b: &Tensor<Self::B, 2>) -> Tensor<Self::B, 2> {
    // Dense fallback: A.to_dense() @ B
    let dense_a = a.to_dense();
    dense_a.matmul(b.clone())
}
```

## Capability System

### KernelSupport Enum

```rust
pub enum KernelSupport {
    /// Format fully supported with optimized kernel
    Supported,

    /// Format not supported, but can convert to this one
    SupportedWithConversion(SparseFormat),

    /// Format not supported at all
    Unsupported,
}
```

### How It Works

1. **Backend declares capabilities** via `supports_*()` methods
2. **Dispatcher queries** before routing operation
3. **Routing logic**:
   - `Supported` → Use native kernel
   - `SupportedWithConversion(fmt)` → Convert format, then run
   - `Unsupported` → Fall back to dense (if allowed) or error

### Example

```rust
// CPU backend supports CSR natively
impl SparseBackend for NdArrayBackend {
    fn supports_spmm(format: SparseFormat) -> KernelSupport {
        match format {
            SparseFormat::CSR => KernelSupport::Supported,
            SparseFormat::COO => KernelSupport::SupportedWithConversion(SparseFormat::CSR),
            _ => KernelSupport::Unsupported,
        }
    }

    fn spmm(a: &SparseTensor<Self::B>, b: &Tensor<Self::B, 2>) -> Tensor<Self::B, 2> {
        match &a.data {
            SparseTensorData::CSR { values, col_indices, row_pointers } => {
                // Optimized CSR SpMM implementation
                cpu_spmm_csr(values, col_indices, row_pointers, b, a.shape())
            }
            _ => {
                // Fallback to default (convert to dense)
                let dense_a = a.to_dense();
                dense_a.matmul(b.clone())
            }
        }
    }
}
```

## SparseDispatch

Located in: `src/backend/dispatch.rs`

### Configuration

```rust
pub struct SparseConfig {
    /// Allow fallback to dense if no sparse kernel available
    pub allow_dense_fallback: bool,  // Default: false (explicit is better)

    /// Allow automatic format conversion
    pub allow_format_conversion: bool,  // Default: true (safe and useful)

    /// Panic on unsupported operations (for debugging)
    pub panic_on_unsupported: bool,  // Default: false
}
```

### Usage

```rust
use burn_sparse::backend::{SparseDispatch, SparseConfig};

// Create config
let config = SparseConfig {
    allow_format_conversion: true,
    allow_dense_fallback: false,  // Error if no sparse kernel
    ..Default::default()
};

// Dispatch operation
let result = SparseDispatch::<B>::spmm(&sparse_matrix, &dense_matrix, &config)?;
```

