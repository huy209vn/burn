//! Kernel dispatch with capability routing and fallbacks
//!
//! For now, all operations fall back to dense.
//! Real sparse kernels will be implemented with CubeCL for GPU acceleration.

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseError, SparseResult, SparseTensor};

/// Configuration for sparse dispatch
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Allow fallback to dense if no sparse kernel available
    pub allow_dense_fallback: bool,

    /// Allow automatic format conversion
    pub allow_format_conversion: bool,

    /// Panic on unsupported operations (for debugging)
    pub panic_on_unsupported: bool,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            allow_dense_fallback: false, // Explicit is better
            allow_format_conversion: true, // Safe and useful
            panic_on_unsupported: false,
        }
    }
}

/// Global sparse kernel dispatcher
pub struct SparseDispatch<B: Backend> {
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> SparseDispatch<B> {
    /// Sparse-dense matrix multiply: Y = A_sparse @ B_dense
    ///
    /// For now, falls back to dense (A.to_dense() @ B).
    /// Real sparse kernels will be implemented with CubeCL for GPU.
    pub fn spmm(
        a: &SparseTensor<B>,
        b: &Tensor<B, 2>,
        _config: &SparseConfig,
    ) -> SparseResult<Tensor<B, 2>> {
        // Dense fallback: convert sparse to dense and use standard matmul
        let dense_a = a.to_dense();
        Ok(dense_a.matmul(b.clone()))
    }

    /// Sampled dense-dense matrix multiply: C_sparse = (A @ B) ⊙ mask
    ///
    /// Not yet implemented - will be added with CubeCL.
    pub fn sddmm(
        _a: &Tensor<B, 2>,
        _b: &Tensor<B, 2>,
        _mask: &SparseTensor<B>,
        _config: &SparseConfig,
    ) -> SparseResult<SparseTensor<B>> {
        Err(SparseError::UnsupportedOperation {
            operation: "sddmm".to_string(),
            backend: "dense_fallback".to_string(),
        })
    }
}
