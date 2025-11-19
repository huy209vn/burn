/// Sparse kernel API - Backend trait for sparse operations
///
/// This module defines the trait that all backends must implement
/// to support sparse tensor operations.

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseTensor, SparseFormat, SparseResult};

/// Backend capability for a sparse format
#[derive(Debug, Clone)]
pub enum KernelSupport {
    /// Format fully supported with optimized kernel
    Supported,

    /// Format not supported, but can convert to this one
    SupportedWithConversion(SparseFormat),

    /// Format not supported at all
    Unsupported,
}

/// Sparse kernel operations (backend-specific)
///
/// Each backend (CPU, CUDA, WGPU) implements this trait to provide
/// sparse tensor operations.
///
/// Note: All methods are non-static to make the trait object-safe
pub trait SparseKernel<B: Backend> {
    // ===== Core Operations =====

    /// Sparse-Dense Matrix Multiply: Y = A_sparse @ B
    ///
    /// # Arguments
    /// * `a` - Sparse matrix [n, m]
    /// * `b` - Dense matrix [m, k]
    ///
    /// # Returns
    /// Dense result [n, k]
    fn spmm(
        &self,
        a: &SparseTensor<B>,
        b: &Tensor<B, 2>,
    ) -> SparseResult<Tensor<B, 2>>;

    /// Sampled Dense-Dense Matrix Multiply: C_sparse = (A @ B) sampled at mask
    ///
    /// Computes dense matmul but only returns values at sparse positions.
    /// Critical for backprop through sparse weights.
    fn sddmm(
        &self,
        a: &Tensor<B, 2>,
        b: &Tensor<B, 2>,
        mask: &SparseTensor<B>,  // Used for sampling positions
    ) -> SparseResult<SparseTensor<B>>;

    // ===== Format Conversions =====

    /// Convert sparse tensor to different format
    fn to_format(
        &self,
        a: &SparseTensor<B>,
        target: SparseFormat,
    ) -> SparseResult<SparseTensor<B>>;

    /// Convert sparse to dense
    fn to_dense(&self, a: &SparseTensor<B>) -> Tensor<B, 2>;

    // ===== Capability Query =====

    /// Check if backend supports a sparse format
    fn supports(&self, format: SparseFormat) -> KernelSupport;
}
