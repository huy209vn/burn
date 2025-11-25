//! Core SparseBackend trait definition
//!
//! This trait defines the interface that all backends must implement to support
//! sparse tensor operations. Backends can implement a subset of operations and
//! declare their capabilities via the `supports_*()` methods.
//!
//! # Philosophy
//!
//! - **Construction**: Panic on invalid input (programmer error)
//! - **Kernel calls**: Return Result (runtime limitation)
//! - **Capability**: Declare what you support, dispatcher handles fallback

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseFormat, SparseMask, SparseTensor};

/// Backend capability for a specific operation + format combination
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelSupport {
    /// Format fully supported with optimized kernel
    Supported,

    /// Format not supported, but can convert to this one
    SupportedWithConversion(SparseFormat),

    /// Format not supported at all
    Unsupported,
}

/// Core trait for backend-specific sparse operations
///
/// # Implementation Guidelines
///
/// 1. **Declare Capabilities**: Use `supports_*()` to declare what formats you handle
/// 2. **Start Simple**: Implement what you can, return Unsupported for the rest
/// 3. **Optimize Later**: Correctness first, then performance
/// 4. **No Panics**: Return errors for runtime failures, panic only for bugs
///
/// # Default Implementations
///
/// Most operations have default implementations that convert to dense.
/// This lets backends start minimal and add optimizations incrementally.
pub trait SparseBackend: Sized {
    /// Backend type this sparse backend operates on
    type B: Backend;

    // ===== Capability Query =====

    /// Check if this backend supports SpMM for a given format
    ///
    /// Returns:
    /// - `Supported`: Native kernel available
    /// - `SupportedWithConversion(fmt)`: Can handle after converting to `fmt`
    /// - `Unsupported`: Cannot handle at all
    fn supports_spmm(format: SparseFormat) -> KernelSupport {
        // Default: unsupported (will use dense fallback)
        KernelSupport::Unsupported
    }

    /// Check support for SddMM (sampled dense-dense matmul)
    fn supports_sddmm(format: SparseFormat) -> KernelSupport {
        KernelSupport::Unsupported
    }

    /// Check support for format conversion
    fn supports_conversion(from: SparseFormat, to: SparseFormat) -> KernelSupport {
        // Default: all conversions route through CPU
        if from == to {
            KernelSupport::Supported
        } else {
            KernelSupport::Unsupported
        }
    }

    // ===== Core Operations =====

    /// Sparse-Dense Matrix Multiply: Y = A_sparse @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse matrix [M, K]
    /// * `b` - Dense matrix [K, N]
    ///
    /// # Returns
    ///
    /// Dense matrix Y [M, N]
    ///
    /// # Default Implementation
    ///
    /// Converts to dense and uses standard matmul. Override for performance.
    fn spmm(
        a: &SparseTensor<Self::B>,
        b: &Tensor<Self::B, 2>,
    ) -> Tensor<Self::B, 2> {
        // Dense fallback
        let dense_a = a.to_dense();
        dense_a.matmul(b.clone())
    }

    /// Sampled Dense-Dense Matrix Multiply: C_sparse = (A @ B) ⊙ mask
    ///
    /// Computes only the entries of (A @ B) that are marked in the mask.
    /// This is the gradient operation for SpMM.
    ///
    /// # Arguments
    ///
    /// * `a` - Dense matrix [M, K]
    /// * `b` - Dense matrix [K, N]
    /// * `mask` - Sparsity pattern for output [M, N]
    ///
    /// # Returns
    ///
    /// Sparse matrix C [M, N] with only masked entries
    ///
    /// # Default Implementation
    ///
    /// Dense matmul + mask application. Very slow.
    fn sddmm(
        a: &Tensor<Self::B, 2>,
        b: &Tensor<Self::B, 2>,
        mask: &SparseMask<Self::B>,
        format: SparseFormat,
    ) -> SparseTensor<Self::B> {
        // Dense fallback: compute full matmul, then apply mask
        let full_result = a.clone().matmul(b.clone());
        let masked = mask.apply(&full_result);
        SparseTensor::from_dense(&masked, format, 0.0)
    }

    // ===== Element-wise Operations =====

    /// Element-wise addition: C = A + B (same sparsity pattern)
    fn sparse_add(
        a: &SparseTensor<Self::B>,
        b: &SparseTensor<Self::B>,
    ) -> SparseTensor<Self::B> {
        // Default: convert to dense, add, convert back
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let result_dense = a_dense + b_dense;
        SparseTensor::from_dense(&result_dense, a.format(), 0.0)
    }

    /// Element-wise multiplication by scalar: C = A * s
    fn sparse_mul_scalar(
        a: &SparseTensor<Self::B>,
        scalar: f32,
    ) -> SparseTensor<Self::B> {
        // Default: convert to dense, multiply, convert back
        let a_dense = a.to_dense();
        let result_dense = a_dense * scalar;
        SparseTensor::from_dense(&result_dense, a.format(), 0.0)
    }

    /// Element-wise multiplication: C = A ⊙ B (same sparsity pattern)
    fn sparse_mul(
        a: &SparseTensor<Self::B>,
        b: &SparseTensor<Self::B>,
    ) -> SparseTensor<Self::B> {
        // Default: convert to dense, multiply, convert back
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let result_dense = a_dense * b_dense;
        SparseTensor::from_dense(&result_dense, a.format(), 0.0)
    }

    // ===== Format Conversions =====

    /// Convert sparse tensor to different format
    ///
    /// # Default Implementation
    ///
    /// Uses the SparseTensor's built-in conversion (happens on CPU).
    /// Backends can override for GPU-native conversions.
    fn to_format(
        a: &SparseTensor<Self::B>,
        target: SparseFormat,
    ) -> SparseTensor<Self::B> {
        // Default: use SparseTensor's conversion method
        // If conversion fails, panic (format constraints not met)
        a.to_format(target).expect("Format conversion failed")
    }

    // ===== Backend Information =====

    /// Get backend name (for logging and error messages)
    fn name() -> &'static str;

    /// Check if backend has GPU acceleration
    fn is_gpu() -> bool {
        false
    }
}
