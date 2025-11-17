/// Runtime dispatch and capability routing
///
/// Handles backend selection and fallback strategies for sparse operations.

use burn_core::tensor::{backend::Backend, Tensor};
use core::marker::PhantomData;

use crate::core::{SparseTensor, SparseFormat, SparseResult, SparseError};
use super::api::{SparseKernel, KernelSupport};

/// Global sparse kernel configuration
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
///
/// Routes operations to backend-specific implementations with fallback logic.
pub struct SparseDispatch<B: Backend> {
    _phantom: PhantomData<B>,
}

impl<B: Backend> SparseDispatch<B> {
    /// Dispatch SpMM with capability routing
    ///
    /// # Fallback Strategy
    /// 1. Try exact format
    /// 2. Try conversion to supported format (if allowed)
    /// 3. Try dense fallback (if allowed)
    /// 4. Error or panic (based on config)
    pub fn spmm(
        a: &SparseTensor<B>,
        b: &Tensor<B, 2>,
        config: &SparseConfig,
    ) -> SparseResult<Tensor<B, 2>> {
        // TODO: Implement actual dispatch
        // For now, return error
        Err(SparseError::UnsupportedFormat {
            backend: "unknown".to_string(),
            format: a.format(),
        })
    }

    /// Dispatch SddMM (sampled dense-dense matmul)
    pub fn sddmm(
        a: &Tensor<B, 2>,
        b: &Tensor<B, 2>,
        mask: &SparseTensor<B>,
        config: &SparseConfig,
    ) -> SparseResult<SparseTensor<B>> {
        // TODO: Implement
        Err(SparseError::InvalidOperation("SddMM not yet implemented".to_string()))
    }
}
