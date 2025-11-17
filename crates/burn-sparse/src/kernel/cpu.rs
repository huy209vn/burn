/// CPU reference implementation of sparse kernels
///
/// This is the baseline, not optimized for performance.
/// Serves as correctness reference and fallback.

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseTensor, SparseFormat, SparseResult, SparseError};
use super::api::{SparseKernel, KernelSupport};

/// CPU sparse kernel implementation
pub struct CpuKernel;

impl<B: Backend> SparseKernel<B> for CpuKernel {
    fn spmm(
        a: &SparseTensor<B>,
        b: &Tensor<B, 2>,
    ) -> SparseResult<Tensor<B, 2>> {
        // TODO: Implement CPU SpMM
        Err(SparseError::UnsupportedFormat {
            backend: "CPU".to_string(),
            format: a.format(),
        })
    }

    fn sddmm(
        a: &Tensor<B, 2>,
        b: &Tensor<B, 2>,
        mask: &SparseTensor<B>,
    ) -> SparseResult<SparseTensor<B>> {
        // TODO: Implement
        Err(SparseError::InvalidOperation("CPU SddMM not implemented".to_string()))
    }

    fn to_format(
        a: &SparseTensor<B>,
        target: SparseFormat,
    ) -> SparseResult<SparseTensor<B>> {
        // Delegate to core::convert
        crate::core::convert::convert_format(a, target)
    }

    fn to_dense(a: &SparseTensor<B>) -> Tensor<B, 2> {
        // Delegate to core::convert
        crate::core::convert::to_dense(a)
    }

    fn supports(format: SparseFormat) -> KernelSupport {
        match format {
            // CPU supports CSR and COO natively (eventually)
            SparseFormat::CSR => KernelSupport::Unsupported, // TODO: Implement
            SparseFormat::COO => KernelSupport::Unsupported, // TODO: Implement

            // Everything else converts to CSR
            _ => KernelSupport::SupportedWithConversion(SparseFormat::CSR),
        }
    }
}
