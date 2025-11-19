//! Kernel dispatch with capability routing and fallbacks

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseError, SparseFormat, SparseResult, SparseTensor};
use crate::kernel::api::{KernelSupport, SparseKernel};
use crate::kernel::cpu::CpuKernel;

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
    /// Dispatch SpMM with capability routing
    pub fn spmm(
        a: &SparseTensor<B>,
        b: &Tensor<B, 2>,
        config: &SparseConfig,
    ) -> SparseResult<Tensor<B, 2>> {
        let kernel = Self::get_kernel();

        match kernel.supports(a.format()) {
            KernelSupport::Supported => kernel.spmm(a, b),

            KernelSupport::SupportedWithConversion(target) => {
                if config.allow_format_conversion {
                    #[cfg(feature = "std")]
                    eprintln!(
                        "Warning: Converting {:?} → {:?} for SpMM",
                        a.format(),
                        target
                    );

                    let converted = a.to_format(target)?;
                    kernel.spmm(&converted, b)
                } else {
                    Err(SparseError::UnsupportedFormat {
                        backend: Self::backend_name(),
                        format: a.format(),
                    })
                }
            }

            KernelSupport::Unsupported => {
                if config.allow_dense_fallback {
                    #[cfg(feature = "std")]
                    eprintln!("Warning: No sparse kernel available, falling back to dense");

                    let dense_a = a.to_dense();
                    Ok(dense_a.matmul(b.clone()))
                } else if config.panic_on_unsupported {
                    panic!(
                        "Unsupported sparse format: {:?} on {}",
                        a.format(),
                        Self::backend_name()
                    );
                } else {
                    Err(SparseError::UnsupportedFormat {
                        backend: Self::backend_name(),
                        format: a.format(),
                    })
                }
            }
        }
    }

    /// Dispatch SddMM
    pub fn sddmm(
        a: &Tensor<B, 2>,
        b: &Tensor<B, 2>,
        mask: &SparseTensor<B>,
        config: &SparseConfig,
    ) -> SparseResult<SparseTensor<B>> {
        let kernel = Self::get_kernel();
        kernel.sddmm(a, b, mask).or_else(|_| {
            if config.panic_on_unsupported {
                panic!("SddMM not supported on {}", Self::backend_name());
            }
            Err(SparseError::UnsupportedOperation {
                operation: "sddmm".to_string(),
                backend: Self::backend_name(),
            })
        })
    }

    /// Get appropriate kernel for backend
    fn get_kernel() -> Box<dyn SparseKernel<B>> {
        // TODO: Add CUDA and WGPU support
        // For now, everything uses CPU kernel
        Box::new(CpuKernel)
    }

    /// Get backend name
    fn backend_name() -> String {
        // TODO: Get actual backend name
        "cpu".to_string()
    }
}
