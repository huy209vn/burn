#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # burn-sparse v2.0
//!
//! **Official sparse tensor and sparse training infrastructure for Burn.**
//!
//! Not just a pruning library - this is Burn's sparse subsystem.
//!
//! ## Core Capabilities
//!
//! - **Sparse Tensor Formats**: Mask, CSR, CSC, COO, BlockCSR, N:M (2:4, 4:8)
//! - **Backend Kernels**: CPU (reference), CUDA (planned), WGPU (planned)
//! - **Sparse Operations**: SpMM, SddMM, element-wise, format conversions
//! - **Training Methods**: Static (Wanda, Magnitude), Iterative (DSnoT), Dynamic (RigL, MEST)
//! - **Neural Network**: SparseLinear with autodiff support
//!
//! ## Architecture
//!
//! ```text
//! burn-sparse/
//! ├── core/           # Foundation (formats, tensors, masks)
//! ├── kernel/         # Backend dispatch (CPU/CUDA/WGPU)
//! ├── nn/             # Neural network modules
//! ├── optim/          # Sparse optimizers
//! ├── methods/        # Training algorithms
//! │   ├── static_pruning/     # One-shot pruning (Wanda, Magnitude)
//! │   ├── iterative/  # Mask refinement (DSnoT)
//! │   └── dynamic/    # Dynamic training (RigL, MEST)
//! └── experimental/   # Research features
//! ```
//!
//! ## Design Principles
//!
//! - **Infrastructure-first**: Methods build on solid foundation
//! - **Format-polymorphic**: Multiple sparse formats without kernel rewrites
//! - **Backend-independent**: CPU/CUDA/WGPU implement what they can
//! - **Zero orchestration**: Users compose methods themselves
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use burn_sparse::prelude::*;
//!
//! // 1. Prune with Wanda (static, one-shot)
//! let mask = Wanda::new(config).prune(&weights, &calibration_data);
//!
//! // 2. Refine with DSnoT (iterative, mask optimization)
//! let refined_mask = DSnoT::new(config).refine(&weights, &mask, &calibration_data);
//!
//! // 3. Convert to execution format
//! let sparse_tensor = refined_mask.to_sparse_tensor(&weights, SparseFormat::CSR);
//!
//! // 4. Use in neural network
//! let sparse_linear = SparseLinear::from_mask(d_in, d_out, refined_mask, format);
//! ```
//!
//! ## Examples
//!
//! - `pruning_cycle.rs` - Wanda + DSnoT workflow
//! - `benchmark_spmm.rs` - Sparse kernel performance
//!
//! ## Phase Status
//!
//! - ✅ **Phase 0**: Architecture design
//! - 🚧 **Phase 1**: Core + CPU (in progress)
//! - ⏳ **Phase 2**: GPU kernels (planned)
//! - ⏳ **Phase 3**: Dynamic training (planned)

extern crate alloc;

/// Core sparse tensor infrastructure
///
/// - Format definitions (CSR, COO, BlockCSR, N:M)
/// - SparseTensor type (execution format)
/// - SparseMask type (algorithm format)
/// - Format conversions, validation
pub mod core;

/// Backend kernel dispatch
///
/// - SparseKernel trait
/// - Capability routing
/// - CPU/CUDA/WGPU implementations
pub mod kernel;

/// Neural network modules
///
/// - SparseLinear
/// - Dense ↔ Sparse conversions
#[cfg(feature = "nn")]
pub mod nn;

/// Sparse optimizers
///
/// - SparseOptimizer adapter
/// - Sparse state management
/// - Momentum transfer for dynamic sparsity
#[cfg(feature = "optim")]
pub mod optim;

/// Training methods
///
/// - Static pruning (Wanda, Magnitude)
/// - Iterative refinement (DSnoT)
/// - Dynamic training (RigL, MEST)
pub mod methods;

/// Experimental features (feature-gated)
///
/// - N:M structured sparsity (2:4)
/// - BlockCSR optimization
#[cfg(feature = "experimental")]
pub mod experimental;

/// Prelude for convenient imports
pub mod prelude {
    //! Convenient re-exports for common usage
    //!
    //! ```rust,ignore
    //! use burn_sparse::prelude::*;
    //! ```

    // Core types
    pub use crate::core::{
        ActivationStats, CalibrationData, SparseError, SparseFormat, SparseMask, SparseResult,
        SparseTensor, SparseTensorData,
    };

    // Methods
    pub use crate::methods::dynamic::{Mest, MestConfig, RigL, RigLConfig};
    pub use crate::methods::iterative::{DSnoT, DSnoTConfig};
    pub use crate::methods::static_pruning::{Magnitude, MagnitudeConfig, Wanda, WandaConfig};

    // Kernel
    pub use crate::kernel::{SparseConfig, SparseDispatch};

    // Neural network (when enabled)
    #[cfg(feature = "nn")]
    pub use crate::nn::SparseLinear;
}

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "test-tch"))]
/// Backend for test cases
pub type TestBackend = burn_tch::LibTorch<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type TestBackend = burn_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type TestBackend = burn_cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type TestBackend = burn_rocm::Rocm;
