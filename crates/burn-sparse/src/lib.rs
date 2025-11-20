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
//! ### One-Shot Pruning (Wanda)
//!
//! ```rust,ignore
//! use burn_sparse::prelude::*;
//!
//! // Collect calibration data from your dataset
//! let calibration_data = CalibrationData::from_samples(activation_samples);
//!
//! // Configure and run Wanda pruning
//! let config = WandaConfig {
//!     sparsity: 0.9,           // 90% sparsity
//!     n_calibration: 128,      // Calibration samples
//!     use_l2: true,            // L2 norm for activations
//! };
//! let mut wanda = Wanda::new(config);
//! let sparse_mask = wanda.prune(&weights, &calibration_data);
//!
//! // Apply mask to weights
//! let sparse_weights = sparse_mask.apply(&weights);
//! ```
//!
//! ### Dynamic Sparse Training (RigL)
//!
//! ```rust,ignore
//! use burn_sparse::methods::dynamic::rigl::{RigL, RigLConfig};
//!
//! // Initialize RigL
//! let config = RigLConfig {
//!     sparsity: 0.9,           // 90% sparsity
//!     update_frequency: 100,   // Update mask every 100 steps
//!     drop_fraction: 0.3,      // Drop 30% of active weights
//! };
//! let mut rigl = RigL::new(config, initial_mask);
//!
//! // In training loop:
//! for step in 0..total_steps {
//!     let loss = forward_and_loss(&model, &batch);
//!     let grads = loss.backward();
//!
//!     // Update mask with RigL
//!     if step % config.update_frequency == 0 {
//!         let new_mask = rigl.update_mask(&weights, &gradients);
//!         // Apply new mask to model
//!     }
//!
//!     optimizer.step(&grads);
//! }
//! ```
//!
//! ### Dynamic Sparse Training (MEST)
//!
//! ```rust,ignore
//! use burn_sparse::methods::dynamic::mest::{Mest, MestConfig};
//!
//! // Initialize MEST with elastic decay
//! let config = MestConfig {
//!     sparsity: 0.9,
//!     mutation_rate_init: 0.3, // Start with 30% mutation
//!     mutation_rate_final: 0.05, // Decay to 5%
//!     lambda: 0.01,            // Gradient weight in salience
//!     use_gradient_ema: true,  // Smooth gradients with EMA
//!     ema_decay: 0.9,
//! };
//! let mut mest = Mest::new(config, initial_mask, total_steps);
//!
//! // MEST updates every step (no update_frequency)
//! for step in 0..total_steps {
//!     let loss = forward_and_loss(&model, &batch);
//!     let grads = loss.backward();
//!
//!     let new_mask = mest.update_mask(&weights, &gradients);
//!     optimizer.step(&grads);
//! }
//! ```
//!
//! ## Examples
//!
//! See `examples/` directory for complete working code:
//! - `rigl_training.rs` - RigL dynamic sparse training
//! - `mest_training.rs` - MEST with elastic decay
//! - `wanda_pruning.rs` - One-shot pruning workflow
//!
//! Run with: `cargo run --example <name>`
//!
//! ## Implementation Status
//!
//! - ✅ **Core Infrastructure**: Formats (CSR/COO/Mask), conversions, validation
//! - ✅ **Static Pruning**: Wanda, Magnitude
//! - ✅ **Iterative Refinement**: DSnoT
//! - ✅ **Dynamic Training**: RigL, MEST
//! - ✅ **Neural Network**: SparseLinear with autodiff
//! - ✅ **Cross-backend**: NdArray, CUDA compatibility
//! - ⏳ **GPU Kernels**: Will use CubeCL (planned)

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
/// - Dense fallback for now
/// - Real sparse kernels will use CubeCL
pub mod kernel;

/// Neural network modules
///
/// - SparseLinear
/// - Dense ↔ Sparse conversions
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
        ActivationStats, CalibrationData, SparseError, SparseFormat, SparseMask, SparseParam,
        SparseResult, SparseTensor, SparseTensorData,
    };

    // Methods
    pub use crate::methods::dynamic::{Mest, MestConfig, RigL, RigLConfig};
    pub use crate::methods::iterative::{DSnoT, DSnoTConfig};
    pub use crate::methods::static_pruning::{Magnitude, MagnitudeConfig, Wanda, WandaConfig};

    // Kernel
    pub use crate::kernel::{SparseConfig, SparseDispatch};

    // Neural network
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
