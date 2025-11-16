#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # burn-sparse
//!
//! Sparse training and pruning toolkit for Burn.
//!
//! ## Architecture
//!
//! - **Primitives**: Core infrastructure (masks, stats, calibration)
//! - **Methods**: Stable pruning algorithms (Wanda, DSnoT)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use burn_sparse::prelude::*;
//!
//! // Simple Wanda pruning
//! let mask = Wanda::new(config).prune(&weights, &data);
//!
//! // Refine with DSnoT
//! let refined = DSnoT::new(config).refine(&weights, &mask, &data);
//! ```
//!
//! ## Examples
//!
//! See the repository examples for complete usage patterns.

extern crate alloc;

/// Core primitives for sparse operations
pub mod primitives;

/// Stable pruning methods
pub mod methods;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::primitives::*;
    pub use crate::methods::*;
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
