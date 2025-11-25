//! Backend implementations of SparseBackend trait
//!
//! NOTE: Backend implementations will be added in separate crates:
//! - burn-ndarray will implement SparseBackend for NdArray
//! - burn-cuda will implement SparseBackend for Cuda
//! - burn-wgpu will implement SparseBackend for Wgpu
//! - burn-autodiff will implement SparseBackend for Autodiff<B>
//!
//! This module is a placeholder for documentation purposes.
//!
//! # Example Implementation (for backend crates)
//!
//! ```rust,ignore
//! use burn_sparse::backend::SparseBackend;
//!
//! impl SparseBackend for NdArray<f32> {
//!     type B = Self;
//!
//!     fn name() -> &'static str {
//!         "NdArray"
//!     }
//!
//!     // All other methods use default implementations (dense fallback)
//!     // Override specific methods for optimized kernels
//! }
//! ```
