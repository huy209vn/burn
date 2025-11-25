//! Backend trait and dispatch system for sparse operations
//!
//! # Architecture
//!
//! The sparse backend system is built on three principles:
//!
//! 1. **Format Polymorphism**: Different formats (CSR, COO, BlockCSR, N:M) for different workloads
//! 2. **Backend Independence**: Each backend (CPU, CUDA, WGPU) implements what it can
//! 3. **Graceful Degradation**: Capability negotiation + fallback, never silent failures
//!
//! # Design
//!
//! ```text
//! User Code
//!     ↓
//! SparseDispatch (capability routing)
//!     ↓
//! SparseBackend trait
//!     ↓
//! ┌─────────┬──────────┬──────────┐
//! │ NdArray │   CUDA   │   WGPU   │  (backend implementations)
//! └─────────┴──────────┴──────────┘
//! ```
//!
//! Each backend declares capabilities via `supports()`. The dispatcher:
//! - Tries native kernel if supported
//! - Tries format conversion if needed
//! - Falls back to dense if allowed
//! - Returns error otherwise
//!
//! # Example
//!
//! ```rust,ignore
//! use burn_sparse::backend::{SparseBackend, SparseDispatch, SparseConfig};
//! use burn_sparse::core::{SparseTensor, SparseFormat};
//!
//! // Create sparse tensor
//! let sparse = SparseTensor::from_dense(&dense, SparseFormat::CSR, 0.9);
//!
//! // Configure dispatch
//! let config = SparseConfig {
//!     allow_format_conversion: true,
//!     allow_dense_fallback: false,
//!     ..Default::default()
//! };
//!
//! // Dispatch SpMM (will convert format if needed)
//! let result = SparseDispatch::<B>::spmm(&sparse, &dense_b, &config)?;
//! ```

pub mod api;
pub mod dispatch;
pub mod impls;

// Re-exports
pub use api::{KernelSupport, SparseBackend};
pub use dispatch::{SparseConfig, SparseDispatch};
