/// Backend kernel dispatch layer
///
/// This module provides:
/// - `SparseKernel` trait: Backend-specific sparse operations
/// - `KernelSupport`: Capability negotiation
/// - `SparseDispatch`: Runtime routing with fallbacks
/// - Backend implementations: CPU, CUDA, WGPU

pub mod api;
pub mod dispatch;
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// Re-exports
pub use api::{SparseKernel, KernelSupport};
pub use dispatch::{SparseDispatch, SparseConfig};
