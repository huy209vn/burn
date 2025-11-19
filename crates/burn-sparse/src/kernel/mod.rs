/// Backend kernel dispatch for sparse operations
///
/// This module provides:
/// - SparseKernel trait that all backends implement
/// - Capability-based dispatch with fallbacks
/// - Backend-specific implementations (CPU, CUDA, WGPU)

pub mod api;
pub mod cpu;
pub mod dispatch;

// Re-exports
pub use api::{KernelSupport, SparseKernel};
pub use dispatch::{SparseConfig, SparseDispatch};
