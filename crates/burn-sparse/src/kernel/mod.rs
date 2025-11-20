/// Sparse operation dispatch
///
/// For now, all operations use dense fallback.
/// Real sparse kernels will be implemented with CubeCL for GPU acceleration.

pub mod dispatch;

// Re-exports
pub use dispatch::{SparseConfig, SparseDispatch};
