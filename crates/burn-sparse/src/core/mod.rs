/// Core sparse tensor types and operations
///
/// This module contains the foundation of burn-sparse:
/// - Format definitions (CSR, COO, N:M, etc.)
/// - SparseTensor type (execution format)
/// - SparseMask type (algorithm format)
/// - Format conversions
/// - Validation logic

pub mod calibration;
pub mod convert;
pub mod error;
pub mod format;
pub mod mask;
pub mod sparse_tensor;
pub mod stats;
pub mod utils;
pub mod validate;

// Re-exports
pub use calibration::CalibrationData;
pub use error::{SparseError, SparseResult};
pub use format::SparseFormat;
pub use mask::SparseMask;
pub use sparse_tensor::{SparseTensor, SparseTensorData};
pub use stats::ActivationStats;
