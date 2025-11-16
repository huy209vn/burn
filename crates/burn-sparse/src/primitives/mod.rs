//! Core primitives for sparse operations.
//!
//! This module provides the fundamental building blocks for sparse training:
//! - [`SparseMask`]: Binary sparsity masks with efficient indexing
//! - [`ActivationStats`]: Statistical moments of layer activations
//! - [`CalibrationData`]: Wrapper for calibration sample management
//! - [`utils`]: Utility functions for tensor operations

mod mask;
mod stats;
mod calibration;
mod utils;

pub use mask::SparseMask;
pub use stats::ActivationStats;
pub use calibration::CalibrationData;
pub use utils::*;
