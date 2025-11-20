//! Autodiff-aware sparse operations
//!
//! This module provides sparse operations that properly register backward passes
//! in Burn's autodiff computational graph. These operations work with both regular
//! backends and autodiff backends.

mod sparse_matmul;

pub use sparse_matmul::*;
