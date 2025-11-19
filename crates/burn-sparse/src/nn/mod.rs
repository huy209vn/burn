//! Neural network modules with sparse weights
//!
//! This module provides drop-in replacements for dense layers that use
//! sparse tensors for weights, enabling memory-efficient inference and training.

pub mod linear;

pub use linear::SparseLinear;
