/// Dynamic sparse training methods
///
/// These methods evolve the sparsity mask during training:
/// - RigL: Gradient-based dynamic sparsity
/// - MEST: Momentum-based dynamic sparsity

pub mod mest;
pub mod rigl;

pub use mest::{Mest, MestConfig};
pub use rigl::{RigL, RigLConfig};
