/// Sparse training methods
///
/// This module provides various pruning and sparse training algorithms:
/// - Static pruning: One-shot weight removal (Wanda, Magnitude, SNIP)
/// - Iterative refinement: Mask optimization (DSnoT)
/// - Dynamic training: Mask evolution during training (RigL, MEST)

pub mod dynamic;
pub mod iterative;
pub mod static_pruning;
