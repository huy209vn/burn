//! Training methods for sparse neural networks
//!
//! This module contains:
//! - **Static pruning**: One-shot methods (Wanda, Magnitude, SNIP)
//! - **Iterative refinement**: Mask optimization (DSnoT)
//! - **Dynamic training**: Training with evolving masks (RigL, MEST)

pub mod static_pruning;
pub mod iterative;

#[cfg(feature = "dynamic")]
pub mod dynamic;

// Legacy compat - temporary, will be removed
#[doc(hidden)]
pub use static_pruning::wanda::{Wanda, WandaConfig};
#[doc(hidden)]
pub use iterative::dsnot::{DSnoT, DSnoTConfig};
