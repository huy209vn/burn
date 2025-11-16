//! Stable pruning methods.
//!
//! This module contains production-ready pruning algorithms:
//! - [`Wanda`]: Activation-weighted magnitude pruning
//! - [`DSnoT`]: Variance-weighted iterative mask refinement

mod wanda;
mod dsnot;

pub use wanda::{Wanda, WandaConfig};
pub use dsnot::{DSnoT, DSnoTConfig};
