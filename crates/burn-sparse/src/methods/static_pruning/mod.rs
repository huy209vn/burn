/// Static (one-shot) pruning methods
///
/// These methods prune weights once based on some criterion:
/// - Wanda: Activation-weighted magnitude
/// - Magnitude: Simple magnitude-based pruning
/// - SNIP: Gradient-based importance

pub mod magnitude;
pub mod wanda;

pub use magnitude::{Magnitude, MagnitudeConfig};
pub use wanda::{Wanda, WandaConfig};
