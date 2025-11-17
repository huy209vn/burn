/// Static pruning methods (one-shot, no retraining)
///
/// These methods prune based on a single pass over calibration data.
/// - Fast: no iterative optimization
/// - Simple: score → threshold → mask
/// - Effective: competitive with more complex methods at moderate sparsity

pub mod wanda;

// Future: magnitude, snip, grasp
// pub mod magnitude;
// pub mod snip;
