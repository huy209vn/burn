//! Magnitude pruning: Simple weight magnitude-based pruning
//!
//! The simplest pruning method: keep weights with largest absolute values.

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{SparseMask, utils};

/// Configuration for magnitude pruning
#[derive(Debug, Clone)]
pub struct MagnitudeConfig {
    /// Target sparsity ratio (0.0 = dense, 1.0 = all pruned)
    pub sparsity: f32,

    /// Apply layer-wise (true) or global (false) pruning
    pub layerwise: bool,
}

impl Default for MagnitudeConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.5,
            layerwise: true,
        }
    }
}

/// Magnitude pruning: Keep top-(1-sparsity) weights by absolute value
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::methods::static_pruning::*;
///
/// let config = MagnitudeConfig {
///     sparsity: 0.5,
///     layerwise: true,
/// };
///
/// let magnitude = Magnitude::new(config);
/// let mask = magnitude.prune(&weights);
/// ```
pub struct Magnitude {
    config: MagnitudeConfig,
}

impl Magnitude {
    /// Create a new magnitude pruner
    pub fn new(config: MagnitudeConfig) -> Self {
        Self { config }
    }

    /// Prune weights by magnitude
    ///
    /// # Arguments
    /// * `weights` - Weight matrix [n_out, n_in]
    ///
    /// # Returns
    /// Binary mask indicating which weights to keep
    pub fn prune<B: Backend>(&self, weights: &Tensor<B, 2>) -> SparseMask<B> {
        let scores = weights.clone().abs();
        SparseMask::from_scores(&scores, self.config.sparsity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{Tensor, TensorData};

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_magnitude_pruning() {
        let weights = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, -3.0], [2.0, -4.0]]),
            &Default::default(),
        );

        let magnitude = Magnitude::new(MagnitudeConfig {
            sparsity: 0.5,
            layerwise: true,
        });

        let mask = magnitude.prune(&weights);

        // Should keep 2 largest: -4.0 and -3.0 (by absolute value)
        assert_eq!(mask.actual_sparsity(), 0.5);
        assert_eq!(mask.n_active(), 2);
    }
}
