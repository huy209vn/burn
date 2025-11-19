//! Wanda: Activation-weighted magnitude pruning.
//!
//! **Reference**: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024
//! https://arxiv.org/abs/2306.11695

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::{ActivationStats, CalibrationData, SparseMask};

/// Configuration for Wanda pruning.
#[derive(Debug, Clone)]
pub struct WandaConfig {
    /// Target sparsity ratio (0.0 = dense, 1.0 = all pruned)
    pub sparsity: f32,

    /// Number of calibration samples to use
    pub n_calibration: usize,

    /// Use L2 norm (true) or L1 norm/mean (false) for activation scoring
    pub use_l2: bool,
}

impl Default for WandaConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.5,
            n_calibration: 128,
            use_l2: true,
        }
    }
}

/// Wanda: Activation-Weighted Magnitude Pruning
///
/// Computes importance scores as: `S[i,j] = |W[i,j]| × σ[x[j]]`
///
/// where `σ[x[j]]` is the L2 norm (or mean absolute value) of activation j
/// across calibration samples.
///
/// # Algorithm
///
/// 1. Collect activation statistics from calibration data
/// 2. Compute importance scores = |weights| × activation_norms
/// 3. Keep top-(1-sparsity) weights, prune the rest
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::prelude::*;
///
/// let config = WandaConfig {
///     sparsity: 0.5,
///     n_calibration: 128,
///     use_l2: true,
/// };
///
/// let mut wanda = Wanda::new(config);
/// let mask = wanda.prune(&weights, &calibration_data);
/// let sparse_weights = mask.apply(&weights);
/// ```
pub struct Wanda<B: Backend> {
    config: WandaConfig,
    activation_stats: Option<ActivationStats<B>>,
}

impl<B: Backend> Wanda<B> {
    /// Create a new Wanda pruner with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Wanda configuration
    ///
    /// # Returns
    ///
    /// Wanda instance ready for calibration and pruning
    pub fn new(config: WandaConfig) -> Self {
        Self {
            config,
            activation_stats: None,
        }
    }

    /// Calibrate on activation data (collect statistics).
    ///
    /// # Arguments
    ///
    /// * `data` - Calibration data containing activation samples
    ///
    /// # Note
    ///
    /// This step is required before calling `score()` or `create_mask()`.
    /// The `prune()` method calls this automatically.
    pub fn calibrate(&mut self, data: &CalibrationData<B>) {
        // Take only the requested number of calibration samples
        let n_use = self.config.n_calibration.min(data.len());
        let subset = if n_use < data.len() {
            data.take(n_use)
        } else {
            data.clone()
        };

        // Convert to vector of 2D tensors for ActivationStats
        let samples: Vec<Tensor<B, 2>> = subset
            .iter()
            .map(|sample| sample.unsqueeze()) // [n_features] -> [1, n_features]
            .collect();

        self.activation_stats = Some(ActivationStats::from_samples(&samples));
    }

    /// Compute importance scores for weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix [n_out, n_in]
    ///
    /// # Returns
    ///
    /// Importance scores [n_out, n_in]
    ///
    /// # Panics
    ///
    /// Panics if `calibrate()` has not been called first
    pub fn score(&self, weights: &Tensor<B, 2>) -> Tensor<B, 2> {
        let stats = self
            .activation_stats
            .as_ref()
            .expect("Must calibrate before scoring. Call calibrate() first.");

        // |W[i,j]|
        let abs_weights = weights.clone().abs();

        // Get activation norms σ[x[j]]
        let norms = if self.config.use_l2 {
            stats.l2_norm().clone()
        } else {
            // Use mean absolute value as approximation
            stats.mean().clone().abs()
        };

        // Broadcast multiply: [n_out, n_in] * [n_in] -> [n_out, n_in]
        // norms needs to be unsqueezed to [1, n_in] for broadcasting
        abs_weights * norms.unsqueeze()
    }

    /// Create sparse mask from importance scores.
    ///
    /// # Arguments
    ///
    /// * `scores` - Importance scores [n_out, n_in]
    ///
    /// # Returns
    ///
    /// Binary mask keeping top-(1-sparsity) weights
    pub fn create_mask(&self, scores: &Tensor<B, 2>) -> SparseMask<B> {
        SparseMask::from_scores(scores, self.config.sparsity)
    }

    /// One-shot pruning: calibrate + score + create mask.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix [n_out, n_in]
    /// * `data` - Calibration data
    ///
    /// # Returns
    ///
    /// Binary sparse mask
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut wanda = Wanda::new(config);
    /// let mask = wanda.prune(&weights, &calibration_data);
    /// ```
    pub fn prune(
        &mut self,
        weights: &Tensor<B, 2>,
        data: &CalibrationData<B>,
    ) -> SparseMask<B> {
        self.calibrate(data);
        let scores = self.score(weights);
        self.create_mask(&scores)
    }

    /// Apply mask to weights (convenience method).
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix [n_out, n_in]
    /// * `mask` - Sparse mask
    ///
    /// # Returns
    ///
    /// Sparse weights with pruned positions zeroed
    pub fn apply_mask(&self, weights: &Tensor<B, 2>, mask: &SparseMask<B>) -> Tensor<B, 2> {
        mask.apply(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::CalibrationData;
    use crate::TestBackend as TB;

    fn create_test_data() -> (Tensor<TB, 2>, CalibrationData<TB>) {
        let weights = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &Default::default(),
        );

        let cal_samples = vec![
            Tensor::<TB, 2>::from_data([[1.0, 1.0, 1.0]], &Default::default()),
            Tensor::<TB, 2>::from_data([[2.0, 2.0, 2.0]], &Default::default()),
            Tensor::<TB, 2>::from_data([[3.0, 3.0, 3.0]], &Default::default()),
        ];

        let calibration = CalibrationData::from_samples(cal_samples);

        (weights, calibration)
    }

    #[test]
    fn test_wanda_calibrate() {
        let (_, calibration) = create_test_data();

        let config = WandaConfig {
            sparsity: 0.5,
            n_calibration: 2,
            use_l2: true,
        };

        let mut wanda = Wanda::new(config);
        wanda.calibrate(&calibration);

        assert!(wanda.activation_stats.is_some());
    }

    #[test]
    fn test_wanda_score() {
        let (weights, calibration) = create_test_data();

        let config = WandaConfig::default();
        let mut wanda = Wanda::new(config);
        wanda.calibrate(&calibration);

        let scores = wanda.score(&weights);

        // Scores should be non-negative
        assert_eq!(scores.dims(), weights.dims());

        let score_data: Vec<f32> = scores.into_data().to_vec().unwrap();
        assert!(score_data.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_wanda_prune() {
        let (weights, calibration) = create_test_data();

        let config = WandaConfig {
            sparsity: 0.5,
            n_calibration: 3,
            use_l2: true,
        };

        let mut wanda = Wanda::new(config);
        let mask = wanda.prune(&weights, &calibration);

        assert_eq!(mask.shape(), [2, 3]);
        assert_eq!(mask.n_active(), 3); // 50% sparsity on 6 elements = 3 active
        assert_eq!(mask.n_pruned(), 3);
    }

    #[test]
    fn test_wanda_apply_mask() {
        let (weights, calibration) = create_test_data();

        let config = WandaConfig {
            sparsity: 0.5,
            n_calibration: 3,
            use_l2: true,
        };

        let mut wanda = Wanda::new(config);
        let mask = wanda.prune(&weights, &calibration);
        let sparse_weights = wanda.apply_mask(&weights, &mask);

        assert_eq!(sparse_weights.dims(), weights.dims());

        let sparse_data: Vec<f32> = sparse_weights.into_data().to_vec().unwrap();

        // Should have exactly 3 zeros (50% sparsity)
        let n_zeros = sparse_data.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(n_zeros, 3);
    }

    #[test]
    fn test_wanda_l1_vs_l2() {
        let (weights, calibration) = create_test_data();

        let config_l2 = WandaConfig {
            sparsity: 0.5,
            n_calibration: 3,
            use_l2: true,
        };

        let config_l1 = WandaConfig {
            sparsity: 0.5,
            n_calibration: 3,
            use_l2: false,
        };

        let mut wanda_l2 = Wanda::new(config_l2);
        let mut wanda_l1 = Wanda::new(config_l1);

        wanda_l2.calibrate(&calibration);
        wanda_l1.calibrate(&calibration);

        let scores_l2 = wanda_l2.score(&weights);
        let scores_l1 = wanda_l1.score(&weights);

        // Both should produce valid scores
        assert_eq!(scores_l2.dims(), weights.dims());
        assert_eq!(scores_l1.dims(), weights.dims());
    }
}
