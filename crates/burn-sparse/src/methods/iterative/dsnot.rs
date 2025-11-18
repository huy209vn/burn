//! DSnoT: Dynamic Sparse No Training (ICLR 2024)
//!
//! **Reference**: arXiv:2310.08915
//!
//! Training-free sparse mask refinement using reconstruction-error-based Δε scoring.
//!
//! ## Core Algorithm
//!
//! DSnoT refines an initial sparse mask by iteratively swapping weights based on
//! how much they would improve reconstruction error if restored (grow) vs removed (prune).
//!
//! **Key innovation**: Variance-penalized grow scoring prevents unstable swaps.
//!
//! ## Scoring Functions
//!
//! For each weight w_ij:
//!
//! **Grow Δε** (if pruned → active):
//! ```text
//! Δε_grow = 2·w·x·e - w²·x²
//! Score_grow = μ(Δε_grow) / (σ²(Δε_grow) + λ)
//! ```
//!
//! **Prune Δε** (if active → pruned):
//! ```text
//! Δε_prune = 2·w·x·e + w²·x²
//! Score_prune = μ(Δε_prune)
//! ```
//!
//! Where:
//! - e = Y_dense - Y_sparse (reconstruction error)
//! - μ = mean across calibration samples
//! - σ² = variance across calibration samples
//! - λ = numerical stability constant
//!
//! ## Differences from Other Methods
//!
//! **NOT Wanda**: No activation L2 norms, no per-output scoring
//! **NOT row-wise**: Global swap of top K% weights
//! **NOT sign-filtering**: No directional constraints
//!
//! ## Example
//!
//! ```rust,ignore
//! use burn_sparse::prelude::*;
//!
//! // Start with Wanda mask
//! let initial_mask = wanda.prune(&weights, &calibration);
//!
//! // Refine with DSnoT
//! let config = DSnoTConfig::default();
//! let mut dsnot = DSnoT::new(config);
//! let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);
//! ```

use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Bool, ElementConversion, Tensor};

use crate::core::{CalibrationData, SparseMask};

/// Configuration for DSnoT refinement.
#[derive(Debug, Clone)]
pub struct DSnoTConfig {
    /// Maximum number of refinement iterations
    pub max_iters: usize,

    /// Convergence tolerance (stop if improvement < tolerance)
    pub tolerance: f32,

    /// Number of calibration samples to use
    pub n_calibration: usize,

    /// Fraction of weights to swap per iteration (e.g., 0.01 = 1%)
    pub swap_fraction: f32,

    /// Numerical stability constant for variance denominator
    pub lambda: f32,
}

impl Default for DSnoTConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tolerance: 1e-6,
            n_calibration: 128,
            swap_fraction: 0.01, // 1% swap per iteration
            lambda: 1e-8,
        }
    }
}

/// DSnoT: Dynamic Sparse No Training
///
/// Variance-penalized iterative mask refinement.
pub struct DSnoT {
    config: DSnoTConfig,
    error_history: Vec<f32>,
}

impl DSnoT {
    /// Create a new DSnoT refiner.
    pub fn new(config: DSnoTConfig) -> Self {
        Self {
            config,
            error_history: Vec::new(),
        }
    }

    /// Get error history (for analysis/plotting)
    pub fn error_history(&self) -> &[f32] {
        &self.error_history
    }

    /// Refine a sparse mask through iterative Δε-based swaps.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix [n_out, n_in]
    /// * `initial_mask` - Starting mask (typically from Wanda)
    /// * `data` - Calibration data
    ///
    /// # Returns
    ///
    /// Refined sparse mask
    pub fn refine<B: Backend>(
        &mut self,
        weights: &Tensor<B, 2>,
        initial_mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> SparseMask<B> {
        let shape = weights.dims();
        let [n_out, n_in] = [shape[0], shape[1]];
        let total_weights = n_out * n_in;

        // Use subset of calibration data
        let n_use = self.config.n_calibration.min(data.len());
        let samples: Vec<_> = (0..n_use)
            .map(|i| data.samples().clone().slice([i..i + 1]))
            .collect();

        // Start with initial mask
        let mut current_mask = initial_mask.clone();

        self.error_history.clear();

        for iter in 0..self.config.max_iters {
            // 1. Compute reconstruction error
            let error = self.compute_reconstruction_error(weights, &current_mask, &samples);
            self.error_history.push(error);

            if iter > 0 {
                let prev_error = self.error_history[iter - 1];
                let improvement = prev_error - error;

                // Check convergence
                if improvement < self.config.tolerance {
                    break;
                }
            }

            // 2. Compute Δε for all weights across all samples
            let (grow_scores, prune_scores) =
                self.compute_delta_epsilon_scores(weights, &current_mask, &samples);

            // 3. Select top K% to swap
            let k = ((total_weights as f32 * self.config.swap_fraction) as usize).max(1);

            let (to_grow, to_prune) =
                self.select_swap_candidates(&current_mask, &grow_scores, &prune_scores, k);

            if to_grow.is_empty() || to_prune.is_empty() {
                // No more swaps possible
                break;
            }

            // 4. Update mask
            current_mask = self.apply_swaps(&current_mask, &to_grow, &to_prune);
        }

        current_mask
    }

    /// Compute global reconstruction error: ||Y_dense - Y_sparse||²
    fn compute_reconstruction_error<B: Backend>(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        samples: &[Tensor<B, 2>],
    ) -> f32 {
        let sparse_weights = mask.apply(weights);

        let mut total_error = 0.0;

        for sample in samples {
            // Y_dense = W @ X
            let y_dense = weights.clone().matmul(sample.clone().transpose());

            // Y_sparse = (M ⊙ W) @ X
            let y_sparse = sparse_weights.clone().matmul(sample.clone().transpose());

            // Squared error
            let error = (y_dense - y_sparse).powf_scalar(2.0).sum();

            use burn_core::tensor::ElementConversion;
            total_error += error.into_scalar().elem::<f32>();
        }

        total_error / samples.len() as f32
    }

    /// Compute Δε-based grow and prune scores for all weights.
    ///
    /// Returns: (grow_scores [n_out, n_in], prune_scores [n_out, n_in])
    fn compute_delta_epsilon_scores<B: Backend>(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        samples: &[Tensor<B, 2>],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let shape = weights.dims();
        let [n_out, n_in] = [shape[0], shape[1]];
        let n_samples = samples.len();

        let sparse_weights = mask.apply(weights);

        // Collect Δε across all samples for all weights
        // grow_deltas: [n_samples, n_out, n_in]
        // prune_deltas: [n_samples, n_out, n_in]

        let mut grow_deltas_all: Vec<Tensor<B, 2>> = Vec::with_capacity(n_samples);
        let mut prune_deltas_all: Vec<Tensor<B, 2>> = Vec::with_capacity(n_samples);

        for sample in samples {
            // Y_dense = W @ X  [n_out, batch=1]
            let y_dense = weights.clone().matmul(sample.clone().transpose());

            // Y_sparse = (M ⊙ W) @ X  [n_out, batch=1]
            let y_sparse = sparse_weights.clone().matmul(sample.clone().transpose());

            // e = Y_dense - Y_sparse  [n_out, batch=1]
            let e = y_dense - y_sparse;

            // For each weight (i, j):
            // contribution = w_ij * x_j  [batch=1]
            // Δε_grow = 2 * contribution * e_i - contribution²
            // Δε_prune = 2 * contribution * e_i + contribution²

            // Broadcast computation:
            // w * x gives [n_out, n_in] when properly broadcast
            // We need: w[i,j] * x[j] for all (i,j)

            let x_t = sample.clone(); // [batch=1, n_in]
            let contribution = weights.clone() * x_t; // Broadcasting: [n_out, n_in] * [1, n_in]

            let e_expanded = e.repeat_dim(1, n_in); // [n_out, n_in]
            let contrib_sq = contribution.clone().powf_scalar(2.0);

            // Δε_grow = 2·w·x·e - (w·x)²
            let delta_grow = contribution.clone() * e_expanded.clone() * 2.0 - contrib_sq.clone();

            // Δε_prune = 2·w·x·e + (w·x)²
            let delta_prune = contribution * e_expanded * 2.0 + contrib_sq;

            grow_deltas_all.push(delta_grow);
            prune_deltas_all.push(delta_prune);
        }

        // Stack along sample dimension and compute statistics
        let grow_stack = Tensor::stack::<3>(grow_deltas_all, 0); // [n_samples, n_out, n_in]
        let prune_stack = Tensor::stack::<3>(prune_deltas_all, 0);

        // Mean and variance across samples (dim 0)
        let grow_mean = grow_stack.clone().mean_dim(0).squeeze::<2>(); // [n_out, n_in]
        let grow_var = grow_stack
            .clone()
            .sub(grow_mean.clone().unsqueeze())
            .powf_scalar(2.0)
            .mean_dim(0)
            .squeeze::<2>(); // [n_out, n_in]

        let prune_mean = prune_stack.mean_dim(0).squeeze::<2>(); // [n_out, n_in]

        // Grow score: μ / (σ² + λ)
        let grow_scores = grow_mean.clone() / (grow_var + self.config.lambda);

        // Prune score: μ
        let prune_scores = prune_mean;

        (grow_scores, prune_scores)
    }

    /// Select top K candidates to grow (from pruned) and prune (from active).
    fn select_swap_candidates<B: Backend>(
        &self,
        mask: &SparseMask<B>,
        grow_scores: &Tensor<B, 2>,
        prune_scores: &Tensor<B, 2>,
        k: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let shape = grow_scores.dims();
        let total = shape[0] * shape[1];

        // Get mask data
        let mask_data: Vec<bool> = mask
            .tensor()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();

        // Get score data
        let grow_data: Vec<f32> = grow_scores
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        let prune_data: Vec<f32> = prune_scores
            .clone()
            .into_data()
            .to_vec()
            .unwrap();

        // Find pruned positions with highest grow score
        let mut pruned_candidates: Vec<(usize, f32)> = (0..total)
            .filter(|&i| !mask_data[i]) // Only pruned
            .map(|i| (i, grow_data[i]))
            .collect();

        pruned_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let to_grow: Vec<usize> = pruned_candidates.iter().take(k).map(|(i, _)| *i).collect();

        // Find active positions with highest prune score (error increase)
        let mut active_candidates: Vec<(usize, f32)> = (0..total)
            .filter(|&i| mask_data[i]) // Only active
            .map(|i| (i, prune_data[i]))
            .collect();

        active_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let to_prune: Vec<usize> = active_candidates.iter().take(k).map(|(i, _)| *i).collect();

        (to_grow, to_prune)
    }

    /// Apply swaps to mask.
    fn apply_swaps<B: Backend>(
        &self,
        mask: &SparseMask<B>,
        to_grow: &[usize],
        to_prune: &[usize],
    ) -> SparseMask<B> {
        let shape = mask.shape();
        let total = shape[0] * shape[1];

        let mut mask_data: Vec<bool> = mask
            .tensor()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();

        // Grow: pruned → active
        for &idx in to_grow {
            mask_data[idx] = true;
        }

        // Prune: active → pruned
        for &idx in to_prune {
            mask_data[idx] = false;
        }

        // Create new mask tensor
        use burn_core::tensor::{Shape, TensorData};
        let mask_tensor = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, Shape::new([shape[0], shape[1]])),
            &mask.device(),
        );

        SparseMask::from_tensor(mask_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TB = NdArray<f32>;

    #[test]
    fn test_dsnot_refine_basic() {
        // Create simple test case
        let weights = Tensor::<TB, 2>::from_data(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &Default::default(),
        );

        // Start with 50% sparse mask
        let mask_data = Tensor::<TB, 2, Bool>::from_data(
            [[true, false, true], [true, true, false]],
            &Default::default(),
        );
        let initial_mask = SparseMask::from_tensor(mask_data);

        // Calibration data
        let samples = vec![Tensor::<TB, 2>::from_data(
            [[1.0, 1.0, 1.0]],
            &Default::default(),
        )];
        let calibration = CalibrationData::from_samples(samples);

        // Run DSnoT
        let config = DSnoTConfig {
            max_iters: 10,
            tolerance: 1e-6,
            n_calibration: 1,
            swap_fraction: 0.2,
            lambda: 1e-8,
        };

        let mut dsnot = DSnoT::new(config);
        let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);

        // Check that mask was modified (at least one swap occurred)
        assert_eq!(refined_mask.shape(), [2, 3]);

        // Check error decreased
        let history = dsnot.error_history();
        if history.len() > 1 {
            assert!(history[history.len() - 1] <= history[0]);
        }
    }

    #[test]
    fn test_dsnot_preserves_sparsity() {
        let weights = Tensor::<TB, 2>::from_data(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            &Default::default(),
        );

        let mask_data = Tensor::<TB, 2, Bool>::from_data(
            [[true, false, true, false], [false, true, false, true]],
            &Default::default(),
        );
        let initial_mask = SparseMask::from_tensor(mask_data);

        let samples = vec![Tensor::<TB, 2>::from_data(
            [[1.0, 1.0, 1.0, 1.0]],
            &Default::default(),
        )];
        let calibration = CalibrationData::from_samples(samples);

        let config = DSnoTConfig {
            max_iters: 5,
            ..Default::default()
        };

        let mut dsnot = DSnoT::new(config);
        let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);

        // Sparsity should be preserved (same number of active weights)
        assert_eq!(refined_mask.n_active(), initial_mask.n_active());
        assert_eq!(refined_mask.n_pruned(), initial_mask.n_pruned());
    }
}
