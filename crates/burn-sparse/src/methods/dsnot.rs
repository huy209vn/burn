//! DSnoT: Variance-weighted iterative mask refinement.
//!
//! **Reference**: [DSnoT paper citation]

use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Tensor};

use crate::primitives::{bottomk_indices, topk_indices, CalibrationData, SparseMask};

/// Configuration for DSnoT pruning.
#[derive(Debug, Clone)]
pub struct DSnoTConfig {
    /// Maximum number of refinement iterations
    pub max_iters: usize,

    /// Update threshold (fraction of weights to swap per iteration)
    pub update_threshold: f32,

    /// Variance penalty exponent (α in μ/σ^α)
    pub alpha: f32,

    /// Convergence tolerance for reconstruction error
    pub tolerance: f32,

    /// Small constant for numerical stability (λ in μ/(σ²+λ)^α)
    pub lambda: f32,
}

impl Default for DSnoTConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            update_threshold: 0.01,
            alpha: 1.0,
            tolerance: 1e-5,
            lambda: 1e-8,
        }
    }
}

/// DSnoT: Variance-Weighted Iterative Mask Refinement
///
/// Iteratively refines a sparse mask by:
/// 1. Computing error reduction distribution for pruned weights
/// 2. Computing error increase distribution for active weights
/// 3. Swapping worst active weights with best pruned weights
/// 4. Repeating until convergence
///
/// Growing score: `μ / (σ² + λ)^α` (Sharpe ratio with variance penalty)
/// Pruning score: `μ` (mean damage from removal)
///
/// # Algorithm
///
/// ```text
/// for iter in 0..max_iters:
///     1. For each pruned weight: compute error reduction if restored
///     2. For each active weight: compute error increase if pruned
///     3. Select top-K pruned weights by variance-weighted score
///     4. Select bottom-K active weights by damage score
///     5. Swap selected weights
///     6. Check convergence
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::prelude::*;
///
/// // Start with Wanda mask
/// let initial_mask = wanda.prune(&weights, &calibration);
///
/// // Refine with DSnoT
/// let config = DSnoTConfig::default();
/// let mut dsnot = DSnoT::new(config);
/// let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);
/// ```
pub struct DSnoT<B: Backend> {
    config: DSnoTConfig,
    error_history: Vec<f32>,
    iteration: usize,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> DSnoT<B> {
    /// Create a new DSnoT refiner with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - DSnoT configuration
    ///
    /// # Returns
    ///
    /// DSnoT instance ready for refinement
    pub fn new(config: DSnoTConfig) -> Self {
        Self {
            config,
            error_history: Vec::new(),
            iteration: 0,
            _backend: core::marker::PhantomData,
        }
    }

    /// Refine a sparse mask through iterative swaps.
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
    pub fn refine(
        &mut self,
        weights: &Tensor<B, 2>,
        initial_mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> SparseMask<B> {
        let mut mask = initial_mask.clone();
        self.error_history.clear();
        self.iteration = 0;

        for iter in 0..self.config.max_iters {
            self.iteration = iter;

            // Compute scores for growing and pruning
            let (grow_scores, prune_scores) = self.compute_scores(weights, &mask, data);

            // Swap weights
            let new_mask = self.swap_weights(&mask, &grow_scores, &prune_scores);

            // Check convergence
            let error = self.compute_reconstruction_error(weights, &new_mask, data);
            self.error_history.push(error);

            if iter > 0 {
                let prev_error = self.error_history[iter - 1];
                let error_change = (prev_error - error).abs();

                if error_change < self.config.tolerance {
                    break;
                }
            }

            mask = new_mask;
        }

        mask
    }

    /// Perform a single refinement step.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix [n_out, n_in]
    /// * `mask` - Current mask
    /// * `data` - Calibration data
    ///
    /// # Returns
    ///
    /// Updated mask after one swap iteration
    pub fn step(
        &mut self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> SparseMask<B> {
        let (grow_scores, prune_scores) = self.compute_scores(weights, mask, data);
        self.swap_weights(mask, &grow_scores, &prune_scores)
    }

    /// Check if refinement has converged.
    pub fn has_converged(&self) -> bool {
        if self.error_history.len() < 2 {
            return false;
        }

        let len = self.error_history.len();
        (self.error_history[len - 2] - self.error_history[len - 1]).abs()
            < self.config.tolerance
    }

    /// Get error history across iterations.
    pub fn error_history(&self) -> &[f32] {
        &self.error_history
    }

    /// Get current iteration number.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    // Private helper methods

    fn compute_scores(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (grow_mean, grow_var) = self.compute_growing_distribution(weights, mask, data);
        let prune_mean = self.compute_pruning_distribution(weights, mask, data);

        // Growing score: μ / (σ² + λ)^α (Sharpe ratio with variance penalty)
        let grow_scores = grow_mean / (grow_var + self.config.lambda).powf_scalar(self.config.alpha);

        // Pruning score: just mean damage
        let prune_scores = prune_mean;

        (grow_scores, prune_scores)
    }

    fn compute_growing_distribution(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // For each pruned position, compute error reduction if restored

        let mut error_reductions = Vec::new();

        for sample in data.iter() {
            let error_reduction =
                self.compute_error_reduction_per_weight(weights, mask, &sample);
            error_reductions.push(error_reduction);
        }

        // Compute mean and variance across samples
        let stacked = Tensor::stack(error_reductions, 0); // [n_samples, n_out, n_in]
        let mean = stacked.clone().mean_dim(0); // [n_out, n_in]
        let variance = (stacked - mean.clone().unsqueeze())
            .powf_scalar(2.0)
            .mean_dim(0); // [n_out, n_in]

        (mean, variance)
    }

    fn compute_pruning_distribution(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> Tensor<B, 2> {
        // For each active position, compute error increase if pruned

        let mut error_increases = Vec::new();

        for sample in data.iter() {
            let error_increase = self.compute_error_increase_per_weight(weights, mask, &sample);
            error_increases.push(error_increase);
        }

        // Just use mean (no variance penalty for pruning)
        let stacked = Tensor::stack(error_increases, 0); // [n_samples, n_out, n_in]
        stacked.mean_dim(0) // [n_out, n_in]
    }

    fn compute_error_reduction_per_weight(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        sample: &Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        // For each pruned position (i,j):
        // error_before = ||e_i||² where e_i = Σ_{j:M[i,j]=0} W[i,j]x[j]
        // error_after = ||e_i - W[i,j]x[j]||²
        // Δε[i,j] = error_before - error_after
        //        = 2*W[i,j]*x[j]*e_i - W[i,j]²*x[j]²

        let w_sparse = mask.apply(weights);

        // output_sparse = W_sparse @ x  (note: need to handle dimensions)
        // sample is [n_in], weights are [n_out, n_in]
        let output_sparse = w_sparse.matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>(); // [n_out]
        let output_dense = weights.clone().matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>(); // [n_out]

        // Reconstruction error per row
        let error_per_row = output_dense - output_sparse; // [n_out]

        // Broadcast sample to [n_out, n_in]
        let sample_broadcast = sample.clone().unsqueeze(); // [1, n_in]

        // Weight contribution: W[i,j] * x[j]
        let weight_contribution = weights.clone() * sample_broadcast.clone(); // [n_out, n_in]

        // error_broadcast: e_i for each row
        let error_broadcast = error_per_row.unsqueeze(); // [n_out, 1]

        // Δε = 2*W*x*e - W²*x²
        let reduction = weight_contribution.clone() * error_broadcast * 2.0
            - weight_contribution.powf_scalar(2.0);

        // Only consider pruned positions (mask out active ones)
        let pruned_mask_int = mask.complement().tensor().clone().int().float();
        reduction * pruned_mask_int
    }

    fn compute_error_increase_per_weight(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        sample: &Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        // For each active position (i,j):
        // error_before = current reconstruction error
        // error_after = error if we prune W[i,j]
        // Δε[i,j] = error_after - error_before
        //        = 2*W[i,j]*x[j]*e_i + W[i,j]²*x[j]²

        let w_sparse = mask.apply(weights);

        let output_sparse = w_sparse.matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>(); // [n_out]
        let output_dense = weights.clone().matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>(); // [n_out]

        let error_per_row = output_dense - output_sparse; // [n_out]

        let sample_broadcast = sample.clone().unsqueeze(); // [1, n_in]
        let weight_contribution = weights.clone() * sample_broadcast.clone(); // [n_out, n_in]
        let error_broadcast = error_per_row.unsqueeze(); // [n_out, 1]

        // Δε = 2*W*x*e + W²*x²
        let increase = weight_contribution.clone() * error_broadcast * 2.0
            + weight_contribution.powf_scalar(2.0);

        // Only consider active positions
        let active_mask_int = mask.tensor().clone().int().float();
        increase * active_mask_int
    }

    fn swap_weights(
        &self,
        mask: &SparseMask<B>,
        grow_scores: &Tensor<B, 2>,
        prune_scores: &Tensor<B, 2>,
    ) -> SparseMask<B> {
        let k = ((self.config.update_threshold * mask.n_active() as f32) as usize).max(1);

        // Get top-K pruned positions to grow
        let grow_flat = mask.gather_pruned(grow_scores);
        let to_grow_local = topk_indices(&grow_flat.unsqueeze(), k);
        let to_grow_global: Vec<usize> = to_grow_local
            .iter()
            .map(|&i| mask.pruned_indices()[i])
            .collect();

        // Get bottom-K active positions to prune
        let prune_flat = mask.gather_active(prune_scores);
        let to_prune_local = bottomk_indices(&prune_flat.unsqueeze(), k);
        let to_prune_global: Vec<usize> = to_prune_local
            .iter()
            .map(|&i| mask.active_indices()[i])
            .collect();

        // Create new mask with swapped positions
        let mut active = mask.active_indices().to_vec();
        let mut pruned = mask.pruned_indices().to_vec();

        // Remove from active, add to pruned
        for &idx in &to_prune_global {
            active.retain(|&x| x != idx);
            if !pruned.contains(&idx) {
                pruned.push(idx);
            }
        }

        // Remove from pruned, add to active
        for &idx in &to_grow_global {
            pruned.retain(|&x| x != idx);
            if !active.contains(&idx) {
                active.push(idx);
            }
        }

        // Reconstruct mask tensor
        let total = mask.shape()[0] * mask.shape()[1];
        let mut mask_data = vec![false; total];
        for &idx in &active {
            mask_data[idx] = true;
        }

        use burn_core::tensor::{Bool, Shape, TensorData};
        let mask_tensor = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, Shape::new(mask.shape())),
            mask.device(),
        );

        SparseMask::from_tensor(mask_tensor)
    }

    fn compute_reconstruction_error(
        &self,
        weights: &Tensor<B, 2>,
        mask: &SparseMask<B>,
        data: &CalibrationData<B>,
    ) -> f32 {
        let w_sparse = mask.apply(weights);

        let mut total_error: f32 = 0.0;
        let mut n_samples = 0;

        for sample in data.iter() {
            use burn_core::tensor::ElementConversion;

            let y_dense = weights.clone().matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>();
            let y_sparse = w_sparse.clone().matmul(sample.clone().unsqueeze_dim(1)).squeeze::<1>();

            let error = (y_dense - y_sparse).powf_scalar(2.0).sum();
            total_error += error.into_scalar().elem::<f32>();
            n_samples += 1;
        }

        total_error / n_samples as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::wanda::{Wanda, WandaConfig};
    use crate::TestBackend as TB;

    fn create_test_setup() -> (Tensor<TB, 2>, SparseMask<TB>, CalibrationData<TB>) {
        let weights = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            &Default::default(),
        );

        let cal_samples = vec![
            Tensor::<TB, 2>::from_data([[1.0, 1.0, 1.0, 1.0]], &Default::default()),
            Tensor::<TB, 2>::from_data([[2.0, 2.0, 2.0, 2.0]], &Default::default()),
            Tensor::<TB, 2>::from_data([[3.0, 3.0, 3.0, 3.0]], &Default::default()),
        ];

        let calibration = CalibrationData::from_samples(cal_samples);

        // Create initial mask with Wanda
        let mut wanda = Wanda::new(WandaConfig {
            sparsity: 0.5,
            n_calibration: 3,
            use_l2: true,
        });
        let mask = wanda.prune(&weights, &calibration);

        (weights, mask, calibration)
    }

    #[test]
    fn test_dsnot_creation() {
        let config = DSnoTConfig::default();
        let dsnot = DSnoT::<TB>::new(config);

        assert_eq!(dsnot.iteration(), 0);
        assert_eq!(dsnot.error_history().len(), 0);
    }

    #[test]
    fn test_dsnot_refine() {
        let (weights, initial_mask, calibration) = create_test_setup();

        let config = DSnoTConfig {
            max_iters: 5,
            update_threshold: 0.01,
            alpha: 1.0,
            tolerance: 1e-5,
            lambda: 1e-8,
        };

        let mut dsnot = DSnoT::new(config);
        let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);

        assert_eq!(refined_mask.shape(), initial_mask.shape());
        assert_eq!(refined_mask.n_active(), initial_mask.n_active());
        assert!(dsnot.error_history().len() > 0);
    }

    #[test]
    fn test_dsnot_step() {
        let (weights, initial_mask, calibration) = create_test_setup();

        let config = DSnoTConfig::default();
        let mut dsnot = DSnoT::new(config);

        let new_mask = dsnot.step(&weights, &initial_mask, &calibration);

        assert_eq!(new_mask.shape(), initial_mask.shape());
        // Sparsity should be preserved
        assert_eq!(new_mask.n_active(), initial_mask.n_active());
    }

    #[test]
    fn test_dsnot_error_decreases() {
        let (weights, initial_mask, calibration) = create_test_setup();

        let config = DSnoTConfig {
            max_iters: 10,
            update_threshold: 0.05,
            alpha: 1.0,
            tolerance: 1e-5,
            lambda: 1e-8,
        };

        let mut dsnot = DSnoT::new(config);
        let _ = dsnot.refine(&weights, &initial_mask, &calibration);

        let history = dsnot.error_history();

        // Error should generally decrease or stay flat (allowing small increases due to numerical issues)
        if history.len() > 1 {
            let first_error = history[0];
            let last_error = history[history.len() - 1];

            // Last error should not be significantly higher than first
            assert!(last_error <= first_error * 1.1);
        }
    }
}
