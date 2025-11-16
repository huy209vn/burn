//! DSnoT: Variance-weighted iterative mask refinement.
//!
//! **Reference**: DSnoT paper (ICLR 2024)
//!
//! DSnoT refines an initial sparse mask by iteratively swapping weights based on
//! reconstruction error. It uses activation statistics to score pruned weights for
//! restoration and applies sign-aware filtering when selecting weights to prune.

use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Bool, ElementConversion, Shape, Tensor, TensorData};

use crate::primitives::{ActivationStats, CalibrationData, SparseMask};

/// Configuration for DSnoT pruning.
#[derive(Debug, Clone)]
pub struct DSnoTConfig {
    /// Maximum number of refinement iterations per row
    pub max_iters: usize,

    /// Convergence tolerance for reconstruction error
    pub tolerance: f32,

    /// Number of calibration samples to use
    pub n_calibration: usize,
}

impl Default for DSnoTConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-5,
            n_calibration: 128,
        }
    }
}

/// DSnoT: Variance-Weighted Iterative Mask Refinement
///
/// Iteratively refines a sparse mask by:
/// 1. Computing reconstruction error per output row
/// 2. Selecting pruned weights to restore using variance-penalized scoring
/// 3. Selecting active weights to prune using sign-filtered Wanda scoring
/// 4. Swapping weights and repeating until convergence
///
/// **Growing score**: `W · E[A] / Var[A]` (variance-penalized contribution)
/// **Pruning score**: `|W| · ||A||_2` (Wanda), filtered by sign constraint
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
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> DSnoT<B> {
    /// Create a new DSnoT refiner with the given configuration.
    pub fn new(config: DSnoTConfig) -> Self {
        Self {
            config,
            error_history: Vec::new(),
            _backend: core::marker::PhantomData,
        }
    }

    /// Refine a sparse mask through iterative row-wise swaps.
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
        // Precompute activation statistics once
        let n_use = self.config.n_calibration.min(data.len());
        let subset = if n_use < data.len() {
            data.take(n_use)
        } else {
            data.clone()
        };

        let samples: Vec<Tensor<B, 2>> = subset
            .iter()
            .map(|sample| sample.unsqueeze()) // [n_features] -> [1, n_features]
            .collect();

        let act_stats = ActivationStats::from_samples(&samples);

        let shape = weights.dims();
        let n_out = shape[0];
        let n_in = shape[1];

        // Start with initial mask
        let mut mask_data = vec![false; n_out * n_in];
        for &idx in initial_mask.active_indices() {
            mask_data[idx] = true;
        }

        self.error_history.clear();

        // Process each row independently
        for row_idx in 0..n_out {
            for _iter in 0..self.config.max_iters {
                // 1. Compute current reconstruction error for this row
                let error_r = self.compute_row_error(row_idx, weights, &mask_data, data, n_in);

                // 2. Select pruned weight to restore (grow)
                let grow_col = self.select_grow_position(
                    row_idx,
                    weights,
                    &mask_data,
                    &act_stats,
                    error_r,
                    n_in,
                );

                // 3. Select active weight to prune (with sign filtering)
                let prune_col = self.select_prune_position(
                    row_idx,
                    weights,
                    &mask_data,
                    &act_stats,
                    error_r,
                    n_in,
                );

                // If we can't find valid swap, skip this row
                if grow_col.is_none() || prune_col.is_none() {
                    break;
                }

                let grow_col = grow_col.unwrap();
                let prune_col = prune_col.unwrap();

                // 4. Swap: restore grow_col, prune prune_col
                let grow_idx = row_idx * n_in + grow_col;
                let prune_idx = row_idx * n_in + prune_col;

                mask_data[grow_idx] = true;
                mask_data[prune_idx] = false;

                // 5. Check convergence for this row
                let new_error = self.compute_row_error(row_idx, weights, &mask_data, data, n_in);

                if (error_r - new_error).abs() < self.config.tolerance {
                    break; // Row converged
                }
            }
        }

        // Convert mask_data back to SparseMask
        let mask_tensor = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, Shape::new([n_out, n_in])),
            &weights.device(),
        );

        SparseMask::from_tensor(mask_tensor)
    }

    /// Get error history across iterations.
    pub fn error_history(&self) -> &[f32] {
        &self.error_history
    }

    // Private helper methods

    /// Compute reconstruction error for a single row.
    fn compute_row_error(
        &self,
        row_idx: usize,
        weights: &Tensor<B, 2>,
        mask_data: &[bool],
        data: &CalibrationData<B>,
        n_in: usize,
    ) -> f32 {
        // Extract row weights
        let w_row = weights.clone().slice([row_idx..row_idx + 1]);

        // Apply mask to this row
        let mut w_sparse_data = vec![0.0f32; n_in];
        for col in 0..n_in {
            let idx = row_idx * n_in + col;
            if mask_data[idx] {
                let w_val: Vec<f32> = w_row
                    .clone()
                    .slice([0..1, col..col + 1])
                    .into_data()
                    .to_vec()
                    .unwrap();
                w_sparse_data[col] = w_val[0];
            }
        }

        let w_sparse_row = Tensor::<B, 2>::from_data(
            TensorData::new(w_sparse_data, Shape::new([1, n_in])),
            &weights.device(),
        );

        // Compute error across all samples
        let mut total_error = 0.0f32;
        let mut n_samples = 0;

        for sample in data.iter() {
            // Dense output: w_row · sample [1, 1]
            let y_dense = w_row
                .clone()
                .matmul(sample.clone().unsqueeze_dim(1));

            // Sparse output: w_sparse_row · sample [1, 1]
            let y_sparse = w_sparse_row
                .clone()
                .matmul(sample.clone().unsqueeze_dim(1));

            // Squared error for this sample
            let error = (y_dense - y_sparse).powf_scalar(2.0);

            use burn_core::tensor::ElementConversion;
            total_error += error.into_scalar().elem::<f32>();
            n_samples += 1;
        }

        total_error / n_samples as f32
    }

    /// Select pruned position to restore (grow).
    ///
    /// Score: `W · E[A] / Var[A]`
    /// - If error > 0: restore weight with highest score
    /// - If error < 0: restore weight with lowest score
    fn select_grow_position(
        &self,
        row_idx: usize,
        weights: &Tensor<B, 2>,
        mask_data: &[bool],
        act_stats: &ActivationStats<B>,
        error: f32,
        n_in: usize,
    ) -> Option<usize> {
        let mean_act = act_stats.mean();
        let var_act = act_stats.variance();

        let mean_data: Vec<f32> = mean_act.clone().into_data().to_vec().unwrap();
        let var_data: Vec<f32> = var_act.clone().into_data().to_vec().unwrap();

        let w_row = weights.clone().slice([row_idx..row_idx + 1]);
        let w_data: Vec<f32> = w_row.into_data().to_vec().unwrap();

        let mut best_col: Option<usize> = None;
        let mut best_score = if error > 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };

        for col in 0..n_in {
            let idx = row_idx * n_in + col;

            // Only consider pruned positions
            if mask_data[idx] {
                continue;
            }

            // Score: W · E[A] / Var[A]
            let score = w_data[col] * mean_data[col] / (var_data[col] + 1e-8);

            // Select based on error sign
            if error > 0.0 {
                // Want to decrease error -> pick highest score
                if score > best_score {
                    best_score = score;
                    best_col = Some(col);
                }
            } else {
                // Want to increase error -> pick lowest score
                if score < best_score {
                    best_score = score;
                    best_col = Some(col);
                }
            }
        }

        best_col
    }

    /// Select active position to prune.
    ///
    /// Score: `|W| · ||A||_2` (Wanda metric)
    /// Constraint: Only weights where `W · E[A]` has opposite sign to error
    fn select_prune_position(
        &self,
        row_idx: usize,
        weights: &Tensor<B, 2>,
        mask_data: &[bool],
        act_stats: &ActivationStats<B>,
        error: f32,
        n_in: usize,
    ) -> Option<usize> {
        let mean_act = act_stats.mean();
        let l2_norm = act_stats.l2_norm();

        let mean_data: Vec<f32> = mean_act.clone().into_data().to_vec().unwrap();
        let l2_data: Vec<f32> = l2_norm.clone().into_data().to_vec().unwrap();

        let w_row = weights.clone().slice([row_idx..row_idx + 1]);
        let w_data: Vec<f32> = w_row.into_data().to_vec().unwrap();

        let mut best_col: Option<usize> = None;
        let mut best_score = f32::INFINITY;

        for col in 0..n_in {
            let idx = row_idx * n_in + col;

            // Only consider active positions
            if !mask_data[idx] {
                continue;
            }

            // Sign constraint: W · E[A] must have opposite sign to error
            let contribution = w_data[col] * mean_data[col];

            let is_safe = if error > 0.0 {
                contribution < 0.0 // Want negative contribution
            } else {
                contribution > 0.0 // Want positive contribution
            };

            if !is_safe {
                continue;
            }

            // Wanda score: |W| · ||A||_2
            let score = w_data[col].abs() * l2_data[col];

            if score < best_score {
                best_score = score;
                best_col = Some(col);
            }
        }

        best_col
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

        assert_eq!(dsnot.error_history().len(), 0);
    }

    #[test]
    fn test_dsnot_refine() {
        let (weights, initial_mask, calibration) = create_test_setup();

        let config = DSnoTConfig {
            max_iters: 5,
            tolerance: 1e-5,
            n_calibration: 3,
        };

        let mut dsnot = DSnoT::new(config);
        let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);

        assert_eq!(refined_mask.shape(), initial_mask.shape());
        assert_eq!(refined_mask.n_active(), initial_mask.n_active());
    }

    #[test]
    fn test_dsnot_preserves_sparsity() {
        let (weights, initial_mask, calibration) = create_test_setup();

        let config = DSnoTConfig::default();
        let mut dsnot = DSnoT::new(config);

        let initial_sparsity = initial_mask.actual_sparsity();
        let refined_mask = dsnot.refine(&weights, &initial_mask, &calibration);

        // Sparsity should be preserved
        assert!((refined_mask.actual_sparsity() - initial_sparsity).abs() < 0.01);
    }
}
