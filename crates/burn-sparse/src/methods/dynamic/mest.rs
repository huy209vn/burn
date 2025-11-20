//! MEST: Memory-Economic Sparse Training
//!
//! **Reference**: Yuan et al., "MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge", NeurIPS 2021
//!
//! MEST uses salience-based pruning and gradient-based growing with:
//! - Salience S = |W| + λ|g| for both pruning AND growing
//! - Elastic mutation schedule (decay rewire fraction over time)
//! - Optional soft memory bound for stability

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::SparseMask;

/// Configuration for MEST dynamic sparse training
#[derive(Debug, Clone)]
pub struct MestConfig {
    /// Target sparsity ratio
    pub sparsity: f32,

    /// Update mask every N training steps
    pub update_frequency: usize,

    /// Initial mutation rate (fraction of active weights to rewire)
    pub mutation_rate_init: f32,

    /// Final mutation rate (after decay)
    pub mutation_rate_final: f32,

    /// Salience lambda: S = |W| + lambda * |g|
    pub lambda: f32,

    /// Use EMA smoothing for gradients
    pub use_gradient_ema: bool,

    /// EMA beta for gradient smoothing (if enabled)
    pub gradient_ema_beta: f32,
}

impl Default for MestConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.8,
            update_frequency: 100,
            mutation_rate_init: 0.2,
            mutation_rate_final: 0.01,
            lambda: 0.1,
            use_gradient_ema: false,
            gradient_ema_beta: 0.9,
        }
    }
}

/// MEST: Dynamic sparse training with salience-based mask updates
///
/// # Algorithm
/// 1. Compute salience S = |W| + λ|g| for all weights
/// 2. Every N steps:
///    - Prune: Remove k active weights with smallest salience
///    - Grow: Add k zero positions with largest salience
/// 3. Decay mutation rate over time (elastic mutation schedule)
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::methods::dynamic::*;
///
/// let mut mest = Mest::new(config, initial_mask, total_steps);
///
/// // Training loop
/// for batch in dataloader {
///     let loss = model.forward(batch);
///     let grads = loss.backward();
///
///     // Update mask based on weights and gradients
///     let new_mask = mest.update_mask(&weights, &grads.weight);
///     if new_mask != old_mask {
///         sparse_linear.update_mask(new_mask);
///     }
/// }
/// ```
pub struct Mest<B: Backend> {
    config: MestConfig,
    mask: SparseMask<B>,
    step_count: usize,
    grad_ema: Option<Tensor<B, 2>>,
    total_steps: usize, // For mutation rate schedule
}

impl<B: Backend> Mest<B> {
    /// Create new MEST trainer
    ///
    /// # Arguments
    /// * `config` - MEST configuration
    /// * `initial_mask` - Initial sparsity pattern
    /// * `total_steps` - Total training steps (for mutation rate decay)
    pub fn new(config: MestConfig, initial_mask: SparseMask<B>, total_steps: usize) -> Self {
        Self {
            config,
            mask: initial_mask,
            step_count: 0,
            grad_ema: None,
            total_steps,
        }
    }

    /// Update mask based on weights and gradients (call every training step)
    ///
    /// MEST algorithm:
    /// - Computes salience S = |W| + λ|g| for ALL weights
    /// - Prune: Remove bottom-k active weights by salience
    /// - Grow: Add top-k zero positions by salience
    ///
    /// # Arguments
    /// * `weights` - Current weight values [n_out, n_in]
    /// * `gradients` - Weight gradients [n_out, n_in]
    ///
    /// # Returns
    /// Updated mask (may be same as before if not at update frequency)
    pub fn update_mask(&mut self, weights: &Tensor<B, 2>, gradients: &Tensor<B, 2>) -> SparseMask<B> {
        // Update EMA of gradients if enabled
        if self.config.use_gradient_ema {
            let grad_mag = gradients.clone().abs();
            self.grad_ema = Some(match self.grad_ema.take() {
                Some(ema) => {
                    ema.mul_scalar(self.config.gradient_ema_beta)
                        .add(grad_mag.mul_scalar(1.0 - self.config.gradient_ema_beta))
                }
                None => grad_mag,
            });
        }

        self.step_count += 1;

        // Check if it's time to update
        if self.step_count % self.config.update_frequency != 0 {
            return self.mask.clone();
        }

        // Compute current mutation rate (elastic decay schedule)
        let progress = (self.step_count as f32) / (self.total_steps as f32);
        let mutation_rate = self.config.mutation_rate_init
            + (self.config.mutation_rate_final - self.config.mutation_rate_init) * progress;
        let mutation_rate = mutation_rate.max(self.config.mutation_rate_final);

        // Get current mask as tensor
        let mask_tensor = self.mask.tensor();
        let device = mask_tensor.device();
        let shape = self.mask.shape();

        // Compute scores (keep as 2D for topk/bottomk functions)
        let weights_abs = weights.clone().abs();
        let grads_abs = if self.config.use_gradient_ema {
            self.grad_ema.as_ref().unwrap().clone()
        } else {
            gradients.clone().abs()
        };
        let mask_float = mask_tensor.clone().float();

        // MEST spec: Salience S = |W| + lambda * |g|
        let salience = weights_abs.add(grads_abs.mul_scalar(self.config.lambda));

        // Compute number of weights to rewire
        let n_active = self.mask.n_active();
        let k = (n_active as f32 * mutation_rate) as usize;

        if k == 0 {
            return self.mask.clone();
        }

        // Compute salience for active and pruned weights
        // Active weights: multiply salience by mask (zeros out pruned)
        let active_salience = salience.clone().mul(mask_float.clone());

        // Pruned weights: multiply salience by (1 - mask) (zeros out active)
        let pruned_salience = salience.mul(mask_float.clone().neg().add_scalar(1.0));

        // Find bottom-k active weights (smallest salience)
        let drop_indices = crate::core::utils::bottomk_indices(&active_salience, k);

        // Find top-k pruned weights (largest salience)
        let grow_indices = crate::core::utils::topk_indices(&pruned_salience, k);

        // Create new mask: start with current mask, flip the selected indices
        let mask_flat = mask_float.reshape([shape[0] * shape[1]]);
        let mut new_mask_data = mask_flat.to_data().to_vec::<f32>().unwrap();

        // Drop selected active weights
        for idx in drop_indices.iter() {
            new_mask_data[*idx as usize] = 0.0;
        }

        // Grow selected pruned weights
        for idx in grow_indices.iter() {
            new_mask_data[*idx as usize] = 1.0;
        }

        // Reshape back to original shape and convert to Bool
        use burn_core::tensor::TensorData;
        let new_mask_tensor = Tensor::<B, 1>::from_data(TensorData::new(new_mask_data, [shape[0] * shape[1]]), &device)
            .reshape(shape)
            .greater_elem(0.5);

        // Create new mask
        self.mask = SparseMask::from_tensor(new_mask_tensor);

        self.mask.clone()
    }

    /// Get current mask
    pub fn mask(&self) -> &SparseMask<B> {
        &self.mask
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get current mutation rate (for monitoring)
    pub fn current_mutation_rate(&self) -> f32 {
        let progress = (self.step_count as f32) / (self.total_steps as f32);
        let rate = self.config.mutation_rate_init
            + (self.config.mutation_rate_final - self.config.mutation_rate_init) * progress;
        rate.max(self.config.mutation_rate_final)
    }
}
