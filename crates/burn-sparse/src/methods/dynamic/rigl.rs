//! RigL: Rigging the Lottery - Dynamic sparse training
//!
//! **Reference**: Evci et al., "Rigging the Lottery: Making All Tickets Winners", ICML 2020
//! https://arxiv.org/abs/1911.11134

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::SparseMask;

/// Configuration for RigL dynamic sparse training
#[derive(Debug, Clone)]
pub struct RigLConfig {
    /// Target sparsity ratio
    pub sparsity: f32,

    /// Update mask every N training steps
    pub update_frequency: usize,

    /// Fraction of active weights to drop and grow each update
    pub drop_fraction: f32,
}

impl Default for RigLConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.8,
            update_frequency: 100,
            drop_fraction: 0.3,
        }
    }
}

/// RigL: Dynamic sparse training with gradient-based mask updates
///
/// # Algorithm
/// 1. Accumulate gradient magnitudes over multiple steps
/// 2. Every N steps:
///    - Drop: Remove k active weights with smallest |∇W|
///    - Grow: Add k pruned weights with largest |∇W|
/// 3. Reset gradient accumulator
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::methods::dynamic::*;
///
/// let mut rigl = RigL::new(config, initial_mask);
///
/// // Training loop
/// for batch in dataloader {
///     let loss = model.forward(batch);
///     let grads = loss.backward();
///
///     // Update mask based on gradients
///     let new_mask = rigl.update_mask(&grads.weight);
///     if new_mask != old_mask {
///         sparse_linear.update_mask(new_mask);
///     }
/// }
/// ```
pub struct RigL<B: Backend> {
    config: RigLConfig,
    mask: SparseMask<B>,
    step_count: usize,
    grad_accumulator: Option<Tensor<B, 2>>,
}

impl<B: Backend> RigL<B> {
    /// Create new RigL trainer
    pub fn new(config: RigLConfig, initial_mask: SparseMask<B>) -> Self {
        Self {
            config,
            mask: initial_mask,
            step_count: 0,
            grad_accumulator: None,
        }
    }

    /// Update mask based on gradients (call every training step)
    ///
    /// RigL algorithm:
    /// - Prune: Remove bottom-k active weights by **magnitude |W|**
    /// - Grow: Add top-k zero positions by **gradient |g|**
    ///
    /// # Arguments
    /// * `weights` - Current weight values [n_out, n_in]
    /// * `gradients` - Weight gradients [n_out, n_in]
    ///
    /// # Returns
    /// Updated mask (may be same as before if not at update frequency)
    pub fn update_mask(&mut self, weights: &Tensor<B, 2>, gradients: &Tensor<B, 2>) -> SparseMask<B> {
        self.step_count += 1;

        // Check if it's time to update
        if self.step_count % self.config.update_frequency != 0 {
            return self.mask.clone();
        }

        // Get current mask as tensor
        let mask_tensor = self.mask.tensor();
        let device = mask_tensor.device();
        let shape = self.mask.shape();

        // Compute scores (keep as 2D for topk/bottomk functions)
        let weights_abs = weights.clone().abs();
        let grads_abs = gradients.clone().abs();
        let mask_float = mask_tensor.clone().float();

        // Compute number of weights to rewire
        let n_active = self.mask.n_active();
        let k = (n_active as f32 * self.config.drop_fraction) as usize;

        if k == 0 {
            return self.mask.clone();
        }

        // RigL spec: Prune by MAGNITUDE (not gradient!)
        // For bottomk: set pruned positions to +inf (never selected as "smallest")
        // For topk: set active positions to -inf (never selected as "largest")
        let pos_inf = 1e10f32;
        let neg_inf = -1e10f32;

        let active_magnitudes = weights_abs.clone()
            .mul(mask_float.clone())
            .add(mask_float.clone().neg().add_scalar(1.0).mul_scalar(pos_inf));

        // RigL spec: Grow by GRADIENT
        let pruned_gradients = grads_abs
            .mul(mask_float.clone().neg().add_scalar(1.0))
            .add(mask_float.clone().mul_scalar(neg_inf));

        // Find bottom-k active weights (smallest magnitudes)
        // bottomk will skip the +inf values from pruned positions
        let drop_indices = crate::core::utils::bottomk_indices(&active_magnitudes, k);

        // Find top-k pruned weights (largest gradients)
        // topk will skip the -inf values from active positions
        let grow_indices = crate::core::utils::topk_indices(&pruned_gradients, k);

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
}
