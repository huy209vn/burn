//! Binary sparsity mask with efficient indexing.

use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Bool, Distribution, Shape, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use super::utils::percentile;

/// Binary sparsity mask for weight pruning.
///
/// Maintains a boolean mask indicating which weights are active (1) or pruned (0),
/// along with precomputed indices for efficient gather/scatter operations.
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::prelude::*;
///
/// // Create mask from scores (keep top 50%)
/// let mask = SparseMask::from_scores(&scores, 0.5);
///
/// // Apply mask to weights
/// let sparse_weights = mask.apply(&weights);
///
/// // Get sparsity
/// println!("Sparsity: {}", mask.actual_sparsity());
/// ```
#[derive(Clone, Debug)]
pub struct SparseMask<B: Backend> {
    /// Binary mask tensor [n_out, n_in]
    mask: Tensor<B, 2, Bool>,

    /// Target sparsity (0.0 = dense, 1.0 = all pruned)
    sparsity: f32,

    /// Flat indices of active (kept) weights
    active_indices: Vec<usize>,

    /// Flat indices of pruned weights
    pruned_indices: Vec<usize>,

    /// Shape [n_out, n_in]
    shape: [usize; 2],

    /// Device for tensor operations
    device: B::Device,
}

impl<B: Backend> SparseMask<B> {
    /// Create mask from importance scores using top-k selection.
    ///
    /// # Arguments
    ///
    /// * `scores` - Importance scores [n_out, n_in]
    /// * `sparsity` - Target sparsity ratio (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Binary mask keeping the top (1-sparsity) weights
    pub fn from_scores(scores: &Tensor<B, 2>, sparsity: f32) -> Self {
        let threshold = percentile(scores, sparsity * 100.0);
        let mask = scores.clone().greater_elem(threshold);
        Self::from_tensor(mask)
    }

    /// Create mask from existing boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean tensor [n_out, n_in] where true = keep, false = prune
    ///
    /// # Returns
    ///
    /// SparseMask with precomputed indices
    pub fn from_tensor(mask: Tensor<B, 2, Bool>) -> Self {
        let shape = mask.dims();
        let device = mask.device();
        let mask_data = mask.clone().into_data();
        let mask_values: Vec<bool> = mask_data.to_vec().unwrap();

        let mut active = Vec::new();
        let mut pruned = Vec::new();

        for (i, &is_active) in mask_values.iter().enumerate() {
            if is_active {
                active.push(i);
            } else {
                pruned.push(i);
            }
        }

        let total = shape[0] * shape[1];
        let sparsity = pruned.len() as f32 / total as f32;

        Self {
            mask,
            sparsity,
            active_indices: active,
            pruned_indices: pruned,
            shape,
            device,
        }
    }

    /// Create random mask with specified sparsity.
    ///
    /// # Arguments
    ///
    /// * `shape` - Mask shape [n_out, n_in]
    /// * `sparsity` - Target sparsity ratio
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// Randomly initialized mask
    pub fn random(shape: [usize; 2], sparsity: f32, device: &B::Device) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let total = shape[0] * shape[1];
        let n_pruned = (total as f32 * sparsity) as usize;

        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut thread_rng());

        let pruned = indices[..n_pruned].to_vec();
        let active = indices[n_pruned..].to_vec();

        let mut mask_data = vec![true; total];
        for &idx in &pruned {
            mask_data[idx] = false;
        }

        let mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, Shape::new(shape)),
            device,
        );

        Self {
            mask,
            sparsity,
            active_indices: active,
            pruned_indices: pruned,
            shape,
            device: device.clone(),
        }
    }

    /// Apply mask to weights (zero out pruned positions).
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight tensor [n_out, n_in]
    ///
    /// # Returns
    ///
    /// Sparse weights with pruned positions set to zero
    pub fn apply(&self, weights: &Tensor<B, 2>) -> Tensor<B, 2> {
        weights.clone().mask_fill(self.mask.clone().bool_not(), 0.0)
    }

    /// Get number of active (kept) weights.
    pub fn n_active(&self) -> usize {
        self.active_indices.len()
    }

    /// Get number of pruned weights.
    pub fn n_pruned(&self) -> usize {
        self.pruned_indices.len()
    }

    /// Get actual sparsity ratio.
    pub fn actual_sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get mask shape [n_out, n_in].
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Get reference to underlying boolean mask tensor.
    pub fn tensor(&self) -> &Tensor<B, 2, Bool> {
        &self.mask
    }

    /// Get device.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get active indices (for internal use).
    pub(crate) fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    /// Get pruned indices (for internal use).
    pub(crate) fn pruned_indices(&self) -> &[usize] {
        &self.pruned_indices
    }

    /// Compute Hamming distance to another mask (number of differing positions).
    ///
    /// # Arguments
    ///
    /// * `other` - Another mask with the same shape
    ///
    /// # Returns
    ///
    /// Number of positions where masks differ
    pub fn hamming_distance(&self, other: &Self) -> usize {
        assert_eq!(self.shape, other.shape, "Masks must have the same shape");

        // XOR the masks to find differences
        let mask1_int = self.mask.clone().int();
        let mask2_int = other.mask.clone().int();
        let xor = (mask1_int.clone() - mask2_int.clone()).abs();

        use burn_core::tensor::ElementConversion;

        // Convert to float for into_scalar()
        let diff_count_float = xor.float().sum().into_scalar().elem::<f32>();
        diff_count_float as usize
    }

    /// Get complement mask (swap active <-> pruned).
    pub fn complement(&self) -> Self {
        Self {
            mask: self.mask.clone().bool_not(),
            sparsity: 1.0 - self.sparsity,
            active_indices: self.pruned_indices.clone(),
            pruned_indices: self.active_indices.clone(),
            shape: self.shape,
            device: self.device.clone(),
        }
    }

    /// Extract values at active positions as 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor [n_out, n_in]
    ///
    /// # Returns
    ///
    /// Values at active positions [n_active]
    pub fn gather_active(&self, tensor: &Tensor<B, 2>) -> Tensor<B, 1> {
        use burn_core::tensor::Int;

        let flat = tensor.clone().flatten(0, 1);

        // Convert Vec<usize> to Vec<i32>
        let indices_i32: Vec<i32> = self.active_indices.iter().map(|&i| i as i32).collect();
        let active_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices_i32, Shape::new([self.active_indices.len()])),
            &self.device,
        );

        flat.gather(0, active_tensor)
    }

    /// Extract values at pruned positions as 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor [n_out, n_in]
    ///
    /// # Returns
    ///
    /// Values at pruned positions [n_pruned]
    pub fn gather_pruned(&self, tensor: &Tensor<B, 2>) -> Tensor<B, 1> {
        use burn_core::tensor::Int;

        let flat = tensor.clone().flatten(0, 1);

        // Convert Vec<usize> to Vec<i32>
        let indices_i32: Vec<i32> = self.pruned_indices.iter().map(|&i| i as i32).collect();
        let pruned_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices_i32, Shape::new([self.pruned_indices.len()])),
            &self.device,
        );

        flat.gather(0, pruned_tensor)
    }

    /// Scatter values to active positions, creating 2D tensor.
    ///
    /// # Arguments
    ///
    /// * `values` - Values to scatter [n_active]
    ///
    /// # Returns
    ///
    /// Tensor with values at active positions, zeros elsewhere [n_out, n_in]
    pub fn scatter_active(&self, values: &Tensor<B, 1>) -> Tensor<B, 2> {
        use burn_core::tensor::Int;

        assert_eq!(
            values.dims()[0],
            self.n_active(),
            "Values length must match number of active positions"
        );

        let total = self.shape[0] * self.shape[1];
        let mut result = Tensor::<B, 1>::zeros([total], &self.device);

        // Convert Vec<usize> to Vec<i32>
        let indices_i32: Vec<i32> = self.active_indices.iter().map(|&i| i as i32).collect();
        let active_indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices_i32, Shape::new([self.active_indices.len()])),
            &self.device,
        );

        result = result.scatter(0, active_indices, values.clone());
        result.reshape(self.shape)
    }

    /// Scatter values to pruned positions, creating 2D tensor.
    ///
    /// # Arguments
    ///
    /// * `values` - Values to scatter [n_pruned]
    ///
    /// # Returns
    ///
    /// Tensor with values at pruned positions, zeros elsewhere [n_out, n_in]
    pub fn scatter_pruned(&self, values: &Tensor<B, 1>) -> Tensor<B, 2> {
        use burn_core::tensor::Int;

        assert_eq!(
            values.dims()[0],
            self.n_pruned(),
            "Values length must match number of pruned positions"
        );

        let total = self.shape[0] * self.shape[1];
        let mut result = Tensor::<B, 1>::zeros([total], &self.device);

        // Convert Vec<usize> to Vec<i32>
        let indices_i32: Vec<i32> = self.pruned_indices.iter().map(|&i| i as i32).collect();
        let pruned_indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices_i32, Shape::new([self.pruned_indices.len()])),
            &self.device,
        );

        result = result.scatter(0, pruned_indices, values.clone());
        result.reshape(self.shape)
    }
}

/// Serializable mask data for persistence.
#[derive(Serialize, Deserialize)]
struct SparseMaskData {
    mask_values: Vec<bool>,
    shape: [usize; 2],
    sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend as TB;

    #[test]
    fn test_from_scores() {
        let scores = Tensor::<TB, 2>::from_data(
            [
                [1.0, 5.0, 3.0],
                [4.0, 2.0, 6.0],
            ],
            &Default::default(),
        );

        let mask = SparseMask::from_scores(&scores, 0.5);

        assert_eq!(mask.shape(), [2, 3]);
        assert_eq!(mask.n_active(), 3);
        assert_eq!(mask.n_pruned(), 3);
        assert!((mask.actual_sparsity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_apply_mask() {
        let weights = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &Default::default(),
        );

        let mask_data = Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false, true],
                [false, true, true],
            ],
            &Default::default(),
        );

        let mask = SparseMask::from_tensor(mask_data);
        let sparse = mask.apply(&weights);

        let sparse_data: Vec<f32> = sparse.into_data().to_vec().unwrap();

        assert_eq!(sparse_data, vec![1.0, 0.0, 3.0, 0.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gather_scatter() {
        let device = Default::default();
        let mask_data = Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false, true],
                [false, true, true],
            ],
            &device,
        );
        let mask = SparseMask::from_tensor(mask_data);

        let tensor = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &device,
        );

        // Gather active values
        let active = mask.gather_active(&tensor);
        assert_eq!(active.dims()[0], 4);

        // Scatter back
        let scattered = mask.scatter_active(&active);
        assert_eq!(scattered.dims(), [2, 3]);
    }

    #[test]
    fn test_hamming_distance() {
        let device = Default::default();
        let mask1 = SparseMask::from_tensor(Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false],
                [true, false],
            ],
            &device,
        ));

        let mask2 = SparseMask::from_tensor(Tensor::<TB, 2, Bool>::from_data(
            [
                [true, true],
                [false, false],
            ],
            &device,
        ));

        let distance = mask1.hamming_distance(&mask2);
        assert_eq!(distance, 2); // Differ at positions (0,1) and (1,0)
    }

    #[test]
    fn test_complement() {
        let device = Default::default();
        let mask = SparseMask::from_tensor(Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false],
                [true, false],
            ],
            &device,
        ));

        let comp = mask.complement();

        assert_eq!(mask.n_active(), comp.n_pruned());
        assert_eq!(mask.n_pruned(), comp.n_active());
        assert!((mask.actual_sparsity() + comp.actual_sparsity() - 1.0).abs() < 1e-5);
    }
}
