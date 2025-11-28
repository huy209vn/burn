//! Utility functions for tensor operations.

use burn_core::tensor::{backend::Backend, Tensor};
use alloc::vec::Vec;

/// Compute percentile of a 2D tensor.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `percentile` - Percentile value (0-100)
///
/// # Returns
///
/// The value at the given percentile
pub fn percentile<B: Backend>(tensor: &Tensor<B, 2>, percentile: f32) -> f32 {
    let data = tensor.clone().flatten::<1>(0, 1).into_data();
    let mut values: Vec<f32> = data.to_vec().unwrap();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let idx = ((percentile / 100.0) * values.len() as f32) as usize;
    let clamped_idx = idx.min(values.len().saturating_sub(1));

    values[clamped_idx]
}

/// Get indices of top-k largest values in a flattened tensor.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `k` - Number of top values
///
/// # Returns
///
/// Vector of flat indices
pub fn topk_indices<B: Backend>(tensor: &Tensor<B, 2>, k: usize) -> Vec<usize> {
    let data = tensor.clone().flatten::<1>(0, 1).into_data();
    let values: Vec<f32> = data.to_vec().unwrap();

    let mut indexed: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Sort descending by value
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

/// Get indices of bottom-k smallest values in a flattened tensor.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `k` - Number of bottom values
///
/// # Returns
///
/// Vector of flat indices
pub fn bottomk_indices<B: Backend>(tensor: &Tensor<B, 2>, k: usize) -> Vec<usize> {
    let data = tensor.clone().flatten::<1>(0, 1).into_data();
    let values: Vec<f32> = data.to_vec().unwrap();

    let mut indexed: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Sort ascending by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

/// Compute reconstruction error: ||W_dense @ x - W_sparse @ x||²
///
/// # Arguments
///
/// * `w_dense` - Dense weight matrix [m, n]
/// * `w_sparse` - Sparse weight matrix [m, n]
/// * `x` - Input activations [batch, n]
///
/// # Returns
///
/// Per-sample squared reconstruction error [batch]
pub fn reconstruction_error<B: Backend>(
    w_dense: &Tensor<B, 2>,
    w_sparse: &Tensor<B, 2>,
    x: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    // y_dense = x @ W^T  (note: we need to transpose W)
    let y_dense = x.clone().matmul(w_dense.clone().transpose());
    let y_sparse = x.clone().matmul(w_sparse.clone().transpose());

    // ||y_dense - y_sparse||² per sample
    (y_dense - y_sparse)
        .powf_scalar(2.0)
        .sum_dim(1) // [batch, 1]
        .flatten::<1>(0, 1) // [batch]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend as TB;

    #[test]
    fn test_percentile() {
        let tensor = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &Default::default(),
        );

        let p50 = percentile(&tensor, 50.0);
        // Values: [1, 2, 3, 4, 5, 6], 50th percentile at index 3 -> value 4
        assert!((p50 - 4.0).abs() < 0.1);

        let p100 = percentile(&tensor, 100.0);
        assert!((p100 - 6.0).abs() < 0.1); // Max should be 6.0
    }

    #[test]
    fn test_topk_indices() {
        let tensor = Tensor::<TB, 2>::from_data(
            [
                [1.0, 5.0, 3.0],
                [4.0, 2.0, 6.0],
            ],
            &Default::default(),
        );

        let top2 = topk_indices(&tensor, 2);
        assert_eq!(top2.len(), 2);
        // Should contain indices for 6.0 (index 5) and 5.0 (index 1)
        assert!(top2.contains(&5)); // 6.0
        assert!(top2.contains(&1)); // 5.0
    }

    #[test]
    fn test_bottomk_indices() {
        let tensor = Tensor::<TB, 2>::from_data(
            [
                [1.0, 5.0, 3.0],
                [4.0, 2.0, 6.0],
            ],
            &Default::default(),
        );

        let bottom2 = bottomk_indices(&tensor, 2);
        assert_eq!(bottom2.len(), 2);
        // Should contain indices for 1.0 (index 0) and 2.0 (index 4)
        assert!(bottom2.contains(&0)); // 1.0
        assert!(bottom2.contains(&4)); // 2.0
    }

    #[test]
    fn test_reconstruction_error() {
        let w_dense = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            &Default::default(),
        );

        let w_sparse = Tensor::<TB, 2>::from_data(
            [
                [1.0, 0.0], // Pruned second column
                [3.0, 0.0],
            ],
            &Default::default(),
        );

        let x = Tensor::<TB, 2>::from_data(
            [
                [1.0, 1.0],
            ],
            &Default::default(),
        );

        let error = reconstruction_error(&w_dense, &w_sparse, &x);
        let error_val: f32 = error.into_data().to_vec().unwrap()[0];

        // y_dense = [1,1] @ [[1,3],[2,4]] = [3, 7]
        // y_sparse = [1,1] @ [[1,3],[0,0]] = [1, 3]
        // error = (3-1)² + (7-3)² = 4 + 16 = 20
        assert!((error_val - 20.0).abs() < 1e-5);
    }
}
