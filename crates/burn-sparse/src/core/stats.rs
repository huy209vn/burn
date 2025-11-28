//! Activation statistics for importance scoring.

use burn_core::tensor::{backend::Backend, Tensor};

/// Statistical moments of layer activations.
///
/// Computes and stores mean, variance, and L2 norm statistics
/// across calibration samples for use in importance scoring.
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::prelude::*;
///
/// // Collect activations during forward pass
/// let samples = vec![activation1, activation2, activation3];
/// let stats = ActivationStats::from_samples(&samples);
///
/// // Use L2 norm for Wanda scoring
/// let norms = stats.l2_norm();
/// ```
#[derive(Clone, Debug)]
pub struct ActivationStats<B: Backend> {
    /// Mean per feature [n_features]
    mean: Tensor<B, 1>,

    /// Variance per feature [n_features]
    variance: Tensor<B, 1>,

    /// L2 norm per feature [n_features]
    l2_norm: Tensor<B, 1>,

    /// Number of samples used
    n_samples: usize,
}

impl<B: Backend> ActivationStats<B> {
    /// Compute statistics from a batch of activation samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Vector of activation tensors, each [batch, n_features]
    ///
    /// # Returns
    ///
    /// ActivationStats computed across all samples
    ///
    /// # Panics
    ///
    /// Panics if samples is empty
    pub fn from_samples(samples: &[Tensor<B, 2>]) -> Self {
        assert!(!samples.is_empty(), "Cannot compute stats from empty samples");

        // Concatenate all samples along batch dimension
        let all_samples = Tensor::cat(samples.to_vec(), 0); // [total_batch, n_features]
        let n_samples = all_samples.dims()[0];

        // Compute mean across all samples
        let mean = all_samples.clone().mean_dim(0).squeeze::<1>(); // [n_features]

        // Compute variance: E[(x - mean)²]
        let centered = all_samples.clone() - mean.clone().unsqueeze();
        let variance = centered
            .powf_scalar(2.0)
            .mean_dim(0)
            .squeeze::<1>(); // [n_features]

        // Compute L2 norm: sqrt(sum(x²))
        let l2_norm = all_samples.clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            .squeeze::<1>(); // [n_features]

        Self {
            mean,
            variance,
            l2_norm,
            n_samples,
        }
    }

    /// Get mean per feature.
    pub fn mean(&self) -> &Tensor<B, 1> {
        &self.mean
    }

    /// Get variance per feature.
    pub fn variance(&self) -> &Tensor<B, 1> {
        &self.variance
    }

    /// Get standard deviation per feature.
    pub fn std(&self) -> Tensor<B, 1> {
        self.variance.clone().sqrt()
    }

    /// Get L2 norm per feature.
    pub fn l2_norm(&self) -> &Tensor<B, 1> {
        &self.l2_norm
    }

    /// Get number of samples used to compute statistics.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend as TB;

    #[test]
    fn test_activation_stats() {
        let sample1 = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &Default::default(),
        );

        let sample2 = Tensor::<TB, 2>::from_data(
            [
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
            ],
            &Default::default(),
        );

        let stats = ActivationStats::from_samples(&[sample1, sample2]);

        assert_eq!(stats.n_samples(), 4);

        // Check mean
        let mean_data: Vec<f32> = stats.mean().clone().into_data().to_vec().unwrap();
        // Mean of [1,2,3,4,5] for first feature = 3.0
        // Mean of [2,3,4,5,6] for second feature = 4.0
        // Mean of [3,4,5,6,7] for third feature = 5.0
        assert!((mean_data[0] - 3.0).abs() < 0.1);
        assert!((mean_data[1] - 4.0).abs() < 0.1);
        assert!((mean_data[2] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_std() {
        let sample = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            &Default::default(),
        );

        let stats = ActivationStats::from_samples(&[sample]);

        let std = stats.std();
        assert_eq!(std.dims(), [2]);
    }

    #[test]
    fn test_l2_norm() {
        let sample = Tensor::<TB, 2>::from_data(
            [
                [3.0, 4.0],
            ],
            &Default::default(),
        );

        let stats = ActivationStats::from_samples(&[sample]);

        let l2 = stats.l2_norm();
        let l2_data: Vec<f32> = l2.clone().into_data().to_vec().unwrap();

        // L2 norm of [3] = 3.0
        // L2 norm of [4] = 4.0
        assert!((l2_data[0] - 3.0).abs() < 0.1);
        assert!((l2_data[1] - 4.0).abs() < 0.1);
    }
}
