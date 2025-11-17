//! Calibration data management.

use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Tensor};

/// Wrapper for calibration dataset.
///
/// Manages a collection of activation samples used for computing
/// importance scores and error distributions.
///
/// # Example
///
/// ```rust,ignore
/// use burn_sparse::prelude::*;
///
/// // Collect activations during forward passes
/// let samples = vec![activation1, activation2];
/// let calibration = CalibrationData::from_samples(samples);
///
/// println!("Using {} calibration samples", calibration.len());
/// ```
#[derive(Clone, Debug)]
pub struct CalibrationData<B: Backend> {
    /// Stacked samples [n_samples, n_features]
    samples: Tensor<B, 2>,
    n_samples: usize,
}

impl<B: Backend> CalibrationData<B> {
    /// Create from a single tensor containing all samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Tensor [n_samples, n_features]
    ///
    /// # Returns
    ///
    /// CalibrationData wrapping the samples
    pub fn new(samples: Tensor<B, 2>) -> Self {
        let n_samples = samples.dims()[0];
        Self { samples, n_samples }
    }

    /// Create from a vector of sample tensors.
    ///
    /// # Arguments
    ///
    /// * `samples` - Vector of tensors, each [batch, n_features]
    ///
    /// # Returns
    ///
    /// CalibrationData with samples concatenated
    ///
    /// # Panics
    ///
    /// Panics if samples is empty
    pub fn from_samples(samples: Vec<Tensor<B, 2>>) -> Self {
        assert!(!samples.is_empty(), "Cannot create CalibrationData from empty samples");
        let stacked = Tensor::cat(samples, 0);
        Self::new(stacked)
    }

    /// Get number of calibration samples.
    pub fn len(&self) -> usize {
        self.n_samples
    }

    /// Check if calibration data is empty.
    pub fn is_empty(&self) -> bool {
        self.n_samples == 0
    }

    /// Get reference to all samples as a single tensor.
    ///
    /// # Returns
    ///
    /// Tensor [n_samples, n_features]
    pub fn samples(&self) -> &Tensor<B, 2> {
        &self.samples
    }

    /// Take first n samples.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to take
    ///
    /// # Returns
    ///
    /// CalibrationData with first n samples
    ///
    /// # Panics
    ///
    /// Panics if n > number of available samples
    pub fn take(&self, n: usize) -> Self {
        assert!(
            n <= self.n_samples,
            "Cannot take {} samples, only {} available",
            n,
            self.n_samples
        );

        let taken = self.samples.clone().slice([0..n]);
        Self::new(taken)
    }

    /// Get device of calibration data.
    pub fn device(&self) -> B::Device {
        self.samples.device()
    }

    /// Iterate over individual samples.
    ///
    /// # Returns
    ///
    /// Iterator yielding [n_features] tensors
    pub fn iter(&self) -> CalibrationDataIter<B> {
        CalibrationDataIter {
            data: self,
            current: 0,
        }
    }
}

/// Iterator over calibration samples.
pub struct CalibrationDataIter<'a, B: Backend> {
    data: &'a CalibrationData<B>,
    current: usize,
}

impl<'a, B: Backend> Iterator for CalibrationDataIter<'a, B> {
    type Item = Tensor<B, 1>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.data.n_samples {
            return None;
        }

        let sample = self.data.samples.clone().slice([self.current..self.current + 1]);
        self.current += 1;

        Some(sample.squeeze::<1>())
    }
}

impl<'a, B: Backend> ExactSizeIterator for CalibrationDataIter<'a, B> {
    fn len(&self) -> usize {
        self.data.n_samples - self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend as TB;

    #[test]
    fn test_calibration_data_new() {
        let samples = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &Default::default(),
        );

        let cal = CalibrationData::new(samples);

        assert_eq!(cal.len(), 2);
        assert!(!cal.is_empty());
        assert_eq!(cal.samples().dims(), [2, 3]);
    }

    #[test]
    fn test_from_samples() {
        let sample1 = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
            ],
            &Default::default(),
        );

        let sample2 = Tensor::<TB, 2>::from_data(
            [
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            &Default::default(),
        );

        let cal = CalibrationData::from_samples(vec![sample1, sample2]);

        assert_eq!(cal.len(), 3); // 1 + 2 samples
        assert_eq!(cal.samples().dims(), [3, 2]);
    }

    #[test]
    fn test_take() {
        let samples = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            &Default::default(),
        );

        let cal = CalibrationData::new(samples);
        let subset = cal.take(2);

        assert_eq!(subset.len(), 2);
        assert_eq!(subset.samples().dims(), [2, 2]);
    }

    #[test]
    fn test_iter() {
        let samples = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            &Default::default(),
        );

        let cal = CalibrationData::new(samples);
        let mut iter = cal.iter();

        assert_eq!(iter.len(), 2);

        let first = iter.next().unwrap();
        assert_eq!(first.dims(), [2]);

        let second = iter.next().unwrap();
        assert_eq!(second.dims(), [2]);

        assert!(iter.next().is_none());
    }
}
