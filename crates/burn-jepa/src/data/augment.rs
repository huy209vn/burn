//! Data augmentation techniques for JEPA training.
//!
//! This module provides image augmentation transformations commonly used
//! in self-supervised learning, particularly for JEPA training.
//!
//! # Example
//!
//! ```ignore
//! use burn_jepa::data::augment::{JepaAugmentationConfig, JepaAugmentation};
//!
//! let config = JepaAugmentationConfig::new()
//!     .with_horizontal_flip_prob(0.5)
//!     .with_brightness_range(0.1);
//!
//! let augmentation = config.init();
//! let augmented_image = augmentation.apply(image, &device);
//! ```

use burn::config::Config;
use burn::tensor::{backend::Backend, Distribution, Tensor};

/// Configuration for JEPA data augmentation pipeline.
#[derive(Config, Debug)]
pub struct JepaAugmentationConfig {
    /// Probability of applying horizontal flip (default: 0.5)
    #[config(default = 0.5)]
    pub horizontal_flip_prob: f64,

    /// Range for brightness adjustment (default: 0.0 = disabled)
    /// Applied as: image * (1.0 + uniform(-range, range))
    #[config(default = 0.0)]
    pub brightness_range: f32,

    /// Range for contrast adjustment (default: 0.0 = disabled)
    /// Applied as: (image - mean) * (1.0 + uniform(-range, range)) + mean
    #[config(default = 0.0)]
    pub contrast_range: f32,

    /// Whether to apply random grayscale conversion (default: false)
    #[config(default = false)]
    pub grayscale_enabled: bool,

    /// Probability of applying grayscale (default: 0.2)
    #[config(default = 0.2)]
    pub grayscale_prob: f64,
}

impl JepaAugmentationConfig {
    /// Create a new augmentation configuration with JEPA defaults.
    pub fn jepa_defaults() -> Self {
        Self::new()
            .with_horizontal_flip_prob(0.5)
            .with_brightness_range(0.2)
            .with_contrast_range(0.2)
            .with_grayscale_enabled(true)
            .with_grayscale_prob(0.2)
    }

    /// Initialize the augmentation pipeline.
    pub fn init(&self) -> JepaAugmentation {
        JepaAugmentation {
            horizontal_flip_prob: self.horizontal_flip_prob,
            brightness_range: self.brightness_range,
            contrast_range: self.contrast_range,
            grayscale_enabled: self.grayscale_enabled,
            grayscale_prob: self.grayscale_prob,
        }
    }
}

/// JEPA data augmentation pipeline.
///
/// Applies various image augmentations commonly used in self-supervised learning.
#[derive(Debug, Clone)]
pub struct JepaAugmentation {
    horizontal_flip_prob: f64,
    brightness_range: f32,
    contrast_range: f32,
    grayscale_enabled: bool,
    grayscale_prob: f64,
}

impl JepaAugmentation {
    /// Apply augmentations to a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images [B, C, H, W]
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// * Augmented images [B, C, H, W]
    pub fn apply<B: Backend>(&self, images: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
        let mut result = images;

        // Apply horizontal flip
        if self.horizontal_flip_prob > 0.0 {
            result = self.random_horizontal_flip(result, device);
        }

        // Apply brightness adjustment
        if self.brightness_range > 0.0 {
            result = self.adjust_brightness(result, device);
        }

        // Apply contrast adjustment
        if self.contrast_range > 0.0 {
            result = self.adjust_contrast(result, device);
        }

        // Apply grayscale conversion
        if self.grayscale_enabled && self.grayscale_prob > 0.0 {
            result = self.random_grayscale(result, device);
        }

        result
    }

    /// Apply random horizontal flip to images.
    fn random_horizontal_flip<B: Backend>(
        &self,
        images: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, _c, _h, _w] = images.dims();

        // Generate random values for each sample in batch
        let random_vals = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        // Get flip mask as data
        let flip_mask: Vec<bool> = random_vals
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor")
            .iter()
            .map(|&v| v < self.horizontal_flip_prob as f32)
            .collect();

        // Apply flip per sample (this is a simple implementation)
        // For production, a batched flip operation would be more efficient
        let mut flipped_samples = Vec::with_capacity(batch_size);
        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_size {
            #[allow(clippy::single_range_in_vec_init)]
            let sample = images.clone().slice([i..i + 1]);
            let sample = if flip_mask[i] {
                sample.flip([3]) // Flip along width dimension
            } else {
                sample
            };
            flipped_samples.push(sample);
        }

        Tensor::cat(flipped_samples, 0)
    }

    /// Adjust brightness by a random factor.
    fn adjust_brightness<B: Backend>(
        &self,
        images: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, _, _, _] = images.dims();

        // Generate random brightness factors for each sample
        let factors = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(
                (1.0 - self.brightness_range) as f64,
                (1.0 + self.brightness_range) as f64,
            ),
            device,
        );

        // Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        let factors: Tensor<B, 4> = factors.reshape([batch_size, 1, 1, 1]);

        images * factors
    }

    /// Adjust contrast by a random factor.
    fn adjust_contrast<B: Backend>(
        &self,
        images: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, _, _, _] = images.dims();

        // Compute mean per image (over C, H, W dimensions)
        // We need to flatten C, H, W into one dimension and take mean
        let flattened = images.clone().flatten::<2>(1, 3); // [B, C*H*W]
        let mean: Tensor<B, 1> = flattened.mean_dim(1).squeeze(); // [B]

        // Generate random contrast factors
        let factors = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(
                (1.0 - self.contrast_range) as f64,
                (1.0 + self.contrast_range) as f64,
            ),
            device,
        );

        // Reshape for broadcasting
        let mean: Tensor<B, 4> = mean.reshape([batch_size, 1, 1, 1]);
        let factors: Tensor<B, 4> = factors.reshape([batch_size, 1, 1, 1]);

        // Apply contrast: (image - mean) * factor + mean
        (images - mean.clone()) * factors + mean
    }

    /// Convert images to grayscale with some probability.
    fn random_grayscale<B: Backend>(
        &self,
        images: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, channels, _h, _w] = images.dims();

        if channels != 3 {
            return images; // Only apply to RGB images
        }

        // Generate random values for each sample
        let random_vals = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        let grayscale_mask: Vec<bool> = random_vals
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor")
            .iter()
            .map(|&v| v < self.grayscale_prob as f32)
            .collect();

        let mut processed = Vec::with_capacity(batch_size);
        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_size {
            #[allow(clippy::single_range_in_vec_init)]
            let sample = images.clone().slice([i..i + 1]);
            let sample = if grayscale_mask[i] {
                // Convert to grayscale using standard weights
                // Y = 0.299*R + 0.587*G + 0.114*B
                #[allow(clippy::single_range_in_vec_init)]
                let r = sample.clone().slice([0..1, 0..1]);
                #[allow(clippy::single_range_in_vec_init)]
                let g = sample.clone().slice([0..1, 1..2]);
                #[allow(clippy::single_range_in_vec_init)]
                let b = sample.slice([0..1, 2..3]);

                let gray = r.mul_scalar(0.299) + g.mul_scalar(0.587) + b.mul_scalar(0.114);

                // Repeat grayscale across all channels
                Tensor::cat(vec![gray.clone(), gray.clone(), gray], 1)
            } else {
                sample
            };
            processed.push(sample);
        }

        Tensor::cat(processed, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_augmentation_config() {
        let config = JepaAugmentationConfig::jepa_defaults();
        assert_eq!(config.horizontal_flip_prob, 0.5);
        assert!(config.brightness_range > 0.0);
        assert!(config.grayscale_enabled);
    }

    #[test]
    fn test_augmentation_shape_preserved() {
        let device = <TestBackend as Backend>::Device::default();
        let config = JepaAugmentationConfig::jepa_defaults();
        let augmentation = config.init();

        let images = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Default,
            &device,
        );

        let augmented = augmentation.apply(images.clone(), &device);
        assert_eq!(images.dims(), augmented.dims());
    }

    #[test]
    fn test_no_augmentation() {
        let device = <TestBackend as Backend>::Device::default();
        let config = JepaAugmentationConfig::new(); // All disabled by default ranges
        let augmentation = config.init();

        let images = Tensor::<TestBackend, 4>::random(
            [1, 3, 32, 32],
            Distribution::Default,
            &device,
        );

        let augmented = augmentation.apply(images.clone(), &device);
        assert_eq!(images.dims(), augmented.dims());
    }
}
