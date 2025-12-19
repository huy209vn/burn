//! Common utilities and helpers for testing `burn-jepa` components.
//!
//! This module provides functions to easily set up test environments, generate
//! dummy data, and define common test backends, ensuring consistency and
//! reproducibility across unit and integration tests.

use burn::tensor::{backend::Backend, Distribution, Int, Tensor};

/// # Generates a dummy image tensor.
///
/// Creates a random image tensor with the specified dimensions for testing purposes.
///
/// # Arguments
///
/// * `batch_size`: The number of images in the batch.
/// * `channels`: The number of image channels (e.g., 3 for RGB).
/// * `height`: The height of the image.
/// * `width`: The width of the image.
/// * `device`: The device on which the tensor should be created.
///
/// # Returns
///
/// * A `Tensor<B, 4>` representing a batch of random images.
pub fn generate_dummy_image<B: Backend>(
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    Tensor::random(
        [batch_size, channels, height, width],
        Distribution::Default,
        device,
    )
}

/// # Generates dummy mask indices.
///
/// Creates a tensor of random integer indices for testing masking operations.
///
/// # Arguments
///
/// * `batch_size`: The number of samples in the batch.
/// * `num_indices`: The number of indices to generate for each sample.
/// * `max_index`: The upper bound (exclusive) for the generated indices.
/// * `device`: The device on which the tensor should be created.
///
/// # Returns
///
/// * A `Tensor<B, 2, Int>` representing a batch of random indices.
pub fn generate_dummy_mask_indices<B: Backend>(
    batch_size: usize,
    num_indices: usize,
    max_index: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 2>::random(
        [batch_size, num_indices],
        Distribution::Uniform(0.0, max_index as f64),
        device,
    )
    .int()
}