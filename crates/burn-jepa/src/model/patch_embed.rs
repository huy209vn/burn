//! The Patch Embedding module for Vision Transformers.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// # Patch Embedding Configuration.
#[derive(Config, Debug)]
pub struct PatchEmbedConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        let num_patches = (self.image_size / self.patch_size) * (self.image_size / self.patch_size);
        let patch_dim = self.patch_size * self.patch_size * self.in_channels;

        // Linear projection from patch_dim to embed_dim
        let proj = LinearConfig::new(patch_dim, self.embed_dim)
            .with_bias(true)
            .init(device);

        // Learnable position embeddings: [1, num_patches, embed_dim]
        let pos_embed = Tensor::random(
            [1, num_patches, self.embed_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        PatchEmbed {
            patch_size: self.patch_size,
            proj,
            pos_embed: Param::from_tensor(pos_embed),
        }
    }
}

/// # Patch Embedding Module.
///
/// Converts an input image (B, C, H, W) into a sequence of patch embeddings (B, N, D).
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub patch_size: usize,
    pub proj: Linear<B>,
    pub pos_embed: Param<Tensor<B, 3>>,
}

impl<B: Backend> PatchEmbed<B> {
    /// Forward pass: converts images [B, C, H, W] to patches [B, N, D]
    ///
    /// # Arguments
    /// * `images` - Input images with shape [B, C, H, W]
    ///
    /// # Returns
    /// * Patch embeddings with shape [B, N, D] where N = (H/P)*(W/P)
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, channels, height, width] = images.dims();
        let p = self.patch_size;

        // Calculate number of patches
        let num_patches_h = height / p;
        let num_patches_w = width / p;
        let num_patches = num_patches_h * num_patches_w;

        // Reshape image into patches: [B, C, H, W] -> [B, N, P*P*C]
        // We do this by:
        // 1. Reshape to [B, C, num_patches_h, P, num_patches_w, P]
        // 2. Permute to [B, num_patches_h, num_patches_w, P, P, C]
        // 3. Reshape to [B, N, P*P*C]
        let patches = images
            .reshape([batch_size, channels, num_patches_h, p, num_patches_w, p])
            .swap_dims(2, 3)
            .swap_dims(4, 5)
            .reshape([batch_size, num_patches, p * p * channels]);

        // Linear projection: [B, N, P*P*C] -> [B, N, D]
        let embeddings = self.proj.forward(patches);

        // Add position embeddings: [B, N, D] + [1, N, D]
        embeddings + self.pos_embed.val()
    }
}