//! Evaluation example for JEPA learned representations.
//!
//! This example shows how to extract representations from a trained JEPA model
//! and evaluate their quality for downstream tasks.
//!
//! Run with CPU:
//!   cargo run --example eval_representation -p burn-jepa
//!
//! Run with CUDA:
//!   cargo run --example eval_representation -p burn-jepa --features cuda

use burn::tensor::Tensor;
use burn_jepa::model::jepa::JepaConfig;

#[cfg(feature = "cuda")]
mod backend {
    pub use burn::backend::cuda::{Cuda, CudaDevice};
    pub type Backend = Cuda;
    pub fn device() -> CudaDevice {
        CudaDevice::default()
    }
}

#[cfg(not(feature = "cuda"))]
mod backend {
    pub use burn::backend::ndarray::NdArrayDevice;
    pub use burn::backend::NdArray;
    pub type Backend = NdArray<f32>;
    pub fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
}

use backend::{device, Backend};

fn main() {
    let device = device();

    // Create JEPA model (in practice, load pretrained weights)
    let config = JepaConfig::new();
    let model = config.init::<Backend>(&device);

    println!("JEPA Model for evaluation:");
    println!("  Image size: {}x{}", config.image_size, config.image_size);
    println!("  Embed dim: {}", config.embed_dim);

    // Generate a test image
    let test_image = Tensor::random(
        [1, 3, config.image_size, config.image_size],
        burn::tensor::Distribution::Default,
        &device,
    );

    // Extract patch embeddings
    let patches = model.patch_embed.forward(test_image);
    let [batch, num_patches, embed_dim] = patches.dims();

    println!("\nPatch embeddings extracted:");
    println!("  Batch size: {}", batch);
    println!("  Number of patches: {}", num_patches);
    println!("  Embedding dimension: {}", embed_dim);

    // Get teacher encoder representations (what JEPA learns to predict)
    let representations = model.teacher_encoder.forward(patches);
    let [_, seq_len, repr_dim] = representations.dims();

    println!("\nTeacher representations:");
    println!("  Sequence length: {}", seq_len);
    println!("  Representation dimension: {}", repr_dim);

    // Compute mean representation (global image representation)
    // Use reshape to get [batch, embed_dim] from [batch, 1, embed_dim]
    let mean_repr = representations.mean_dim(1);
    let [batch_out, _, global_dim] = mean_repr.dims();

    println!("\nGlobal representation (mean pooled): [batch={}, dim={}]", batch_out, global_dim);
    println!("\nRepresentation extraction complete!");
    println!("These representations can be used for downstream tasks like:");
    println!("  - Image classification (linear probe)");
    println!("  - Object detection");
    println!("  - Semantic segmentation");
}
