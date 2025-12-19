//! Full training example for JEPA with Burn's Training Framework (TUI).
//!
//! This example demonstrates JEPA training using burn-train's LearnerBuilder
//! which provides:
//! - TUI dashboard with live metrics visualization
//! - Automatic checkpointing and model saving
//! - Learning rate scheduling integration
//! - Metric logging (loss tracking)
//!
//! Run with CUDA:
//!   cargo run --example train_jepa -p burn-jepa --release --features cuda
//!
//! Run with CPU:
//!   cargo run --example train_jepa -p burn-jepa --release

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::lr_scheduler::constant::ConstantLr;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend as BackendTrait;
use burn::train::metric::LossMetric;
use burn::train::{LearnerBuilder, LearningStrategy};
use burn_jepa::model::jepa::{JepaBatch, JepaConfig};

// Backend configuration - CUDA by default when available
#[cfg(feature = "cuda")]
mod backend {
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::backend::Autodiff;
    pub type Backend = Autodiff<Cuda>;
    pub fn device() -> CudaDevice {
        CudaDevice::default()
    }
}

#[cfg(not(feature = "cuda"))]
mod backend {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    pub type Backend = Autodiff<NdArray<f32>>;
    pub fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
}

use backend::{device, Backend};

/// Synthetic dataset that generates random images for demo purposes.
/// In practice, replace this with your actual dataset (e.g., ImageNet).
struct SyntheticImageDataset {
    size: usize,
    image_size: usize,
}

impl SyntheticImageDataset {
    fn new(size: usize, image_size: usize) -> Self {
        Self { size, image_size }
    }
}

/// Raw image item before batching - stores metadata only
#[derive(Clone, Debug)]
struct ImageItem {
    image_size: usize,
}

impl Dataset<ImageItem> for SyntheticImageDataset {
    fn get(&self, _index: usize) -> Option<ImageItem> {
        Some(ImageItem {
            image_size: self.image_size,
        })
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Batcher that creates JepaBatch from raw items
/// In a real scenario, this would load and preprocess actual images
#[derive(Clone, Default)]
struct JepaBatcher {}

impl<B: BackendTrait> Batcher<B, ImageItem, JepaBatch<B>> for JepaBatcher {
    fn batch(&self, items: Vec<ImageItem>, device: &B::Device) -> JepaBatch<B> {
        let batch_size = items.len();
        let image_size = items[0].image_size;

        // Generate random images (in real use, load and preprocess actual images)
        let images = Tensor::random(
            [batch_size, 3, image_size, image_size],
            burn::tensor::Distribution::Default,
            device,
        );

        JepaBatch { images }
    }
}

/// Training configuration
struct TrainingConfig {
    // Model architecture
    image_size: usize,
    patch_size: usize,
    embed_dim: usize,
    encoder_layers: usize,
    predictor_layers: usize,
    num_heads: usize,

    // Training hyperparameters
    num_epochs: usize,
    batch_size: usize,
    num_samples: usize,
    learning_rate: f64,
    weight_decay: f32,

    // Loss
    var_reg_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            // ViT-Small configuration
            image_size: 224,
            patch_size: 16,
            embed_dim: 384,
            encoder_layers: 12,
            predictor_layers: 6,
            num_heads: 6,

            // Training hyperparameters (scaled down for demo)
            num_epochs: 5,
            batch_size: 8,
            num_samples: 1000,
            learning_rate: 1.5e-4,
            weight_decay: 0.05,

            // Variance regularization to prevent collapse
            var_reg_weight: 5.0,
        }
    }
}

fn main() {
    let config = TrainingConfig::default();
    let device = device();

    println!("===========================================");
    println!("   JEPA Training with Burn Train TUI");
    println!("===========================================\n");

    println!("Model Configuration:");
    println!("  Image size: {}x{}", config.image_size, config.image_size);
    println!("  Patch size: {}x{}", config.patch_size, config.patch_size);
    println!(
        "  Grid: {}x{} patches",
        config.image_size / config.patch_size,
        config.image_size / config.patch_size
    );
    println!("  Embed dim: {}", config.embed_dim);
    println!("  Encoder layers: {}", config.encoder_layers);
    println!("  Predictor layers: {}", config.predictor_layers);
    println!("  Attention heads: {}", config.num_heads);
    println!();

    println!("Training Configuration:");
    println!("  Epochs: {}", config.num_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Dataset size: {} samples", config.num_samples);
    println!("  Learning rate: {:.2e}", config.learning_rate);
    println!("  Weight decay: {}", config.weight_decay);
    println!("  Variance reg weight: {}", config.var_reg_weight);
    println!();

    // Create JEPA model
    let model_config = JepaConfig::new()
        .with_image_size(config.image_size)
        .with_patch_size(config.patch_size)
        .with_embed_dim(config.embed_dim)
        .with_n_layers(config.encoder_layers)
        .with_n_heads(config.num_heads)
        .with_predictor_n_layers(config.predictor_layers)
        .with_var_reg_weight(config.var_reg_weight);

    let model = model_config.init::<Backend>(&device);

    // Create datasets
    let train_dataset = SyntheticImageDataset::new(config.num_samples, config.image_size);
    let valid_dataset = SyntheticImageDataset::new(config.num_samples / 10, config.image_size);

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(JepaBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(JepaBatcher::default())
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(valid_dataset);

    // Create optimizer
    let optimizer = AdamWConfig::new().with_weight_decay(config.weight_decay);

    // Create learning rate scheduler (constant for simplicity)
    let lr_scheduler = ConstantLr::new(config.learning_rate);

    // Build learner with TUI dashboard
    let learner = LearnerBuilder::new("./artifacts")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            optimizer.init(),
            lr_scheduler,
            LearningStrategy::SingleDevice(device),
        );

    println!("Starting training with TUI dashboard...\n");

    // Train with TUI
    let mut result = learner.fit(dataloader_train, dataloader_valid);

    println!("\n===========================================");
    println!("         Training Complete!");
    println!("===========================================");
    println!("Model checkpoints saved to: ./artifacts/");
    println!();
    println!("The trained model can be used for:");
    println!("  - Linear probing on downstream tasks");
    println!("  - Fine-tuning for classification");
    println!("  - Feature extraction for other models");

    // Close the renderer properly
    result.renderer.manual_close();
}
