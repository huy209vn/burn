# burn-jepa

A [Burn](https://github.com/tracel-ai/burn) implementation of **JEPA** (Joint Embedding Predictive Architecture), a self-supervised learning framework for vision.

## Overview

JEPA learns visual representations by predicting the representations of masked image patches from visible (context) patches. Unlike contrastive methods, JEPA operates in representation space rather than pixel space, making it more efficient and producing better representations for downstream tasks.

### Key Features

- **Vision Transformer Encoder**: Configurable ViT architecture with pre-norm/post-norm options
- **Multi-Block Masking**: Random block masking with configurable scale and aspect ratio
- **EMA Teacher**: Exponential Moving Average for stable target representations
- **Variance Regularization**: Optional regularization to prevent representation collapse
- **Training Infrastructure**: Learning rate scheduling, EMA momentum scheduling, callbacks
- **Data Augmentation**: Random flip, brightness/contrast adjustment, grayscale

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-jepa = { version = "0.20.0", features = ["train"] }
burn = { version = "0.20.0", features = ["ndarray"] }  # or "cuda" for GPU
```

## Quick Start

### Training

```rust
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_jepa::model::jepa::{JepaBatch, JepaConfig};
use burn_jepa::model::ema::CosineAnnealingMomentum;

type Backend = NdArray<f32>;

fn main() {
    let device = Default::default();

    // Create model
    let config = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_embed_dim(768)
        .with_n_layers(12);
    let mut model = config.init::<Backend>(&device);

    // Create momentum scheduler
    let momentum_scheduler = CosineAnnealingMomentum::new(0.996, 1.0);

    // Training loop
    for step in 0..1000 {
        let images = load_batch(); // Your data loading
        let batch = JepaBatch { images };

        // Forward pass
        let output = model.forward_step(batch);

        // Backward pass (with autodiff backend)
        // let grads = output.loss.backward();
        // optimizer.step(grads, &mut model);

        // EMA update
        let momentum = momentum_scheduler.get_momentum(step, 1000);
        model = model.ema_update(momentum);
    }
}
```

### Extracting Representations

```rust
use burn_jepa::model::jepa::JepaConfig;

let config = JepaConfig::new();
let model = config.init::<Backend>(&device);

// Extract patch embeddings
let patches = model.patch_embed.forward(images);

// Get encoder representations
let representations = model.teacher_encoder.forward(patches);
```

## Architecture

```
┌─────────────┐     ┌─────────────┐
│   Images    │     │   Images    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Patch Embed  │    │ Patch Embed  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Context Mask │    │   No Mask    │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│   Student    │    │   Teacher    │ ← EMA Update
│   Encoder    │    │   Encoder    │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   │
┌──────────────┐           │
│  Predictor   │           │
└──────┬───────┘           │
       │                   │
       ▼                   ▼
┌────────────────────────────────┐
│      L2-Normalized MSE Loss    │
└────────────────────────────────┘
```

## Configuration

### Model Configuration

```rust
let config = JepaConfig::new()
    .with_image_size(224)           // Input image size
    .with_patch_size(16)            // Patch size (224/16 = 14x14 patches)
    .with_in_channels(3)            // RGB images
    .with_embed_dim(768)            // Embedding dimension
    .with_n_layers(12)              // Transformer layers
    .with_n_heads(12)               // Attention heads
    .with_dropout(0.0)              // Dropout rate
    .with_predictor_n_layers(6)     // Predictor depth
    .with_num_target_blocks(4)      // Number of target blocks per image
    .with_var_reg_weight(5.0);      // Variance regularization
```

### Training Configuration

```rust
use burn_jepa::train::config::TrainingConfig;

let config = TrainingConfig::new()
    .with_num_epochs(100)
    .with_batch_size(64)
    .with_learning_rate(1.5e-4)
    .with_warmup_epochs(10)
    .with_weight_decay(0.05)
    .with_ema_momentum_base(0.996)
    .with_ema_momentum_end(1.0);
```

## Data Augmentation

```rust
use burn_jepa::data::augment::JepaAugmentationConfig;

let augmentation = JepaAugmentationConfig::jepa_defaults().init();
let augmented = augmentation.apply(images, &device);
```

Available augmentations:
- Random horizontal flip
- Brightness adjustment
- Contrast adjustment
- Random grayscale

## Examples

Run the training example:

```bash
cargo run --example train_jepa -p burn-jepa
```

Run the evaluation example:

```bash
cargo run --example eval_representation -p burn-jepa
```

## Benchmarks

Run benchmarks:

```bash
cargo run --example bench_forward -p burn-jepa --release
```

## Model Variants

| Variant    | Embed Dim | Layers | Heads | Patches |
|------------|-----------|--------|-------|---------|
| ViT-Small  | 384       | 12     | 6     | 14x14   |
| ViT-Base   | 768       | 12     | 12    | 14x14   |
| ViT-Large  | 1024      | 24     | 16    | 14x14   |

## References

- [I-JEPA Paper](https://arxiv.org/abs/2301.08243): Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- [Burn Framework](https://burn.dev): Deep learning framework in Rust

## License

Licensed under the MIT license. See [LICENSE](LICENSE) for details.
