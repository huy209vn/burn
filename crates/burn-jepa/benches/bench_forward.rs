//! Benchmark for the forward pass of the JEPA model.
//!
//! Run with CPU:
//!   cargo run --example bench_forward -p burn-jepa --release
//!
//! Run with CUDA:
//!   cargo run --example bench_forward -p burn-jepa --release --features cuda

use std::time::Instant;

use burn::tensor::Tensor;
use burn_jepa::model::jepa::{JepaBatch, JepaConfig};

#[cfg(feature = "cuda")]
mod backend {
    pub use burn::backend::cuda::{Cuda, CudaDevice};
    pub type Backend = Cuda;
    pub fn device() -> CudaDevice {
        CudaDevice::default()
    }
    pub const NAME: &str = "CUDA (GPU)";
}

#[cfg(not(feature = "cuda"))]
mod backend {
    pub use burn::backend::ndarray::NdArrayDevice;
    pub use burn::backend::NdArray;
    pub type Backend = NdArray<f32>;
    pub fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
    pub const NAME: &str = "NdArray (CPU)";
}

use backend::{device, Backend, NAME};

fn main() {
    let device = device();

    // Test different configurations
    let configs = vec![
        ("ViT-Small (224)", 224, 16, 384, 12, 6),
        ("ViT-Base (224)", 224, 16, 768, 12, 12),
        ("ViT-Small (96)", 96, 8, 384, 12, 6),
    ];

    println!("JEPA Forward Pass Benchmark");
    println!("============================\n");
    println!("Backend: {}", NAME);
    println!();

    for (name, image_size, patch_size, embed_dim, n_layers, n_heads) in configs {
        let config = JepaConfig::new()
            .with_image_size(image_size)
            .with_patch_size(patch_size)
            .with_embed_dim(embed_dim)
            .with_n_layers(n_layers)
            .with_n_heads(n_heads);

        let model = config.init::<Backend>(&device);

        // Warmup
        let batch_size = 4;
        for _ in 0..3 {
            let images = Tensor::random(
                [batch_size, 3, image_size, image_size],
                burn::tensor::Distribution::Default,
                &device,
            );
            let batch = JepaBatch { images };
            let _ = model.forward_step(batch);
        }

        // Benchmark
        let num_iterations = 10;
        let start = Instant::now();
        for _ in 0..num_iterations {
            let images = Tensor::random(
                [batch_size, 3, image_size, image_size],
                burn::tensor::Distribution::Default,
                &device,
            );
            let batch = JepaBatch { images };
            let _ = model.forward_step(batch);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / num_iterations as f64;
        let throughput = (batch_size * num_iterations) as f64 / elapsed.as_secs_f64();

        println!("{}", name);
        println!(
            "  Config: {}x{} patches, embed_dim={}, layers={}",
            image_size / patch_size,
            image_size / patch_size,
            embed_dim,
            n_layers
        );
        println!("  Avg time: {:.1} ms/batch", avg_ms);
        println!("  Throughput: {:.1} images/sec", throughput);
        println!();
    }
}
