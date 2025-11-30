//! Simple Wanda Pruning Example
//!
//! Demonstrates one-shot pruning with Wanda on a simple network.
//! Run with: cargo run --example simple_wanda --features test-cuda

use burn::tensor::{Distribution, Tensor};
use burn_sparse::core::CalibrationData;
use burn_sparse::methods::static_pruning::wanda::{Wanda, WandaConfig};

#[cfg(feature = "test-cuda")]
type Backend = burn_cuda::Cuda;

#[cfg(not(feature = "test-cuda"))]
type Backend = burn_ndarray::NdArray;

fn main() {
    println!("=== Simple Wanda Pruning Example ===\n");

    // Simple network dimensions
    let batch_size = 32;
    let input_dim = 128;
    let output_dim = 64;

    // Create random weights (simulating a pretrained layer)
    let device = Default::default();
    let weights: Tensor<Backend, 2> = Tensor::random(
        [output_dim, input_dim],
        Distribution::Normal(0.0, 0.1),
        &device,
    );

    println!("Original weights: {} x {}", output_dim, input_dim);
    println!("Total parameters: {}\n", output_dim * input_dim);

    // Collect calibration data (simulated activations)
    let n_calibration = 64;
    let mut calibration_samples = Vec::new();

    println!("Collecting {} calibration samples...", n_calibration);
    for _ in 0..n_calibration {
        let sample: Tensor<Backend, 1> =
            Tensor::random([input_dim], Distribution::Normal(0.0, 1.0), &device);
        calibration_samples.push(sample.unsqueeze_dim(0));
    }

    let calibration_data = CalibrationData::from_samples(calibration_samples);

    // Configure Wanda
    let wanda_config = WandaConfig {
        sparsity: 0.8, // 80% sparsity (keep 20% of weights)
        n_calibration,
        use_l2: true,
    };

    println!("Target sparsity: {}%\n", wanda_config.sparsity * 100.0);

    // Run Wanda pruning
    println!("Running Wanda pruning...");
    let mut wanda = Wanda::new(wanda_config);
    let sparse_mask = wanda.prune(&weights, &calibration_data);

    // Results
    let n_active = sparse_mask.n_active();
    let n_total = output_dim * input_dim;
    let actual_sparsity = 1.0 - (n_active as f32 / n_total as f32);

    println!("\n=== Results ===");
    println!("Active weights: {} / {}", n_active, n_total);
    println!("Actual sparsity: {:.1}%", actual_sparsity * 100.0);
    println!("Parameters saved: {}", n_total - n_active);
    println!("Memory reduction: {:.1}x", 1.0 / (1.0 - actual_sparsity));

    println!("\n✓ Pruning complete!");
}
