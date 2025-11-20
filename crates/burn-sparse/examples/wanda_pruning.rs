//! Example: One-shot pruning with Wanda
//!
//! This example demonstrates how to use Wanda (activation-weighted magnitude pruning)
//! to create a sparse neural network from a pretrained dense model.
//!
//! Run with: cargo run --example wanda_pruning --features cuda

use burn::tensor::{backend::Backend, Tensor};
use burn_cuda::CudaBackend;
use burn_sparse::core::CalibrationData;
use burn_sparse::methods::static_pruning::wanda::{Wanda, WandaConfig};
use burn_sparse::nn::{SparseLinear, SparseLinearConfig};

type B = CudaBackend;

fn main() {
    println!("=== Wanda One-Shot Pruning Example ===\n");

    // Network dimensions
    let input_size = 784;
    let hidden_size = 512;
    let n_calibration = 128;

    // Wanda configuration
    let wanda_config = WandaConfig {
        sparsity: 0.9, // 90% sparsity
        n_calibration,
        use_l2: true, // Use L2 norm for activation scoring
    };

    println!("Configuration:");
    println!("  Target sparsity: {:.1}%", wanda_config.sparsity * 100.0);
    println!("  Calibration samples: {}", wanda_config.n_calibration);
    println!("  Activation scoring: L2 norm\n");

    let device = Default::default();

    // Step 1: Create a pretrained dense layer (simulated)
    println!("Step 1: Loading pretrained dense model...");
    let dense_weights: Tensor<B, 2> = Tensor::random(
        [hidden_size, input_size],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    println!("  Dense model: {} × {} = {} parameters", hidden_size, input_size, hidden_size * input_size);

    // Step 2: Collect calibration data
    println!("\nStep 2: Collecting calibration data...");
    let mut calibration_samples = Vec::new();
    for i in 0..n_calibration {
        // Simulate real activations from a calibration dataset
        let sample: Tensor<B, 1> =
            Tensor::random([input_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        calibration_samples.push(sample);
    }
    let calibration_data = CalibrationData::from_samples(
        calibration_samples
            .iter()
            .map(|s| s.clone().unsqueeze_dim(0))
            .collect(),
    );
    println!("  Collected {} calibration samples", calibration_data.len());

    // Step 3: Run Wanda pruning
    println!("\nStep 3: Running Wanda pruning...");
    let mut wanda = Wanda::new(wanda_config);
    let sparse_mask = wanda.prune(&dense_weights, &calibration_data);

    let n_active = sparse_mask.n_active();
    let n_total = hidden_size * input_size;
    let actual_sparsity = 1.0 - (n_active as f32 / n_total as f32);

    println!("  Pruned to {} active weights ({:.1}% sparsity)", n_active, actual_sparsity * 100.0);
    println!("  Removed {} parameters", n_total - n_active);

    // Step 4: Create sparse model
    println!("\nStep 4: Creating sparse model...");

    // Create sparse linear layer
    let config = SparseLinearConfig::new(input_size, hidden_size);
    let sparse_layer: SparseLinear<B> = config.init_with_mask(&device, sparse_mask.clone());

    println!("  Sparse model created successfully");

    // Step 5: Test forward pass
    println!("\nStep 5: Testing sparse model...");
    let test_input: Tensor<B, 2> =
        Tensor::random([1, input_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let output = sparse_layer.forward(test_input);

    println!("  Output shape: {:?}", output.dims());
    println!("  ✓ Forward pass successful");

    println!("\nWanda Pruning Complete!");
    println!("\nHow Wanda Works:");
    println!("  1. Collect activation statistics from calibration data");
    println!("  2. Compute importance: S[i,j] = |W[i,j]| × σ[x[j]]");
    println!("     where σ[x[j]] is the L2 norm of activations for input j");
    println!("  3. Keep top-(1-sparsity) weights, prune the rest");
    println!("  4. No retraining needed - weights remain unchanged");
    println!("\nUse Cases:");
    println!("  • Compress large language models for inference");
    println!("  • Initialize structured pruning methods");
    println!("  • Baseline for dynamic sparse training (RigL, MEST)");
}
