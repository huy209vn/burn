//! Example: Training with MEST (Memory-Economic Sparse Training)
//!
//! This example demonstrates how to use MEST for dynamic sparse training.
//! MEST uses salience scores (S = |W| + λ|g|) for both pruning and growing,
//! with an elastic mutation schedule that decays over time.
//!
//! Run with: cargo run --example mest_training

use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use burn_sparse::methods::dynamic::mest::{Mest, MestConfig};
use burn_sparse::nn::SparseLinear;

type Backend = NdArray<f32>;
type AB = burn_ndarray::NdArrayAutodiff<f32>;

fn main() {
    println!("=== MEST Dynamic Sparse Training Example ===\n");

    // Training configuration
    let n_epochs = 100;
    let batch_size = 32;
    let learning_rate = 0.001;
    let total_steps = n_epochs;

    // Network dimensions
    let input_size = 784; // MNIST-like input
    let hidden_size = 512;
    let output_size = 10;

    // MEST configuration
    let mest_config = MestConfig {
        sparsity: 0.9, // 90% sparsity
        mutation_rate_init: 0.3, // Start with 30% mutation
        mutation_rate_final: 0.05, // End with 5% mutation
        lambda: 0.01, // Balance between magnitude and gradient
        use_gradient_ema: true, // Use EMA for gradient smoothing
        ema_decay: 0.9, // EMA decay factor
    };

    println!("Configuration:");
    println!("  Sparsity: {:.1}%", mest_config.sparsity * 100.0);
    println!(
        "  Mutation rate: {:.1}% → {:.1}% (elastic decay)",
        mest_config.mutation_rate_init * 100.0,
        mest_config.mutation_rate_final * 100.0
    );
    println!("  Lambda (gradient weight): {}", mest_config.lambda);
    println!("  Gradient EMA: {}\n", mest_config.use_gradient_ema);

    // Initialize sparse layer
    let device = Default::default();
    let layer: SparseLinear<AB> =
        SparseLinear::new_random(input_size, hidden_size, mest_config.sparsity, &device);

    // Get initial mask from the layer
    let initial_mask = layer.mask().clone();

    // Initialize MEST
    let mut mest = Mest::new(mest_config, initial_mask, total_steps);

    // Initialize optimizer
    let mut optim = AdamConfig::new().init();

    println!("Starting training...\n");

    for epoch in 0..n_epochs {
        // Simulate a training batch
        let input: Tensor<AB, 2> =
            Tensor::random([batch_size, input_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        // Forward pass
        let output = layer.forward(input.clone());

        // Simulate loss (for demonstration)
        let target: Tensor<AB, 2> = Tensor::random(
            [batch_size, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let loss = (output.clone() - target).powf_scalar(2.0).mean();

        // Backward pass
        let grads = loss.backward();

        // Get weight and gradient tensors
        let weight_grad = layer.weight.grad(&grads).expect("No gradient for weight");

        // Update sparse mask with MEST
        // MEST updates every step (no update_frequency like RigL)
        let new_mask = mest.update_mask(layer.weight.val(), &weight_grad);

        // Calculate current mutation rate
        let progress = epoch as f32 / total_steps as f32;
        let current_mutation = mest_config.mutation_rate_init
            + (mest_config.mutation_rate_final - mest_config.mutation_rate_init) * progress;

        if epoch % 10 == 0 {
            let loss_val: f32 = loss.into_scalar().elem();
            println!(
                "Epoch {}: Loss = {:.6}, Mutation Rate = {:.1}%, Active = {}",
                epoch,
                loss_val,
                current_mutation * 100.0,
                new_mask.n_active()
            );
        }

        // In a real implementation, you would update the layer's mask here
        // layer.set_mask(new_mask);

        // Optimizer step
        // layer = optim.step(learning_rate, layer, grads);
    }

    println!("\nTraining complete!");
    println!("\nKey Points:");
    println!("  • MEST uses salience S = |W| + λ|g| for both prune and grow");
    println!("  • Mutation rate decays elastically from init to final");
    println!("  • Gradient EMA provides smoother salience estimates");
    println!("  • λ parameter balances magnitude vs gradient importance");
}
