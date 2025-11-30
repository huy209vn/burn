//! Simple MEST Training Example
//!
//! Demonstrates dynamic sparse training with MEST on a toy problem.
//! Run with: cargo run --example simple_mest --features test-cuda

use burn::tensor::{Distribution, Tensor};
use burn_sparse::core::SparseMask;
use burn_sparse::methods::dynamic::mest::{Mest, MestConfig};

#[cfg(feature = "test-cuda")]
type Backend = burn_cuda::Cuda;

#[cfg(not(feature = "test-cuda"))]
type Backend = burn_ndarray::NdArray;

fn main() {
    println!("=== Simple MEST Dynamic Sparse Training Example ===\n");

    // Network dimensions
    let input_dim = 100;
    let hidden_dim = 50;
    let n_steps = 100;

    let device = Default::default();

    // Create initial random mask (uniform sparsity)
    let sparsity = 0.8; // 80% sparsity
    let total_weights = hidden_dim * input_dim;
    let n_active = ((1.0 - sparsity) * total_weights as f32) as usize;

    let mut mask_data = vec![false; total_weights];
    for i in 0..n_active {
        mask_data[i] = true;
    }

    let mask_tensor: Tensor<Backend, 2, burn::tensor::Bool> = Tensor::from_bool(
        burn::tensor::TensorData::new(mask_data, vec![hidden_dim, input_dim]),
        &device,
    );
    let initial_mask = SparseMask::from_tensor(mask_tensor);

    println!("Network: {} x {}", hidden_dim, input_dim);
    println!("Sparsity: {}%", sparsity * 100.0);
    println!("Active weights: {} / {}\n", n_active, total_weights);

    // Configure MEST
    let mest_config = MestConfig {
        sparsity,
        mutation_rate_init: 0.3,   // Start with 30% mutation
        mutation_rate_final: 0.05, // Decay to 5%
        lambda: 0.01,              // Gradient weight in salience
        update_frequency: 1,       // Update every step
        use_gradient_ema: true,    // Use EMA for gradients
        gradient_ema_beta: 0.9,    // EMA decay factor
    };

    println!("Configuration:");
    println!(
        "  Mutation rate: {:.0}% → {:.0}% (elastic decay)",
        mest_config.mutation_rate_init * 100.0,
        mest_config.mutation_rate_final * 100.0
    );
    println!("  Lambda: {}", mest_config.lambda);
    println!("  Gradient EMA: {}\n", mest_config.gradient_ema_beta);

    // Initialize MEST
    let mut mest = Mest::new(mest_config.clone(), initial_mask, n_steps);

    println!("Starting training simulation...\n");

    // Simulate training loop
    for step in 0..n_steps {
        // Simulate forward/backward (just random tensors for demo)
        let weights: Tensor<Backend, 2> = Tensor::random(
            [hidden_dim, input_dim],
            Distribution::Normal(0.0, 0.1),
            &device,
        );

        let gradients: Tensor<Backend, 2> = Tensor::random(
            [hidden_dim, input_dim],
            Distribution::Normal(0.0, 0.01),
            &device,
        );

        // Update mask with MEST (happens every step)
        let new_mask = mest.update_mask(&weights, &gradients);

        // Print progress every 20 steps
        if step % 20 == 0 {
            let mutation_rate = mest.current_mutation_rate();
            println!(
                "Step {}: Active = {}, Mutation rate = {:.1}%",
                step,
                new_mask.n_active(),
                mutation_rate * 100.0
            );
        }
    }

    println!("\n=== Final Results ===");
    println!("Training steps: {}", n_steps);
    println!(
        "Final mutation rate: {:.1}%",
        mest.current_mutation_rate() * 100.0
    );
    println!("\n✓ MEST training simulation complete!");
}
