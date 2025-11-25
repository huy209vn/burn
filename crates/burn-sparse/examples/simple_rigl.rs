//! Simple RigL Training Example
//!
//! Demonstrates dynamic sparse training with RigL on a toy problem.
//! Run with: cargo run --example simple_rigl --features test-cuda

use burn::tensor::{Distribution, Tensor};
use burn_sparse::core::SparseMask;
use burn_sparse::methods::dynamic::rigl::{RigL, RigLConfig};

#[cfg(feature = "test-cuda")]
type Backend = burn_cuda::Cuda;

#[cfg(not(feature = "test-cuda"))]
type Backend = burn_ndarray::NdArray;

fn main() {
    println!("=== Simple RigL Dynamic Sparse Training Example ===\n");

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

    // Configure RigL
    let rigl_config = RigLConfig {
        sparsity,
        update_frequency: 10, // Update mask every 10 steps
        drop_fraction: 0.3,   // Drop 30% of active weights per update
    };

    // Initialize RigL
    let mut rigl = RigL::new(rigl_config.clone(), initial_mask);

    println!("Starting training simulation...\n");

    // Simulate training loop (without actual gradients for simplicity)
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

        // Update mask with RigL
        if step > 0 && step % rigl_config.update_frequency == 0 {
            let new_mask = rigl.update_mask(&weights, &gradients);

            println!(
                "Step {}: Mask updated. Active weights: {}",
                step,
                new_mask.n_active()
            );
        }
    }

    println!("\n=== Final Results ===");
    println!("Training steps: {}", n_steps);
    println!("Mask updates: {}", n_steps / rigl_config.update_frequency);
    println!("\n✓ RigL training simulation complete!");
}
