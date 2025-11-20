//! Example: Training with RigL dynamic sparse training
//!
//! This example demonstrates how to use RigL (Rigging the Lottery) for
//! dynamic sparse training. RigL prunes weights by magnitude and grows
//! weights by gradient magnitude.
//!
//! Run with: cargo run --example rigl_training

use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use burn_sparse::methods::dynamic::rigl::{RigL, RigLConfig};
use burn_sparse::nn::SparseLinear;

type Backend = NdArray<f32>;
type AB = burn_ndarray::NdArrayAutodiff<f32>;

fn main() {
    println!("=== RigL Dynamic Sparse Training Example ===\n");

    // Training configuration
    let n_epochs = 100;
    let batch_size = 32;
    let learning_rate = 0.001;

    // Network dimensions
    let input_size = 784; // MNIST-like input
    let hidden_size = 512;
    let output_size = 10;

    // RigL configuration
    let rigl_config = RigLConfig {
        sparsity: 0.9, // 90% sparsity (10% of weights active)
        update_frequency: 100, // Update mask every 100 steps
        drop_fraction: 0.3, // Drop 30% of active weights per update
    };

    println!("Configuration:");
    println!("  Sparsity: {:.1}%", rigl_config.sparsity * 100.0);
    println!("  Update frequency: {} steps", rigl_config.update_frequency);
    println!("  Drop fraction: {:.1}%\n", rigl_config.drop_fraction * 100.0);

    // Initialize sparse layer
    let device = Default::default();
    let layer: SparseLinear<AB> =
        SparseLinear::new_random(input_size, hidden_size, rigl_config.sparsity, &device);

    // Get initial mask from the layer
    let initial_mask = layer.mask().clone();

    // Initialize RigL
    let mut rigl = RigL::new(rigl_config, initial_mask);

    // Initialize optimizer
    let mut optim = AdamConfig::new().init();

    println!("Starting training...\n");

    let mut step = 0;
    for epoch in 0..n_epochs {
        // Simulate a training batch
        let input: Tensor<AB, 2> =
            Tensor::random([batch_size, input_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        // Forward pass
        let output = layer.forward(input.clone());

        // Simulate loss (for demonstration, we'll use mean squared error with random targets)
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

        // Update sparse mask with RigL
        if step > 0 && step % rigl.update_frequency() == 0 {
            let new_mask = rigl.update_mask(layer.weight.val(), &weight_grad);

            println!(
                "Epoch {}: Updated mask at step {}. Active weights: {} ({:.1}%)",
                epoch,
                step,
                new_mask.n_active(),
                (new_mask.n_active() as f32 / (input_size * hidden_size) as f32) * 100.0
            );

            // In a real implementation, you would update the layer's mask here
            // layer.set_mask(new_mask);
        }

        // Optimizer step
        // layer = optim.step(learning_rate, layer, grads);

        step += 1;

        if epoch % 10 == 0 {
            let loss_val: f32 = loss.into_scalar().elem();
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }

    println!("\nTraining complete!");
    println!("\nKey Points:");
    println!("  • RigL prunes weights with smallest magnitude");
    println!("  • RigL grows weights with largest gradient magnitude");
    println!("  • Maintains constant sparsity throughout training");
    println!("  • Update frequency controls mask refresh rate");
}
