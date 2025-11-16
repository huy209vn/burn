//! Example: Wanda + DSnoT pruning cycle
//!
//! Demonstrates how to use burn-sparse to prune a neural network layer:
//! 1. Create synthetic weight matrix and calibration data
//! 2. Apply Wanda for initial pruning
//! 3. Refine with DSnoT for improved accuracy
//! 4. Compare reconstruction errors
//!
//! Run with:
//! ```bash
//! cargo run --example pruning_cycle
//! ```

use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_ndarray::NdArray;
use burn_sparse::prelude::*;

type B = NdArray<f32>;

fn main() {
    println!("=== burn-sparse: Pruning Cycle Example ===\n");

    // 1. Create synthetic weight matrix [128, 256]
    let device = Default::default();
    let weights = Tensor::<B, 2>::random(
        [128, 256],
        Distribution::Normal(0.0, 0.1),
        &device,
    );

    println!("Weight matrix shape: {:?}", weights.dims());

    // 2. Generate calibration data (simulating activations)
    let n_calibration = 64;
    let mut cal_samples = Vec::new();

    for _ in 0..n_calibration {
        let sample = Tensor::<B, 2>::random(
            [1, 256],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        cal_samples.push(sample);
    }

    let calibration = CalibrationData::from_samples(cal_samples);
    println!("Calibration samples: {}\n", calibration.len());

    // 3. Apply Wanda pruning
    println!("--- Step 1: Wanda Pruning ---");

    let wanda_config = WandaConfig {
        sparsity: 0.5,
        n_calibration: 64,
        use_l2: true,
    };

    let mut wanda = Wanda::new(wanda_config);
    let wanda_mask = wanda.prune(&weights, &calibration);

    println!("Wanda mask sparsity: {:.2}%", wanda_mask.actual_sparsity() * 100.0);
    println!("Active weights: {}", wanda_mask.n_active());
    println!("Pruned weights: {}\n", wanda_mask.n_pruned());

    // Compute Wanda reconstruction error
    let wanda_sparse = wanda_mask.apply(&weights);
    let wanda_error = compute_error(&weights, &wanda_sparse, &calibration);
    println!("Wanda reconstruction error: {:.6}\n", wanda_error);

    // 4. Refine with DSnoT
    println!("--- Step 2: DSnoT Refinement ---");

    let dsnot_config = DSnoTConfig {
        max_iters: 20,
        update_threshold: 0.01,
        alpha: 1.0,
        tolerance: 1e-5,
        lambda: 1e-8,
    };

    let mut dsnot = DSnoT::new(dsnot_config);
    let dsnot_mask = dsnot.refine(&weights, &wanda_mask, &calibration);

    println!("DSnoT iterations: {}", dsnot.error_history().len());
    println!("DSnoT mask sparsity: {:.2}%", dsnot_mask.actual_sparsity() * 100.0);

    // Compute DSnoT reconstruction error
    let dsnot_sparse = dsnot_mask.apply(&weights);
    let dsnot_error = compute_error(&weights, &dsnot_sparse, &calibration);
    println!("DSnoT reconstruction error: {:.6}\n", dsnot_error);

    // 5. Compare results
    println!("--- Comparison ---");
    println!("Wanda error:  {:.6}", wanda_error);
    println!("DSnoT error:  {:.6}", dsnot_error);

    let improvement = (wanda_error - dsnot_error) / wanda_error * 100.0;
    if improvement > 0.0 {
        println!("Improvement:  {:.2}%", improvement);
    } else {
        println!("Change:       {:.2}%", improvement);
    }

    // 6. Show error history
    println!("\n--- DSnoT Error History ---");
    for (i, &err) in dsnot.error_history().iter().enumerate() {
        println!("Iteration {}: {:.6}", i, err);
    }

    // 7. Compute Hamming distance
    let hamming = wanda_mask.hamming_distance(&dsnot_mask);
    let n_swapped = hamming / 2; // Each swap creates 2 differences
    println!("\n--- Mask Changes ---");
    println!("Weights swapped: {}", n_swapped);
    println!("Swap percentage: {:.2}%", n_swapped as f32 / wanda_mask.n_active() as f32 * 100.0);

    println!("\n=== Example Complete ===");
}

/// Compute reconstruction error between dense and sparse weights
fn compute_error(
    dense: &Tensor<B, 2>,
    sparse: &Tensor<B, 2>,
    calibration: &CalibrationData<B>,
) -> f32 {
    let mut total_error = 0.0;
    let mut n_samples = 0;

    for sample in calibration.iter() {
        let y_dense = dense.matmul(sample.clone().unsqueeze_dim(1)).squeeze(1);
        let y_sparse = sparse.matmul(sample.clone().unsqueeze_dim(1)).squeeze(1);

        let error = (y_dense - y_sparse).powf_scalar(2.0).sum();
        total_error += error.into_scalar();
        n_samples += 1;
    }

    total_error / n_samples as f32
}
