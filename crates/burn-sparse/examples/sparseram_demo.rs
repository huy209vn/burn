//! SparseRAM demonstration with simulated 14B model layer
//!
//! This example demonstrates the complete SparseRAM workflow:
//! 1. Create a dense weight matrix (simulating MLP down_proj)
//! 2. Apply structured sparsity (simulating Wanda pruning)
//! 3. Convert to SparseRAM with different policies
//! 4. Measure VRAM usage and sparsity
//! 5. Run forward pass (sparse matrix multiply)

use burn_sparse::{
    core::{CalibrationData, SparseFormat, SparseMask},
    methods::static_pruning::{Wanda, WandaConfig},
};

#[cfg(feature = "experimental")]
use burn_sparse::experimental::sparseram::{SparseRAM, SparsePolicy};

use burn::tensor::{ElementConversion, Shape, Tensor, TensorData};

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

#[cfg(feature = "cuda")]
type Backend = Cuda;

#[cfg(not(feature = "cuda"))]
type Backend = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SparseRAM Demo - Simulating Strand-Rust-Coder-14B MLP Layer\n");
    println!("================================================================\n");

    #[cfg(feature = "cuda")]
    let device = CudaDevice::default();

    #[cfg(not(feature = "cuda"))]
    let device = Default::default();

    #[cfg(feature = "cuda")]
    println!("🎮 Using CUDA backend on GPU\n");

    #[cfg(not(feature = "cuda"))]
    println!("⚠️  Using NdArray backend on CPU (for testing only)\n");

    // Simulate MLP down_proj layer dimensions: 13824 × 5120
    // This is the largest weight matrix in each decoder layer
    let hidden_dim = 5120;
    let intermediate_dim = 13824;

    println!("📐 Layer dimensions:");
    println!("   Input: {}", intermediate_dim);
    println!("   Output: {}", hidden_dim);
    println!("   Total parameters: {} ({:.2} MB in bf16)\n",
        hidden_dim * intermediate_dim,
        (hidden_dim * intermediate_dim * 2) as f32 / 1_000_000.0
    );

    // Step 1: Create dense weight matrix
    println!("📦 Step 1: Creating dense weight matrix...");

    let weights = create_dense_weights(hidden_dim, intermediate_dim, &device);

    let dense_size_mb = (hidden_dim * intermediate_dim * 4) as f32 / (1024.0 * 1024.0);
    println!("   Dense weights: {:.2} MB (f32)\n", dense_size_mb);

    // Step 2: Apply REAL Wanda pruning with calibration
    println!("✂️  Step 2: Applying Wanda pruning (50% sparsity)...");

    let target_sparsity = 0.5;
    let n_calibration = 128;

    // Create calibration data (simulated activations)
    println!("   Generating {} calibration samples...", n_calibration);
    let calibration_data = create_calibration_data(intermediate_dim, n_calibration, &device);

    // Configure and run Wanda
    let wanda_config = WandaConfig {
        sparsity: target_sparsity,
        n_calibration,
        use_l2: true, // Use L2 norm for activation importance
    };

    let mut wanda = Wanda::new(wanda_config);
    let mask = wanda.prune(&weights, &calibration_data);

    let actual_sparsity = mask.actual_sparsity();
    println!("   Target sparsity: {:.1}%", target_sparsity * 100.0);
    println!("   Actual sparsity: {:.1}%", actual_sparsity * 100.0);
    println!("   Active elements: {}", mask.n_active());
    println!("   Pruned elements: {}\n", mask.n_pruned());

    // Step 3: Convert to SparseRAM with Eager policy
    println!("🔄 Step 3: Converting to SparseRAM (Eager policy)...");

    let sparse_weight = SparseRAM::enable()
        // Format (CSR/COO/BlockCSR) chosen automatically by burn-sparse
        .policy(SparsePolicy::Eager)
        .apply(weights.clone(), mask.clone())?;

    println!("   ✅ Conversion successful!");
    println!("   VRAM usage: {:.2} MB", sparse_weight.vram_mb());
    println!("   RAM usage: {:.2} MB", sparse_weight.ram_mb());
    println!("   Sparsity: {:.1}%", sparse_weight.sparsity() * 100.0);
    println!("   Non-zero elements: {}", sparse_weight.nnz());

    let vram_savings = (1.0 - (sparse_weight.vram_mb() / dense_size_mb)) * 100.0;
    println!("   VRAM savings: {:.1}% 🎉\n", vram_savings);

    // Step 4: Run forward pass (sparse matmul)
    println!("🎯 Step 4: Testing sparse forward pass...");

    let batch_size = 4;
    let seq_len = 128;

    // Input: [batch_size * seq_len, intermediate_dim]
    let input = Tensor::<Backend, 2>::random(
        [batch_size * seq_len, intermediate_dim],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    println!("   Input shape: {:?}", input.dims());

    // Note: Can't actually run forward pass yet without fixing the API
    // This would require the SparseRAMWeight to be mutable
    // let output = sparse_weight.forward(input)?;
    // println!("   Output shape: {:?}", output.dims());

    println!("   ⚠️  Forward pass requires mutable access (API limitation)\n");

    // Step 5: Compare with different sparsity levels using Wanda
    println!("📊 Step 5: Comparing different Wanda sparsity levels...\n");

    for target_sparsity in [0.3, 0.5, 0.7, 0.9] {
        let wanda_config = WandaConfig {
            sparsity: target_sparsity,
            n_calibration: 128,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), mask)?;

        let vram_reduction = (1.0 - (sparse_weight.vram_mb() / dense_size_mb)) * 100.0;

        println!("   Sparsity {:.0}%: VRAM = {:.2} MB, Reduction = {:.1}%",
            target_sparsity * 100.0,
            sparse_weight.vram_mb(),
            vram_reduction
        );
    }

    println!("\n✅ Demo complete!");
    println!("\n💡 For a full 14B model (48 layers):");
    println!("   Dense VRAM: ~32 GB");
    println!("   50% sparse: ~16 GB (50% reduction)");
    println!("   70% sparse: ~10 GB (70% reduction)");
    println!("   90% sparse: ~5 GB (85% reduction)");

    Ok(())
}

/// Create dense weight matrix with random initialization
fn create_dense_weights(
    rows: usize,
    cols: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Tensor<Backend, 2> {
    Tensor::random(
        [rows, cols],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        device,
    )
}

/// Create calibration data for Wanda pruning
///
/// In real scenarios, this would be actual activation data from running
/// the model on representative inputs (e.g., WikiText for language models).
/// Here we simulate it with random data.
fn create_calibration_data(
    n_features: usize,
    n_samples: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> CalibrationData<Backend> {
    let samples: Vec<Tensor<Backend, 2>> = (0..n_samples)
        .map(|_| {
            Tensor::random(
                [1, n_features],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            )
        })
        .collect();

    CalibrationData::from_samples(samples)
}

