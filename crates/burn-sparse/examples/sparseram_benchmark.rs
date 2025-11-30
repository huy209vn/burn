//! Comprehensive SparseRAM Benchmark
//!
//! This benchmark demonstrates and compares:
//! 1. Dense model VRAM usage vs SparseRAM
//! 2. Wanda pruning vs DSnoT-refined pruning
//! 3. Inference speed comparison
//! 4. Different sparsity levels (30%, 50%, 70%, 90%)
//!
//! Run with:
//! ```bash
//! # CPU (for testing)
//! cargo run --example sparseram_benchmark --features experimental
//!
//! # CUDA (for actual VRAM measurements)
//! cargo run --example sparseram_benchmark --features experimental,cuda
//! ```
//!
//! For realistic VRAM measurements, use CUDA backend.

use burn::tensor::{Distribution, Tensor};
use burn_core::tensor::backend::Backend;

use burn_sparse::{
    core::CalibrationData,
    methods::{
        iterative::dsnot::{DSnoT, DSnoTConfig},
        static_pruning::{Wanda, WandaConfig},
    },
};

#[cfg(feature = "experimental")]
use burn_sparse::experimental::sparseram::{PrunedStorageConfig, SparsePolicy, SparseRAM};

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

// Backend selection
#[cfg(feature = "cuda")]
type MyBackend = Cuda;

#[cfg(not(feature = "cuda"))]
type MyBackend = NdArray<f32>;

/// Statistics for a single benchmark run
#[derive(Debug, Clone)]
struct BenchmarkResult {
    method: String,
    sparsity_target: f32,
    sparsity_actual: f32,
    vram_mb: f32,
    ram_mb: f32,
    inference_time_ms: f32,
    throughput_tokens_per_sec: f32,
}

impl BenchmarkResult {
    fn print_row(&self) {
        println!(
            "│ {:12} │ {:7.1}% │ {:7.1}% │ {:9.2} │ {:8.2} │ {:11.2} │ {:14.1} │",
            self.method,
            self.sparsity_target * 100.0,
            self.sparsity_actual * 100.0,
            self.vram_mb,
            self.ram_mb,
            self.inference_time_ms,
            self.throughput_tokens_per_sec,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║            SparseRAM Comprehensive Benchmark Suite                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Setup device
    #[cfg(feature = "cuda")]
    let device = CudaDevice::default();

    #[cfg(not(feature = "cuda"))]
    let device = Default::default();

    #[cfg(feature = "cuda")]
    {
        println!("🎮 Backend: CUDA (GPU) - VRAM measurements are accurate");
        println!("   Device: {:?}", device);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  Backend: NdArray (CPU) - VRAM measurements are simulated");
        println!("   Run with --features cuda for accurate VRAM tracking");
    }

    println!();

    // Model configuration - 1B parameter model MLP layer
    // Based on typical 1B model architectures (e.g., GPT-2 1.5B, OPT-1.3B)
    // Standard ratio: intermediate_dim = 4 * hidden_dim
    let hidden_dim = 2048; // Output dimension (typical for 1B models)
    let intermediate_dim = 8192; // Input dimension (4x hidden_dim for MLP)

    println!("📐 Model Configuration:");
    println!("   Simulating: 1B parameter model MLP layer (down_proj)");
    println!("   Hidden dim: {}", hidden_dim);
    println!("   Intermediate dim: {}", intermediate_dim);
    println!(
        "   Total parameters: {} ({:.2} MB in fp32)\n",
        hidden_dim * intermediate_dim,
        (hidden_dim * intermediate_dim * 4) as f32 / (1024.0 * 1024.0)
    );

    // Inference configuration
    let batch_size = 4;
    let seq_len = 512; // Typical context length
    let n_inference_runs = 10; // Number of runs for averaging

    println!("🔬 Benchmark Configuration:");
    println!("   Batch size: {}", batch_size);
    println!("   Sequence length: {}", seq_len);
    println!("   Inference runs per test: {}", n_inference_runs);
    println!("   Calibration samples: 128\n");

    // Create dense weights
    println!("📦 Creating dense weight matrix...");
    let weights = create_dense_weights(hidden_dim, intermediate_dim, &device);
    let dense_vram = estimate_tensor_vram(&weights);
    println!("   Dense VRAM: {:.2} MB\n", dense_vram);

    // Create calibration data
    println!("🎲 Generating calibration data...");
    let calibration_data = create_calibration_data(intermediate_dim, 128, &device);
    println!("   Created 128 calibration samples\n");

    // Benchmark configuration
    let sparsity_levels = vec![0.3, 0.5, 0.7, 0.9];
    let mut results = Vec::new();

    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "║                            BENCHMARK RESULTS                                                    ║"
    );
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "┌──────────────┬─────────┬─────────┬───────────┬──────────┬─────────────┬────────────────┐"
    );
    println!(
        "│    Method    │ Target  │ Actual  │   VRAM    │   RAM    │  Inference  │   Throughput   │"
    );
    println!(
        "│              │ Sparsity│ Sparsity│    (MB)   │   (MB)   │    (ms)     │  (tokens/sec)  │"
    );
    println!(
        "├──────────────┼─────────┼─────────┼───────────┼──────────┼─────────────┼────────────────┤"
    );

    // Baseline: Dense model
    let dense_result = benchmark_dense(&weights, &device, batch_size, seq_len, n_inference_runs);
    dense_result.print_row();
    results.push(dense_result);

    // Benchmark each sparsity level with both Wanda and DSnoT
    for &sparsity in &sparsity_levels {
        // Wanda-only pruning
        let wanda_result = benchmark_wanda(
            &weights,
            &calibration_data,
            sparsity,
            &device,
            batch_size,
            seq_len,
            n_inference_runs,
        )?;
        wanda_result.print_row();
        results.push(wanda_result);

        // DSnoT refinement (starting from Wanda)
        let dsnot_result = benchmark_dsnot(
            &weights,
            &calibration_data,
            sparsity,
            &device,
            batch_size,
            seq_len,
            n_inference_runs,
        )?;
        dsnot_result.print_row();
        results.push(dsnot_result);
    }

    println!(
        "└──────────────┴─────────┴─────────┴───────────┴──────────┴─────────────┴────────────────┘"
    );

    // Summary statistics
    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY ANALYSIS                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Find best VRAM savings
    let dense_vram = results[0].vram_mb;
    let max_vram_saving = results
        .iter()
        .skip(1)
        .map(|r| (1.0 - r.vram_mb / dense_vram) * 100.0)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    println!("📊 Key Findings:");
    println!("   Maximum VRAM reduction: {:.1}%", max_vram_saving);

    // Compare Wanda vs DSnoT at each sparsity level
    println!("\n   Wanda vs DSnoT Comparison:");
    for &sparsity in &sparsity_levels {
        let wanda = results
            .iter()
            .find(|r| r.method == "Wanda" && r.sparsity_target == sparsity);
        let dsnot = results
            .iter()
            .find(|r| r.method == "DSnoT" && r.sparsity_target == sparsity);

        if let (Some(w), Some(d)) = (wanda, dsnot) {
            let vram_diff = ((d.vram_mb - w.vram_mb) / w.vram_mb) * 100.0;
            let perf_diff =
                ((d.inference_time_ms - w.inference_time_ms) / w.inference_time_ms) * 100.0;

            println!(
                "   Sparsity {:.0}%: DSnoT VRAM {:+.1}%, Inference time {:+.1}%",
                sparsity * 100.0,
                vram_diff,
                perf_diff
            );
        }
    }

    // Extrapolate to full model
    println!("\n💡 Extrapolation to Full 1B Model:");
    println!("   (Assuming 40 layers, typical architecture)");
    let layers = 40;
    for &sparsity in &sparsity_levels {
        if let Some(r) = results
            .iter()
            .find(|r| r.method == "DSnoT" && r.sparsity_target == sparsity)
        {
            let full_model_vram = r.vram_mb * layers as f32;
            let dense_full = dense_vram * layers as f32;
            let savings = (1.0 - full_model_vram / dense_full) * 100.0;

            println!(
                "   {:.0}% sparse: {:.2} GB VRAM ({:.1}% reduction)",
                sparsity * 100.0,
                full_model_vram / 1024.0,
                savings
            );
        }
    }

    #[cfg(feature = "cuda")]
    println!("\n✅ Benchmark complete! VRAM measurements are accurate (CUDA backend).");

    #[cfg(not(feature = "cuda"))]
    println!("\n✅ Benchmark complete! Run with --features cuda for accurate VRAM measurements.");

    Ok(())
}

/// Benchmark dense model (baseline)
fn benchmark_dense(
    weights: &Tensor<MyBackend, 2>,
    device: &<MyBackend as Backend>::Device,
    batch_size: usize,
    seq_len: usize,
    n_runs: usize,
) -> BenchmarkResult {
    let vram = estimate_tensor_vram(weights);

    // Measure inference time
    let inference_time = measure_inference_time(weights, device, batch_size, seq_len, n_runs);

    let throughput = (batch_size * seq_len) as f32 / (inference_time / 1000.0);

    BenchmarkResult {
        method: "Dense".to_string(),
        sparsity_target: 0.0,
        sparsity_actual: 0.0,
        vram_mb: vram,
        ram_mb: 0.0,
        inference_time_ms: inference_time,
        throughput_tokens_per_sec: throughput,
    }
}

/// Benchmark Wanda pruning + SparseRAM
fn benchmark_wanda(
    weights: &Tensor<MyBackend, 2>,
    calibration: &CalibrationData<MyBackend>,
    sparsity: f32,
    device: &<MyBackend as Backend>::Device,
    batch_size: usize,
    seq_len: usize,
    n_runs: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    // Apply Wanda pruning
    let wanda_config = WandaConfig {
        sparsity,
        n_calibration: 128,
        use_l2: true,
    };

    let mut wanda = Wanda::new(wanda_config);
    let mask = wanda.prune(weights, calibration);
    let actual_sparsity = mask.actual_sparsity();

    #[cfg(feature = "experimental")]
    {
        // Convert to SparseRAM
        let mut sparse_weight = SparseRAM::enable()
            .pruned_storage(PrunedStorageConfig::None)
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), mask)?;

        let vram = sparse_weight.vram_mb();
        let ram = sparse_weight.ram_mb();

        // Measure SPARSE inference time (using sparse matmul, not dense fallback)
        let inference_time =
            measure_sparse_inference_time(&mut sparse_weight, device, batch_size, seq_len, n_runs);

        let throughput = (batch_size * seq_len) as f32 / (inference_time / 1000.0);

        Ok(BenchmarkResult {
            method: "Wanda".to_string(),
            sparsity_target: sparsity,
            sparsity_actual: actual_sparsity,
            vram_mb: vram,
            ram_mb: ram,
            inference_time_ms: inference_time,
            throughput_tokens_per_sec: throughput,
        })
    }

    #[cfg(not(feature = "experimental"))]
    {
        Err("experimental feature required for SparseRAM".into())
    }
}

/// Benchmark DSnoT refinement + SparseRAM
fn benchmark_dsnot(
    weights: &Tensor<MyBackend, 2>,
    calibration: &CalibrationData<MyBackend>,
    sparsity: f32,
    device: &<MyBackend as Backend>::Device,
    batch_size: usize,
    seq_len: usize,
    n_runs: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    // Start with Wanda
    let wanda_config = WandaConfig {
        sparsity,
        n_calibration: 128,
        use_l2: true,
    };

    let mut wanda = Wanda::new(wanda_config);
    let initial_mask = wanda.prune(weights, calibration);

    // Refine with DSnoT
    let dsnot_config = DSnoTConfig {
        max_iters: 50,
        tolerance: 1e-6,
        n_calibration: 128,
        swap_fraction: 0.01,
        lambda: 1e-8,
    };

    let mut dsnot = DSnoT::new(dsnot_config);
    let refined_mask = dsnot.refine(weights, &initial_mask, calibration);
    let actual_sparsity = refined_mask.actual_sparsity();

    #[cfg(feature = "experimental")]
    {
        // Convert to SparseRAM
        let mut sparse_weight = SparseRAM::enable()
            .pruned_storage(PrunedStorageConfig::None)
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), refined_mask)?;

        let vram = sparse_weight.vram_mb();
        let ram = sparse_weight.ram_mb();

        // Measure SPARSE inference time (using sparse matmul, not dense fallback)
        let inference_time =
            measure_sparse_inference_time(&mut sparse_weight, device, batch_size, seq_len, n_runs);

        let throughput = (batch_size * seq_len) as f32 / (inference_time / 1000.0);

        Ok(BenchmarkResult {
            method: "DSnoT".to_string(),
            sparsity_target: sparsity,
            sparsity_actual: actual_sparsity,
            vram_mb: vram,
            ram_mb: ram,
            inference_time_ms: inference_time,
            throughput_tokens_per_sec: throughput,
        })
    }

    #[cfg(not(feature = "experimental"))]
    {
        Err("experimental feature required for SparseRAM".into())
    }
}

/// Create dense weight matrix
fn create_dense_weights(
    rows: usize,
    cols: usize,
    device: &<MyBackend as Backend>::Device,
) -> Tensor<MyBackend, 2> {
    Tensor::random([rows, cols], Distribution::Normal(0.0, 0.02), device)
}

/// Create calibration data
fn create_calibration_data(
    n_features: usize,
    n_samples: usize,
    device: &<MyBackend as Backend>::Device,
) -> CalibrationData<MyBackend> {
    let samples: Vec<Tensor<MyBackend, 2>> = (0..n_samples)
        .map(|_| Tensor::random([1, n_features], Distribution::Normal(0.0, 1.0), device))
        .collect();

    CalibrationData::from_samples(samples)
}

/// Estimate VRAM usage for a tensor
fn estimate_tensor_vram(tensor: &Tensor<MyBackend, 2>) -> f32 {
    let dims = tensor.dims();
    let elements = dims[0] * dims[1];
    // Assuming f32 (4 bytes per element)
    (elements * 4) as f32 / (1024.0 * 1024.0)
}

/// Measure inference time (matrix multiplication)
fn measure_inference_time(
    weights: &Tensor<MyBackend, 2>,
    device: &<MyBackend as Backend>::Device,
    batch_size: usize,
    seq_len: usize,
    n_runs: usize,
) -> f32 {
    let dims = weights.dims();
    let intermediate_dim = dims[1];

    // Create input tensor [batch_size * seq_len, intermediate_dim]
    let input = Tensor::<MyBackend, 2>::random(
        [batch_size * seq_len, intermediate_dim],
        Distribution::Uniform(0.0, 1.0),
        device,
    );

    // Warmup
    for _ in 0..2 {
        let _ = input.clone().matmul(weights.clone().transpose());
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _output = input.clone().matmul(weights.clone().transpose());
    }
    let elapsed = start.elapsed();

    (elapsed.as_secs_f32() * 1000.0) / n_runs as f32
}

#[cfg(feature = "experimental")]
/// Measure sparse inference time using SparseRAM (real sparse matmul)
fn measure_sparse_inference_time(
    sparse_weight: &mut burn_sparse::experimental::sparseram::SparseRAMWeight<MyBackend>,
    device: &<MyBackend as Backend>::Device,
    batch_size: usize,
    seq_len: usize,
    n_runs: usize,
) -> f32 {
    let shape = sparse_weight.shape();
    let intermediate_dim = shape[1];

    // Create input tensor [intermediate_dim, batch_size * seq_len]
    // Note: SparseRAM forward expects [n_cols, batch]
    let input = Tensor::<MyBackend, 2>::random(
        [intermediate_dim, batch_size * seq_len],
        Distribution::Uniform(0.0, 1.0),
        device,
    );

    // Warmup
    for _ in 0..2 {
        let _ = sparse_weight.forward(input.clone()).unwrap();
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _output = sparse_weight.forward(input.clone()).unwrap();
    }
    let elapsed = start.elapsed();

    (elapsed.as_secs_f32() * 1000.0) / n_runs as f32
}
