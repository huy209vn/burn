# SparseRAM Benchmark - Quick Start

## Running the Full Benchmark

### With CUDA (Recommended for accurate VRAM measurements)

```bash
cd crates/burn-sparse
cargo run --example sparseram_benchmark --features experimental,cuda --release
```

### Without CUDA (CPU only, simulated VRAM)

```bash
cd crates/burn-sparse
cargo run --example sparseram_benchmark --features experimental --release
```

## What This Benchmark Does

1. **Creates a 1B model MLP layer** (2048 × 8192 weights = ~64 MB)
2. **Applies Wanda pruning** at 30%, 50%, 70%, 90% sparsity
3. **Refines with DSnoT** for improved quality
4. **Converts to SparseRAM** with Eager policy (all on GPU)
5. **Measures**:
   - VRAM usage (dense vs sparse)
   - RAM usage (pruned blocks)
   - Inference time (matrix multiplication)
   - Throughput (tokens/sec)

## Expected Runtime

- **CPU (NdArray)**: ~2-5 minutes
- **CUDA (GPU)**: ~1-3 minutes

DSnoT refinement takes the most time (50 iterations per sparsity level).

## Configuration

Edit `sparseram_benchmark.rs` to customize:

```rust
// Model size
let hidden_dim = 2048;          // Change for different model sizes
let intermediate_dim = 8192;    // Typically 4x hidden_dim

// Sparsity levels to test
let sparsity_levels = vec![0.3, 0.5, 0.7, 0.9];

// Inference configuration
let batch_size = 4;
let seq_len = 512;
let n_inference_runs = 10;
```

## Output

The benchmark produces a formatted table:

```
┌──────────────┬─────────┬─────────┬───────────┬──────────┬─────────────┬────────────────┐
│    Method    │ Target  │ Actual  │   VRAM    │   RAM    │  Inference  │   Throughput   │
│              │ Sparsity│ Sparsity│    (MB)   │   (MB)   │    (ms)     │  (tokens/sec)  │
├──────────────┼─────────┼─────────┼───────────┼──────────┼─────────────┼────────────────┤
│ Dense        │   0.0%  │   0.0%  │     64.00 │     0.00 │        X.XX │          XXX.X │
│ Wanda        │  30.0%  │  30.0%  │     44.80 │     0.00 │        X.XX │          XXX.X │
│ DSnoT        │  30.0%  │  30.0%  │     44.80 │     0.00 │        X.XX │          XXX.X │
...
```

Plus summary analysis and extrapolation to full model.

## Troubleshooting

### CUDA Out of Memory

Reduce model size:
```rust
let hidden_dim = 1024;
let intermediate_dim = 4096;
```

### Takes Too Long

Reduce DSnoT iterations:
```rust
let dsnot_config = DSnoTConfig {
    max_iters: 20,  // Default: 50
    ..Default::default()
};
```

Or test fewer sparsity levels:
```rust
let sparsity_levels = vec![0.5, 0.9];  // Just two levels
```

## See Also

- `BENCHMARK.md` - Comprehensive documentation
- `sparseram_demo.rs` - Simple demo of SparseRAM API
- `simple_wanda.rs` - Wanda pruning example
- `simple_rigl.rs` - Dynamic sparse training example
