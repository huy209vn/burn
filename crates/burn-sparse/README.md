# burn-sparse

Sparse training and pruning toolkit for the Burn deep learning framework.

## Features

### Stable Methods
- **Wanda**: Activation-weighted magnitude pruning for efficient one-shot sparsification
- **DSnoT**: Variance-weighted iterative mask refinement for improved sparse accuracy

### Architecture
- **Primitives**: Core infrastructure (masks, statistics, calibration data)
- **Methods**: Production-ready pruning algorithms
- **Backend-agnostic**: Works with any Burn backend

## Quick Start

```rust
use burn::prelude::*;
use burn_sparse::prelude::*;

fn prune_model<B: Backend>(
    weights: Tensor<B, 2>,
    calibration: CalibrationData<B>,
) -> SparseMask<B> {
    // Wanda one-shot pruning
    let config = WandaConfig {
        sparsity: 0.5,
        n_calibration: 128,
        ..Default::default()
    };

    let mut wanda = Wanda::new(config);
    let mask = wanda.prune(&weights, &calibration);

    // Optional: Refine with DSnoT
    let dsnot_config = DSnoTConfig::default();
    let mut dsnot = DSnoT::new(dsnot_config);
    let refined_mask = dsnot.refine(&weights, &mask, &calibration);

    refined_mask
}
```

## References

- **Wanda**: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024
- **DSnoT**: [Citation TBD]

## License

MIT OR Apache-2.0
