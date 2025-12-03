//! Unit tests for the masking functionality in `burn-jepa`.

// This test module needs to be declared in `lib.rs` under `#[cfg(test)]`
// to access crate-internal items.

use burn::tensor::{backend::Backend, Data, Shape, Tensor};
use burn_ndarray::NdArrayBackend;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

use crate::model::mask::{sample_block_masks, MaskOutput};
use crate::config::MaskingConfig;

type TestBackend = NdArrayBackend<f32>;

#[test]
fn test_mask_output_structure_and_coverage() {
    let mut rng = StdRng::seed_from_u64(42);
    let device = Default::default();

    // 1. Define Test Parameters
    let config = MaskingConfig {
        num_target_blocks: 2,
        target_scale_range: [0.15, 0.25],
        target_aspect_ratio_range: [0.75, 1.5],
    };
    let grid_size = (14, 14);
    let num_patches = grid_size.0 * grid_size.1;

    // 2. Call the function
    let mask_output = sample_block_masks::<TestBackend>(&config, grid_size, &mut rng, &device);

    // 3. Assert Shapes (for a single mask pattern, batch size is 1)
    let context_shape = mask_output.context_indices.dims();
    let target_shape = mask_output.target_indices.dims();
    let target_pos_shape = mask_output.target_positions.dims();

    assert_eq!(context_shape.len(), 2, "Context indices should be rank 2");
    assert_eq!(target_shape.len(), 2, "Target indices should be rank 2");
    assert_eq!(target_pos_shape.len(), 3, "Target positions should be rank 3");

    assert_eq!(context_shape[0], 1, "Batch size should be 1 for a single mask pattern");
    assert_eq!(target_shape[0], 1, "Batch size should be 1 for a single mask pattern");
    assert_eq!(target_pos_shape[0], 1, "Batch size should be 1 for a single mask pattern");

    assert_eq!(target_pos_shape[2], 2, "Target positions should have 2 coordinates (row, col)");

    // 4. Assert Coverage and Disjointedness
    let n_ctx = context_shape[1];
    let n_tgt = target_shape[1];

    // Total patches must be conserved
    assert_eq!(n_ctx + n_tgt, num_patches, "Total number of patches (context + target) must equal total grid size");

    let context_data: Vec<usize> = mask_output.context_indices.into_data().value.into_iter().map(|v| v as usize).collect();
    let target_data: Vec<usize> = mask_output.target_indices.into_data().value.into_iter().map(|v| v as usize).collect();

    let context_set: HashSet<usize> = context_data.into_iter().collect();
    let target_set: HashSet<usize> = target_data.into_iter().collect();

    assert_eq!(context_set.len() + target_set.len(), num_patches, "Union of context and target sets must have the same size as the total number of patches");
    assert!(context_set.is_disjoint(&target_set), "Context and target index sets must be disjoint");
}