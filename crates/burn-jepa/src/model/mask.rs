//! Multi-block masking strategy for JEPA.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// # Mask Output
///
/// Contains the indices for context and target patches, along with their positions.
#[derive(Debug, Clone)]
pub struct MaskOutput<B: Backend> {
    /// Indices of context patches [B, N_ctx]
    pub context_indices: Tensor<B, 2, Int>,
    /// Indices of target patches [B, N_tgt]
    pub target_indices: Tensor<B, 2, Int>,
}

/// # Masking Configuration
#[derive(Debug, Clone)]
pub struct MaskingConfig {
    /// Number of target blocks to sample
    pub num_target_blocks: usize,
    /// Range for target block scale (fraction of total patches)
    pub target_scale_range: [f32; 2],
    /// Range for target block aspect ratio
    pub target_aspect_ratio_range: [f32; 2],
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self {
            num_target_blocks: 4,
            target_scale_range: [0.15, 0.2],
            target_aspect_ratio_range: [0.75, 1.5],
        }
    }
}

/// # Sample Block Masks
///
/// Generates multi-block masks for JEPA training.
/// Samples rectangular target blocks and treats remaining patches as context.
///
/// # Arguments
/// * `batch_size` - Number of samples in the batch
/// * `grid_h` - Height of the patch grid
/// * `grid_w` - Width of the patch grid
/// * `config` - Masking configuration
/// * `device` - Device to create tensors on
///
/// # Returns
/// * `MaskOutput` containing context and target indices
pub fn sample_block_masks<B: Backend>(
    batch_size: usize,
    grid_h: usize,
    grid_w: usize,
    config: &MaskingConfig,
    device: &B::Device,
) -> MaskOutput<B> {
    let num_patches = grid_h * grid_w;
    let mut rng = StdRng::from_entropy();

    let mut all_context_indices = Vec::new();
    let mut all_target_indices = Vec::new();

    for _ in 0..batch_size {
        // Create a boolean mask for this sample
        let mut target_mask = vec![false; num_patches];

        // Sample multiple target blocks
        for _ in 0..config.num_target_blocks {
            // Sample aspect ratio and scale
            let aspect_ratio = rng.gen_range(
                config.target_aspect_ratio_range[0]..=config.target_aspect_ratio_range[1],
            );
            let scale =
                rng.gen_range(config.target_scale_range[0]..=config.target_scale_range[1]);

            // Calculate block dimensions
            let block_area = (scale * num_patches as f32) as usize;
            let block_h = ((block_area as f32 / aspect_ratio).sqrt() as usize).max(1);
            let block_w = ((block_area as f32 * aspect_ratio).sqrt() as usize).max(1);

            // Ensure block fits in grid
            let block_h = block_h.min(grid_h);
            let block_w = block_w.min(grid_w);

            // Sample random top-left corner
            let top = if grid_h > block_h {
                rng.gen_range(0..=(grid_h - block_h))
            } else {
                0
            };
            let left = if grid_w > block_w {
                rng.gen_range(0..=(grid_w - block_w))
            } else {
                0
            };

            // Mark block positions as targets
            for i in top..(top + block_h) {
                for j in left..(left + block_w) {
                    let idx = i * grid_w + j;
                    if idx < num_patches {
                        target_mask[idx] = true;
                    }
                }
            }
        }

        // Collect context and target indices
        let context_idx: Vec<i64> = target_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_target)| if !is_target { Some(i as i64) } else { None })
            .collect();

        let target_idx: Vec<i64> = target_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_target)| if is_target { Some(i as i64) } else { None })
            .collect();

        all_context_indices.push(context_idx);
        all_target_indices.push(target_idx);
    }

    // Find max lengths for padding
    let max_context = all_context_indices.iter().map(|v| v.len()).max().unwrap();
    let max_target = all_target_indices.iter().map(|v| v.len()).max().unwrap();

    // Pad sequences to max length (padding with 0, will be masked)
    let mut context_data = Vec::new();
    let mut target_data = Vec::new();

    for ctx_indices in all_context_indices {
        let mut padded = ctx_indices.clone();
        padded.resize(max_context, 0);
        context_data.extend(padded);
    }

    for tgt_indices in all_target_indices {
        let mut padded = tgt_indices.clone();
        padded.resize(max_target, 0);
        target_data.extend(padded);
    }

    // Create tensors
    let context_indices = Tensor::from_data(
        burn::tensor::TensorData::new(context_data, [batch_size, max_context]),
        device,
    );

    let target_indices = Tensor::from_data(
        burn::tensor::TensorData::new(target_data, [batch_size, max_target]),
        device,
    );

    MaskOutput {
        context_indices,
        target_indices,
    }
}