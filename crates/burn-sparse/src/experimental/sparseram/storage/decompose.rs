//! Block decomposition utilities for SparseTensor

use crate::core::SparseTensor;
use crate::experimental::sparseram::{
    error::{SparseRAMError, SparseRAMResult},
    storage::{Block, BlockData, BlockLocation},
};
use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Decompose a SparseTensor into individual blocks
///
/// This is needed for Paged and Streaming policies to manage blocks individually.
///
/// # Arguments
/// * `sparse` - SparseTensor to decompose
/// * `block_size` - Size of blocks (B in B×B)
///
/// # Returns
/// Vector of (Block, BlockData) pairs
pub fn decompose_sparse_tensor<B: Backend>(
    sparse: &SparseTensor<B>,
    block_size: usize,
) -> SparseRAMResult<Vec<BlockData>> {
    // For now, convert to dense first (inefficient but correct)
    // TODO: Optimize to work directly with sparse format
    let dense = sparse.to_dense();
    let [n_rows, n_cols] = sparse.shape();

    let n_block_rows = (n_rows + block_size - 1) / block_size;
    let n_block_cols = (n_cols + block_size - 1) / block_size;

    let mut blocks = Vec::new();

    for block_row in 0..n_block_rows {
        for block_col in 0..n_block_cols {
            let block_data = extract_block_from_dense(
                &dense,
                block_row,
                block_col,
                block_size,
            )?;
            blocks.push(block_data);
        }
    }

    Ok(blocks)
}

/// Extract single block from dense tensor
fn extract_block_from_dense<B: Backend>(
    dense: &Tensor<B, 2>,
    block_row: usize,
    block_col: usize,
    block_size: usize,
) -> SparseRAMResult<BlockData> {
    let [n_rows, n_cols] = dense.dims();

    let row_start = block_row * block_size;
    let row_end = (row_start + block_size).min(n_rows);
    let col_start = block_col * block_size;
    let col_end = (col_start + block_size).min(n_cols);

    // Extract block region
    let mut block_tensor = dense
        .clone()
        .slice([row_start..row_end, col_start..col_end]);

    // Pad to full block size if needed (boundary blocks)
    let actual_rows = row_end - row_start;
    let actual_cols = col_end - col_start;

    if actual_rows < block_size || actual_cols < block_size {
        // Create padded tensor with zeros
        let mut padded_data = vec![0.0f32; block_size * block_size];

        // Copy actual data into top-left corner
        let block_data: Vec<f32> = block_tensor.clone().into_data().to_vec().unwrap();

        for r in 0..actual_rows {
            for c in 0..actual_cols {
                padded_data[r * block_size + c] = block_data[r * actual_cols + c];
            }
        }

        let tensor_data = TensorData::new(padded_data, Shape::new([block_size, block_size]));
        block_tensor = Tensor::from_data(tensor_data, &dense.device());
    }

    // Create Block metadata
    let block = Block {
        row: block_row,
        col: block_col,
        size: block_size,
        is_active: true, // Assume all decomposed blocks are active
        location: BlockLocation::None, // Will be set by residency engine
    };

    // Convert to backend-agnostic TensorData
    let data = block_tensor.into_data();

    Ok(BlockData::new(block, data))
}

/// Reconstruct SparseTensor from blocks
///
/// Inverse of decompose - combines blocks back into a single tensor.
///
/// # Arguments
/// * `blocks` - Vector of block data
/// * `shape` - Final tensor shape [n_rows, n_cols]
/// * `device` - Target device
///
/// # Returns
/// Reconstructed dense tensor (can be converted to sparse)
pub fn reconstruct_from_blocks<B: Backend>(
    blocks: &[BlockData],
    shape: [usize; 2],
    block_size: usize,
    device: &B::Device,
) -> SparseRAMResult<Tensor<B, 2>> {
    let [n_rows, n_cols] = shape;

    // Create output tensor filled with zeros
    let mut output_data = vec![0.0f32; n_rows * n_cols];

    // Fill in blocks
    for block_data in blocks {
        let block = &block_data.block;
        let row_start = block.row * block_size;
        let col_start = block.col * block_size;

        let row_end = (row_start + block_size).min(n_rows);
        let col_end = (col_start + block_size).min(n_cols);

        // Get block values
        let block_values: Vec<f32> = block_data.data.to_vec().unwrap();

        // Copy into output (respecting boundaries)
        for r in row_start..row_end {
            for c in col_start..col_end {
                let block_r = r - row_start;
                let block_c = c - col_start;
                output_data[r * n_cols + c] = block_values[block_r * block_size + block_c];
            }
        }
    }

    let tensor_data = TensorData::new(output_data, Shape::new([n_rows, n_cols]));
    Ok(Tensor::from_data(tensor_data, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseMask;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_decompose_and_reconstruct() {
        let device = Default::default();

        // Create 8x8 tensor
        let dense = Tensor::<TestBackend, 2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
                [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
            ],
            &device,
        );

        // Create sparse tensor
        let mask_data = dense.clone().greater_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);
        let sparse = SparseTensor::from_mask(&mask, &dense).unwrap();

        // Decompose into 4x4 blocks
        let blocks = decompose_sparse_tensor(&sparse, 4).unwrap();

        // Should have 2x2 = 4 blocks
        assert_eq!(blocks.len(), 4);

        // Reconstruct
        let reconstructed = reconstruct_from_blocks::<TestBackend>(&blocks, [8, 8], 4, &device).unwrap();

        // Should match original
        let original_data: Vec<f32> = dense.into_data().to_vec().unwrap();
        let reconstructed_data: Vec<f32> = reconstructed.into_data().to_vec().unwrap();

        for (o, r) in original_data.iter().zip(reconstructed_data.iter()) {
            assert!((o - r).abs() < 1e-5);
        }
    }

    #[test]
    fn test_non_divisible_size() {
        let device = Default::default();

        // Create 7x7 tensor (not divisible by block_size=4)
        let dense = Tensor::<TestBackend, 2>::ones([7, 7], &device);

        let mask_data = dense.clone().greater_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);
        let sparse = SparseTensor::from_mask(&mask, &dense).unwrap();

        // Decompose into 4x4 blocks
        let blocks = decompose_sparse_tensor(&sparse, 4).unwrap();

        // Should have ceil(7/4) x ceil(7/4) = 2x2 = 4 blocks
        assert_eq!(blocks.len(), 4);

        // Each block should be 4x4 (padded)
        for block_data in &blocks {
            assert_eq!(block_data.block.size, 4);
        }
    }
}
