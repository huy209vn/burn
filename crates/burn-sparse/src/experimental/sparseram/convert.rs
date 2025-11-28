//! Conversion pipeline: Dense tensor → SparseRAM weight

use crate::core::{SparseMask, SparseTensor};
use crate::experimental::sparseram::{
    config::SparseRAMConfig,
    error::{SparseRAMError, SparseRAMResult},
    storage::PrunedStorage,
};
use alloc::vec::Vec;
use burn_core::tensor::{backend::Backend, Tensor};

/// Pipeline for converting dense tensors to SparseRAM format
///
/// Handles the complete transformation:
/// 1. Convert dense + mask → SparseTensor using burn-sparse
/// 2. Extract pruned values (if PrunedStorage is Ram/Disk)
/// 3. Return sparse tensor + pruned storage
///
/// The SparseTensor format (CSR/COO/BlockCSR) is chosen automatically by
/// burn-sparse based on the sparsity pattern.
///
/// # Memory Savings
///
/// VRAM reduction comes from CSR format storing only non-zeros:
/// - Dense: n_rows × n_cols × dtype_size
/// - Sparse (CSR): nnz × dtype_size + index overhead
/// - At 70% sparsity: ~70% VRAM reduction
///
/// # Example
///
/// ```ignore
/// let pipeline = ConversionPipeline::new(config);
/// let (sparse, pruned_storage) = pipeline.convert(dense, mask)?;
/// ```
pub struct ConversionPipeline {
    config: SparseRAMConfig,
}

impl ConversionPipeline {
    /// Create new conversion pipeline
    pub fn new(config: SparseRAMConfig) -> Self {
        Self { config }
    }

    /// Convert dense tensor and mask to SparseRAM components
    ///
    /// # Arguments
    /// * `dense` - Dense weight tensor [n_rows, n_cols]
    /// * `mask` - Sparsity mask indicating active elements
    ///
    /// # Returns
    /// Tuple of (sparse_tensor, pruned_storage, shape, device)
    ///
    /// # Process
    /// 1. Validate dimensions
    /// 2. Use burn-sparse to convert dense + mask → SparseTensor
    ///    - Format chosen automatically (CSR/COO/BlockCSR)
    ///    - Only non-zero elements stored
    /// 3. Extract pruned values if needed (PrunedStorage::Ram/Disk)
    /// 4. Dense tensor is DROPPED, freeing GPU memory
    ///
    /// # VRAM Reduction
    /// The SparseTensor (CSR format) stores only non-zeros, so:
    /// - 50% sparsity → ~50% VRAM reduction (+ index overhead)
    /// - 70% sparsity → ~70% VRAM reduction
    /// - 90% sparsity → ~90% VRAM reduction
    pub fn convert<B: Backend>(
        &self,
        dense: Tensor<B, 2>,
        mask: SparseMask<B>,
    ) -> SparseRAMResult<(SparseTensor<B>, PrunedStorage, [usize; 2], B::Device)> {
        // Validate configuration
        self.config.validate().map_err(|e| SparseRAMError::InvalidConfig { reason: e })?;

        // Validate dimensions match
        let shape = dense.dims();
        let mask_shape = mask.shape();
        if shape != mask_shape {
            return Err(SparseRAMError::DimensionMismatch {
                expected: mask_shape,
                actual: shape,
            });
        }

        // Save device for return
        let device = dense.device();

        // Step 1: Extract pruned values BEFORE converting to sparse
        // (if we need them for PrunedStorage::Ram/Disk)
        let pruned_storage = self.build_pruned_storage(&dense, &mask)?;

        // Step 2: Convert to SparseTensor using burn-sparse
        // Format chosen automatically based on sparsity pattern
        let sparse = SparseTensor::from_mask(&mask, &dense)
            .map_err(|e| SparseRAMError::ConversionError {
                reason: alloc::format!("Failed to convert to sparse tensor: {:?}", e),
            })?;

        // Step 3: Dense tensor goes out of scope here and is DROPPED
        // This FREES the GPU memory! Only sparse tensor remains.

        Ok((sparse, pruned_storage, shape, device))
    }

    /// Build pruned storage according to policy
    ///
    /// Extracts pruned values from dense tensor if needed for training.
    fn build_pruned_storage<B: Backend>(
        &self,
        dense: &Tensor<B, 2>,
        mask: &SparseMask<B>,
    ) -> SparseRAMResult<PrunedStorage> {
        use crate::experimental::sparseram::config::PrunedStorageConfig;

        match &self.config.pruned_storage {
            PrunedStorageConfig::None => {
                // Discard pruned values entirely
                Ok(PrunedStorage::None)
            }
            PrunedStorageConfig::Ram => {
                // Extract pruned values and store in RAM
                let pruned_values = self.extract_pruned_values(dense, mask)?;
                let mut storage = PrunedStorage::new_ram();
                storage.set_values(pruned_values)?;
                Ok(storage)
            }
            #[cfg(feature = "std")]
            PrunedStorageConfig::Disk { path } => {
                // Extract pruned values and store on disk
                let pruned_values = self.extract_pruned_values(dense, mask)?;
                // Entry size: usize (row) + usize (col) + f32 (value) = 8 + 8 + 4 = 20 bytes
                let entry_size = core::mem::size_of::<usize>() * 2 + core::mem::size_of::<f32>();
                let mut storage = PrunedStorage::new_disk(
                    path.clone(),
                    pruned_values.len(),
                    entry_size,
                )?;
                storage.set_values(pruned_values)?;
                Ok(storage)
            }
        }
    }

    /// Extract pruned (zero) values from dense tensor
    ///
    /// Returns a vector of (row, col, value) tuples for all pruned elements.
    fn extract_pruned_values<B: Backend>(
        &self,
        dense: &Tensor<B, 2>,
        mask: &SparseMask<B>,
    ) -> SparseRAMResult<Vec<(usize, usize, f32)>> {
        let [_n_rows, n_cols] = dense.dims();

        // Get dense data
        let dense_data: Vec<f32> = dense
            .clone()
            .into_data()
            .to_vec()
            .map_err(|_| SparseRAMError::ConversionError {
                reason: "Failed to extract dense values".into(),
            })?;

        // Use pruned_indices from SparseMask (already in CPU memory)
        // This avoids extracting boolean tensors from CUDA which can fail
        let pruned_indices = mask.pruned_indices();

        // Extract pruned values using precomputed indices
        let mut pruned = Vec::new();
        for &flat_idx in pruned_indices {
            let row = flat_idx / n_cols;
            let col = flat_idx % n_cols;
            pruned.push((row, col, dense_data[flat_idx]));
        }

        Ok(pruned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseMask;
    use crate::experimental::sparseram::config::{PrunedStorageConfig, SparsePolicy};
    use burn_core::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn create_test_config() -> SparseRAMConfig {
        SparseRAMConfig {
            active_tier: crate::experimental::sparseram::storage::Tier::GPU,
            pruned_storage: PrunedStorageConfig::None,
            policy: SparsePolicy::Eager,
            block_size: 16, // Not used anymore, but kept for compatibility
        }
    }

    #[test]
    fn test_convert_dense_to_sparse() {
        let device = Default::default();

        // Create 10×10 tensor with 50% sparsity
        let mut data = vec![0.0f32; 100];
        for i in 0..50 {
            data[i] = 1.0;
        }
        use burn_core::tensor::{Shape, TensorData};
        let tensor_data = TensorData::new(data, Shape::new([10, 10]));
        let dense = Tensor::<TestBackend, 2>::from_data(tensor_data, &device);

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let config = create_test_config();
        let pipeline = ConversionPipeline::new(config);

        let (sparse, pruned_storage, shape, _device) = pipeline.convert(dense, mask).unwrap();

        // Check shape preserved
        assert_eq!(shape, [10, 10]);

        // Check sparsity
        assert_eq!(sparse.nnz(), 50);

        // Check pruned storage is None
        assert!(!pruned_storage.has_storage());
    }

    #[test]
    fn test_pruned_storage_none() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut config = create_test_config();
        config.pruned_storage = PrunedStorageConfig::None;

        let pipeline = ConversionPipeline::new(config);
        let (sparse, pruned_storage, _, _) = pipeline.convert(dense, mask).unwrap();

        // Pruned storage should be None
        assert!(!pruned_storage.has_storage());

        // Sparse tensor should have 3 non-zeros
        assert_eq!(sparse.nnz(), 3);
    }

    #[test]
    fn test_pruned_storage_ram() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut config = create_test_config();
        config.pruned_storage = PrunedStorageConfig::Ram;

        let pipeline = ConversionPipeline::new(config);
        let (sparse, pruned_storage, _, _) = pipeline.convert(dense, mask).unwrap();

        // Pruned storage should have data
        assert!(pruned_storage.has_storage());

        // Should have 3 pruned values (the zeros)
        assert_eq!(pruned_storage.num_values(), 3);

        // Sparse tensor should have 3 non-zeros
        assert_eq!(sparse.nnz(), 3);
    }
}
