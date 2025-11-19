/// Sparse tensor with format-specific storage
///
/// This is the **execution format** - optimized for kernels, not algorithms.
/// For algorithm manipulation, use SparseMask.
///
/// # Design Philosophy
/// - Immutable after construction (except for mask updates in dynamic sparsity)
/// - Always valid: panics on invalid construction, never creates broken state
/// - Format-polymorphic: supports CSR, COO, BlockCSR, N:M, etc.
/// - Backend-agnostic: works with any Burn backend

use burn_core::module::Parameter;
use burn_core::tensor::{backend::Backend, Bool, Int, Shape, Tensor, TensorData};

use crate::core::{SparseError, SparseFormat, SparseResult};

/// Sparse tensor containing format-specific data
#[derive(Debug, Clone)]
pub struct SparseTensor<B: Backend> {
    /// Storage format
    format: SparseFormat,

    /// Tensor shape [n_rows, n_cols]
    shape: [usize; 2],

    /// Format-specific data
    data: SparseTensorData<B>,

    /// Device location
    device: B::Device,
}

/// Format-specific storage for sparse tensors
///
/// Each variant stores the minimum data needed for that format.
/// All tensors are on the same device as the parent SparseTensor.
#[derive(Debug, Clone)]
pub enum SparseTensorData<B: Backend> {
    /// Boolean mask + dense values (masked storage)
    Mask {
        /// Boolean mask [n_rows, n_cols]
        mask: Tensor<B, 2, Bool>,
        /// Dense values [n_rows, n_cols] (zeros where mask is false)
        values: Tensor<B, 2>,
    },

    /// Compressed Sparse Row
    CSR {
        /// Non-zero values [nnz]
        values: Tensor<B, 1>,
        /// Column indices [nnz]
        col_indices: Tensor<B, 1, Int>,
        /// Row pointers [n_rows + 1]
        row_pointers: Tensor<B, 1, Int>,
    },

    /// Compressed Sparse Column
    CSC {
        /// Non-zero values [nnz]
        values: Tensor<B, 1>,
        /// Row indices [nnz]
        row_indices: Tensor<B, 1, Int>,
        /// Column pointers [n_cols + 1]
        col_pointers: Tensor<B, 1, Int>,
    },

    /// Coordinate list
    COO {
        /// Non-zero values [nnz]
        values: Tensor<B, 1>,
        /// Row indices [nnz]
        row_indices: Tensor<B, 1, Int>,
        /// Column indices [nnz]
        col_indices: Tensor<B, 1, Int>,
    },

    /// Block-sparse CSR (for GPU tensor cores)
    BlockCSR {
        /// Dense blocks [n_blocks, block_size²] flattened
        blocks: Tensor<B, 2>,
        /// Block column indices [n_blocks]
        block_col_indices: Tensor<B, 1, Int>,
        /// Block row pointers [n_block_rows + 1]
        block_row_pointers: Tensor<B, 1, Int>,
        /// Block size (must be power of 2)
        block_size: usize,
    },

    /// N:M structured sparsity (hardware-accelerated)
    NInM {
        /// Packed values (only N per M stored)
        values: Tensor<B, 2>,
        /// Metadata encoding which N of M are active
        metadata: Tensor<B, 2, Int>,
        /// N (non-zeros per group)
        n: usize,
        /// M (group size)
        m: usize,
    },
}

impl<B: Backend> SparseTensor<B> {
    // ============================================================================
    // Construction
    // ============================================================================

    /// Create sparse tensor from dense tensor and format
    ///
    /// # Arguments
    /// * `dense` - Input dense tensor [n_rows, n_cols]
    /// * `format` - Target sparse format
    /// * `threshold` - Values with |val| < threshold are pruned
    ///
    /// # Panics
    /// Panics if format construction fails (invalid format parameters, etc.)
    ///
    /// # Example
    /// ```ignore
    /// use burn_sparse::core::{SparseTensor, SparseFormat};
    ///
    /// let dense = Tensor::from_data([[1.0, 0.0], [2.0, 3.0]]);
    /// let sparse = SparseTensor::from_dense(&dense, SparseFormat::CSR, 0.1);
    /// ```
    pub fn from_dense(
        dense: &Tensor<B, 2>,
        format: SparseFormat,
        threshold: f32,
    ) -> Self {
        let shape = dense.dims();
        let device = dense.device();

        // Construct format-specific data
        let data = match format {
            SparseFormat::Mask => Self::construct_mask(dense, threshold),
            SparseFormat::CSR => Self::construct_csr(dense, threshold),
            SparseFormat::CSC => Self::construct_csc(dense, threshold),
            SparseFormat::COO => Self::construct_coo(dense, threshold),
            SparseFormat::BlockCSR { block_size } => {
                SparseFormat::validate_block_size(block_size)
                    .expect("Invalid block size");
                Self::construct_block_csr(dense, threshold, block_size)
            }
            SparseFormat::NInM { n, m } => {
                SparseFormat::validate_nm(n, m)
                    .expect("Invalid N:M parameters");
                Self::construct_nm(dense, n, m, threshold)
            }
        };

        let sparse = Self {
            format,
            shape,
            data,
            device,
        };

        // Validate construction
        sparse.validate().expect("Sparse tensor construction created invalid structure");

        sparse
    }

    /// Create sparse tensor from CSR format data
    ///
    /// # Arguments
    /// * `values` - Non-zero values [nnz]
    /// * `col_indices` - Column indices [nnz]
    /// * `row_pointers` - Row pointers [n_rows + 1]
    /// * `shape` - Tensor shape [n_rows, n_cols]
    /// * `device` - Device location
    ///
    /// # Panics
    /// Panics if CSR data is invalid
    ///
    /// # Example
    /// ```ignore
    /// use burn_sparse::core::SparseTensor;
    ///
    /// let values = Tensor::from_data([1.0, 2.0, 3.0]);
    /// let col_indices = Tensor::from_data([0, 1, 0]);
    /// let row_pointers = Tensor::from_data([0, 2, 3]);
    /// let sparse = SparseTensor::from_csr(values, col_indices, row_pointers, [2, 3], device);
    /// ```
    pub fn from_csr(
        values: Tensor<B, 1>,
        col_indices: Tensor<B, 1, Int>,
        row_pointers: Tensor<B, 1, Int>,
        shape: [usize; 2],
        device: B::Device,
    ) -> Self {
        let data = SparseTensorData::CSR {
            values,
            col_indices,
            row_pointers,
        };

        let sparse = Self {
            format: SparseFormat::CSR,
            shape,
            data,
            device,
        };

        // Validate construction
        sparse.validate().expect("CSR tensor construction created invalid structure");

        sparse
    }

    /// Create sparse tensor from SparseMask and dense weights
    ///
    /// This converts the algorithm-layer mask to execution-layer CSR format.
    /// Conversion happens on CPU, then tensors are uploaded to the device.
    ///
    /// # Arguments
    /// * `mask` - Sparsity mask indicating active weights
    /// * `weights` - Dense weight tensor [n_out, n_in]
    ///
    /// # Returns
    /// SparseTensor in CSR format containing only active (masked) weights
    ///
    /// # Panics
    /// Panics if mask and weights shapes don't match
    ///
    /// # Example
    /// ```ignore
    /// use burn_sparse::prelude::*;
    ///
    /// // Create mask from Wanda
    /// let mask = SparseMask::from_scores(&scores, 0.5);
    ///
    /// // Convert to CSR sparse tensor
    /// let sparse = SparseTensor::from_mask(&mask, &weights);
    /// ```
    pub fn from_mask(
        mask: &crate::core::SparseMask<B>,
        weights: &Tensor<B, 2>,
    ) -> SparseResult<Self> {
        let shape = mask.shape();
        let device = weights.device();

        if weights.dims() != shape {
            return Err(SparseError::InvalidTensor {
                reason: format!(
                    "Mask shape {:?} doesn't match weights shape {:?}",
                    shape,
                    weights.dims()
                ),
            });
        }

        // Use mask_to_csr conversion
        crate::core::convert::mask_to_csr(mask.tensor(), weights, shape, &device)
    }

    // ============================================================================
    // Accessors
    // ============================================================================

    /// Get storage format
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// Get tensor shape [n_rows, n_cols]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Get device
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        match &self.data {
            SparseTensorData::Mask { mask, .. } => {
                // Count true values in mask
                use burn_core::tensor::ElementConversion;
                let count: i64 = mask.clone().int().sum().into_scalar().elem();
                count as usize
            }
            SparseTensorData::CSR { values, .. } => values.dims()[0],
            SparseTensorData::CSC { values, .. } => values.dims()[0],
            SparseTensorData::COO { values, .. } => values.dims()[0],
            SparseTensorData::BlockCSR { blocks, block_size, .. } => {
                let n_blocks = blocks.dims()[0];
                n_blocks * block_size * block_size
            }
            SparseTensorData::NInM { n, m, .. } => {
                let total_elements = self.shape[0] * self.shape[1];
                let n_groups = total_elements / m;
                n_groups * n
            }
        }
    }

    /// Get actual sparsity (fraction of zeros)
    pub fn sparsity(&self) -> f32 {
        let total = self.shape[0] * self.shape[1];
        let nnz = self.nnz();
        1.0 - (nnz as f32 / total as f32)
    }

    // ============================================================================
    // Format Conversions (Forward declarations - implemented in convert.rs)
    // ============================================================================

    /// Convert to different sparse format
    ///
    /// All conversions route through convert::convert_format()
    pub fn to_format(&self, target: SparseFormat) -> SparseResult<Self> {
        crate::core::convert::convert_format(self, target)
    }

    /// Convert to dense tensor
    pub fn to_dense(&self) -> Tensor<B, 2> {
        crate::core::convert::to_dense(self)
    }

    // ============================================================================
    // Validation (Forward declaration - implemented in validate.rs)
    // ============================================================================

    fn validate(&self) -> SparseResult<()> {
        crate::core::validate::validate_sparse_tensor(self)
    }

    // ============================================================================
    // Construction Helpers (Private - format-specific)
    // ============================================================================

    fn construct_mask(dense: &Tensor<B, 2>, threshold: f32) -> SparseTensorData<B> {
        // Create boolean mask: |value| >= threshold
        let abs_values = dense.clone().abs();
        let mask = abs_values.greater_equal_elem(threshold);

        // Masked values: zero out below threshold
        let values = dense.clone().mask_fill(mask.clone().bool_not(), 0.0);

        SparseTensorData::Mask { mask, values }
    }

    fn construct_csr(dense: &Tensor<B, 2>, threshold: f32) -> SparseTensorData<B> {
        // This is a placeholder - full implementation goes in convert.rs
        // For now, convert Mask → CSR
        let mask_data = Self::construct_mask(dense, threshold);
        // TODO: Implement actual CSR construction
        // For now, return empty CSR
        let nnz = 0;
        SparseTensorData::CSR {
            values: Tensor::zeros([nnz], &dense.device()),
            col_indices: Tensor::zeros([nnz], &dense.device()),
            row_pointers: Tensor::zeros([dense.dims()[0] + 1], &dense.device()),
        }
    }

    fn construct_csc(dense: &Tensor<B, 2>, threshold: f32) -> SparseTensorData<B> {
        // Placeholder
        let nnz = 0;
        SparseTensorData::CSC {
            values: Tensor::zeros([nnz], &dense.device()),
            row_indices: Tensor::zeros([nnz], &dense.device()),
            col_pointers: Tensor::zeros([dense.dims()[1] + 1], &dense.device()),
        }
    }

    fn construct_coo(dense: &Tensor<B, 2>, threshold: f32) -> SparseTensorData<B> {
        // Placeholder
        let nnz = 0;
        SparseTensorData::COO {
            values: Tensor::zeros([nnz], &dense.device()),
            row_indices: Tensor::zeros([nnz], &dense.device()),
            col_indices: Tensor::zeros([nnz], &dense.device()),
        }
    }

    fn construct_block_csr(
        dense: &Tensor<B, 2>,
        threshold: f32,
        block_size: usize,
    ) -> SparseTensorData<B> {
        // Placeholder
        let n_blocks = 0;
        let n_block_rows = (dense.dims()[0] + block_size - 1) / block_size;

        SparseTensorData::BlockCSR {
            blocks: Tensor::zeros([n_blocks, block_size * block_size], &dense.device()),
            block_col_indices: Tensor::zeros([n_blocks], &dense.device()),
            block_row_pointers: Tensor::zeros([n_block_rows + 1], &dense.device()),
            block_size,
        }
    }

    fn construct_nm(
        dense: &Tensor<B, 2>,
        n: usize,
        m: usize,
        threshold: f32,
    ) -> SparseTensorData<B> {
        // Placeholder
        let shape = dense.dims();
        let total_groups = (shape[0] * shape[1]) / m;

        SparseTensorData::NInM {
            values: Tensor::zeros([total_groups, n], &dense.device()),
            metadata: Tensor::zeros([total_groups, 1], &dense.device()),
            n,
            m,
        }
    }
}

// Getters for internal data (used by kernels and conversions)
impl<B: Backend> SparseTensor<B> {
    /// Get reference to format-specific data (internal use)
    pub(crate) fn data(&self) -> &SparseTensorData<B> {
        &self.data
    }
}

// Parameter trait implementation - enables SparseTensor to be used in Param<SparseTensor<B>>
impl<B: Backend> Parameter for SparseTensor<B> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    /// Check if values tensor requires gradients
    ///
    /// Note: Only values can have gradients in sparse tensors.
    /// Indices are discrete and don't participate in gradient flow.
    fn is_require_grad(&self) -> bool {
        match &self.data {
            SparseTensorData::Mask { values, .. } => values.is_require_grad(),
            SparseTensorData::CSR { values, .. } => values.is_require_grad(),
            SparseTensorData::CSC { values, .. } => values.is_require_grad(),
            SparseTensorData::COO { values, .. } => values.is_require_grad(),
            SparseTensorData::BlockCSR { blocks, .. } => blocks.is_require_grad(),
            SparseTensorData::NInM { values, .. } => values.is_require_grad(),
        }
    }

    /// Set gradient requirement on values tensor
    ///
    /// Note: This only applies to values. Indices never require gradients
    /// as the sparsity structure is either fixed or changed discretely by
    /// algorithms like RigL/MEST.
    fn set_require_grad(mut self, require_grad: bool) -> Self {
        self.data = match self.data {
            SparseTensorData::Mask { mask, values } => SparseTensorData::Mask {
                mask,
                values: values.set_require_grad(require_grad),
            },
            SparseTensorData::CSR {
                values,
                col_indices,
                row_pointers,
            } => SparseTensorData::CSR {
                values: values.set_require_grad(require_grad),
                col_indices,
                row_pointers,
            },
            SparseTensorData::CSC {
                values,
                row_indices,
                col_pointers,
            } => SparseTensorData::CSC {
                values: values.set_require_grad(require_grad),
                row_indices,
                col_pointers,
            },
            SparseTensorData::COO {
                values,
                row_indices,
                col_indices,
            } => SparseTensorData::COO {
                values: values.set_require_grad(require_grad),
                row_indices,
                col_indices,
            },
            SparseTensorData::BlockCSR {
                blocks,
                block_col_indices,
                block_row_pointers,
                block_size,
            } => SparseTensorData::BlockCSR {
                blocks: blocks.set_require_grad(require_grad),
                block_col_indices,
                block_row_pointers,
                block_size,
            },
            SparseTensorData::NInM {
                values,
                metadata,
                n,
                m,
            } => SparseTensorData::NInM {
                values: values.set_require_grad(require_grad),
                metadata,
                n,
                m,
            },
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TB = NdArray<f32>;

    #[test]
    fn test_mask_construction() {
        let dense = Tensor::<TB, 2>::from_data(
            [[1.0, 0.1], [0.05, 2.0]],
            &Default::default(),
        );

        let sparse = SparseTensor::from_dense(&dense, SparseFormat::Mask, 0.1);

        assert_eq!(sparse.shape(), [2, 2]);
        assert_eq!(sparse.format(), SparseFormat::Mask);
        // Should have 3 non-zeros (1.0, 0.1, and 2.0), pruning only 0.05
        // Threshold logic: |val| >= threshold, so 0.1 >= 0.1 is kept
        assert_eq!(sparse.nnz(), 3);
        assert!((sparse.sparsity() - 0.25).abs() < 0.01); // 1/4 pruned = 0.25 sparsity
    }

    #[test]
    fn test_format_validation() {
        assert!(SparseFormat::validate_block_size(16).is_ok());
        assert!(SparseFormat::validate_nm(2, 4).is_ok());
    }

    #[test]
    fn test_from_mask() {
        use crate::core::SparseMask;

        let device = Default::default();

        // Create dense weights
        let weights = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            &device,
        );

        // Create mask (prune positions [0,1] and [1,2])
        let mask_tensor = Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false, true],
                [true, true, false],
            ],
            &device,
        );
        let mask = SparseMask::from_tensor(mask_tensor);

        // Convert to SparseTensor
        let sparse = SparseTensor::from_mask(&mask, &weights).unwrap();

        // Verify
        assert_eq!(sparse.format(), SparseFormat::CSR);
        assert_eq!(sparse.shape(), [2, 3]);
        assert_eq!(sparse.nnz(), 4); // 4 active positions

        // Verify CSR structure
        match sparse.data() {
            SparseTensorData::CSR { values, col_indices, row_pointers } => {
                let vals: Vec<f32> = values.clone().into_data().to_vec().unwrap();
                assert_eq!(vals, vec![1.0, 3.0, 4.0, 5.0]);

                let cols: Vec<i64> = col_indices.clone().into_data().to_vec().unwrap();
                assert_eq!(cols, vec![0, 2, 0, 1]);

                let ptrs: Vec<i64> = row_pointers.clone().into_data().to_vec().unwrap();
                assert_eq!(ptrs, vec![0, 2, 4]);
            }
            _ => panic!("Expected CSR format"),
        }
    }
}
