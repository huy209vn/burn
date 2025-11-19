//! Sparse linear layer with efficient sparse matrix multiplication

use burn::{
    module::{Module, Param},
    nn::Linear,
    tensor::{backend::Backend, Int, Tensor},
};

use crate::core::{SparseFormat, SparseMask, SparseTensor, SparseTensorData};
use crate::kernel::{SparseConfig, SparseDispatch};

/// Configuration for sparse linear layer
#[derive(Debug, Clone)]
pub struct SparseLinearConfig {
    /// Sparse tensor format (CSR, COO, etc.)
    pub format: SparseFormat,

    /// Kernel dispatch configuration
    pub sparse_config: SparseConfig,
}

impl Default for SparseLinearConfig {
    fn default() -> Self {
        Self {
            format: SparseFormat::CSR,
            sparse_config: SparseConfig::default(),
        }
    }
}

/// Sparse linear layer: Y = W_sparse @ X + b
///
/// Stores sparse weights as CSR format (values, col_indices, row_pointers).
/// Only values are trainable; indices remain fixed during training.
#[derive(Debug, Module)]
pub struct SparseLinear<B: Backend> {
    /// CSR values [nnz] - trainable weights
    pub weight_values: Param<Tensor<B, 1>>,

    /// Dense bias [d_output] - trainable
    pub bias: Option<Param<Tensor<B, 1>>>,

    /// CSR column indices [nnz] - fixed structure
    #[module(skip)]
    pub weight_col_indices: Tensor<B, 1, Int>,

    /// CSR row pointers [n_rows + 1] - fixed structure
    #[module(skip)]
    pub weight_row_pointers: Tensor<B, 1, Int>,

    /// Weight shape [d_output, d_input]
    #[module(skip)]
    pub shape: [usize; 2],
}

impl<B: Backend> SparseLinear<B> {
    /// Create sparse linear from existing Linear layer and mask
    pub fn from_linear(
        linear: Linear<B>,
        mask: SparseMask<B>,
        format: SparseFormat,
    ) -> Self {
        let weight_data = linear.weight.val();
        let shape = weight_data.dims();

        // Apply mask and convert to sparse format (always CSR for storage)
        let sparse_weight = SparseTensor::from_mask(&mask, &weight_data)
            .expect("Failed to convert mask to sparse tensor");

        // Extract CSR components
        let (values, col_indices, row_pointers) = match sparse_weight.data() {
            SparseTensorData::CSR {
                values,
                col_indices,
                row_pointers,
            } => (values.clone(), col_indices.clone(), row_pointers.clone()),
            _ => {
                // Convert to CSR first if needed
                let csr = sparse_weight
                    .to_format(SparseFormat::CSR)
                    .expect("Failed to convert to CSR");
                match csr.data() {
                    SparseTensorData::CSR {
                        values,
                        col_indices,
                        row_pointers,
                    } => (values.clone(), col_indices.clone(), row_pointers.clone()),
                    _ => unreachable!(),
                }
            }
        };

        Self {
            weight_values: Param::from_tensor(values),
            bias: linear.bias,
            weight_col_indices: col_indices,
            weight_row_pointers: row_pointers,
            shape,
        }
    }

    /// Get SparseTensor representation of weights
    fn weight_sparse(&self) -> SparseTensor<B> {
        SparseTensor::from_csr(
            self.weight_values.val(),
            self.weight_col_indices.clone(),
            self.weight_row_pointers.clone(),
            self.shape,
            self.weight_values.device(),
        )
    }

    /// Forward pass: Y = W @ X + b
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight_sparse();

        // Sparse matmul: Y = W_sparse @ X
        let config = SparseConfig::default();
        let output = SparseDispatch::spmm(&weight, &input, &config)
            .expect("SpMM failed during forward pass");

        // Add bias if present
        // Bias has shape [d_output], output has shape [d_output, batch_size]
        // We need to reshape bias to [d_output, 1] to broadcast correctly
        match &self.bias {
            Some(bias) => {
                let bias_reshaped = bias.val().clone().unsqueeze_dim(1);
                output + bias_reshaped
            }
            None => output,
        }
    }

    /// Convert back to dense Linear layer
    pub fn to_dense(&self) -> Linear<B> {
        let weight = self.weight_sparse();
        let dense_weight = weight.to_dense();

        Linear {
            weight: Param::from_tensor(dense_weight),
            bias: self.bias.clone(),
        }
    }

    /// Update sparsity structure (for dynamic sparse training)
    ///
    /// Reconstructs the layer with a new sparsity pattern.
    pub fn update_from_mask(&mut self, new_mask: SparseMask<B>) {
        // Get current dense weights
        let dense = self.weight_sparse().to_dense();

        // Apply new mask and convert to sparse
        let new_sparse = SparseTensor::from_mask(&new_mask, &dense)
            .expect("Failed to update mask");

        // Extract CSR components
        let csr = new_sparse.to_format(SparseFormat::CSR).unwrap();
        match csr.data() {
            SparseTensorData::CSR {
                values,
                col_indices,
                row_pointers,
            } => {
                self.weight_values = Param::from_tensor(values.clone());
                self.weight_col_indices = col_indices.clone();
                self.weight_row_pointers = row_pointers.clone();
            }
            _ => unreachable!(),
        }
    }

    /// Get current sparsity ratio
    pub fn sparsity(&self) -> f32 {
        self.weight_sparse().sparsity()
    }

    /// Get number of parameters (active + bias)
    pub fn num_params(&self) -> usize {
        let weight_params = self.weight_values.val().dims()[0]; // nnz
        let bias_params = self.bias.as_ref().map(|b| b.val().dims()[0]).unwrap_or(0);
        weight_params + bias_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_sparse_linear_from_linear() {
        let device = <TestBackend as Backend>::Device::default();

        // Create dense linear
        let dense_linear = Linear::<TestBackend> {
            weight: Param::from_tensor(Tensor::from_data(
                TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                &device,
            )),
            bias: Some(Param::from_tensor(Tensor::from_data([0.1, 0.2], &device))),
        };

        // Create mask (prune position [0,1] and [1,2])
        let mask_data = Tensor::from_data(
            TensorData::from([[true, false, true], [true, true, false]]),
            &device,
        );
        let mask = SparseMask::from_tensor(mask_data);

        // Convert to sparse
        let sparse_linear = SparseLinear::from_linear(dense_linear, mask, SparseFormat::CSR);

        // Check properties
        let sparsity = sparse_linear.sparsity();
        assert!((sparsity - 1.0 / 3.0).abs() < 1e-5, "Sparsity mismatch: {}", sparsity); // 2/6 pruned
        assert_eq!(sparse_linear.num_params(), 4 + 2); // 4 active weights + 2 bias
    }

    #[test]
    fn test_sparse_linear_forward() {
        let device = <TestBackend as Backend>::Device::default();

        // Dense linear for comparison
        let weight: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::from([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
            &device,
        );
        let bias: Tensor<TestBackend, 1> = Tensor::from_data([0.1, 0.2], &device);

        // Create mask from non-zeros
        let mask_data = weight.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        // Create dense linear first
        let dense_linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: Some(Param::from_tensor(bias.clone())),
        };

        // Convert to sparse
        let sparse_linear = SparseLinear::from_linear(dense_linear, mask, SparseFormat::CSR);

        // Input
        let input: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::from([[2.0], [3.0], [4.0]]), &device);

        // Forward pass
        let output = sparse_linear.forward(input);

        // Expected: [1*2 + 0*3 + 3*4 + 0.1, 0*2 + 5*3 + 0*4 + 0.2]
        //         = [14.1, 15.2]
        let expected: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::from([[14.1], [15.2]]), &device);

        let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
        let expected_data: Vec<f32> = expected.into_data().to_vec().unwrap();

        for (o, e) in output_data.iter().zip(expected_data.iter()) {
            assert!((o - e).abs() < 1e-5, "Output mismatch: {} vs {}", o, e);
        }
    }
}
