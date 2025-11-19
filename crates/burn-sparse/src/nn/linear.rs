//! Sparse linear layer with efficient sparse matrix multiplication

use burn::{
    module::Module,
    nn::Linear,
    tensor::{backend::Backend, Tensor},
};

use crate::core::{SparseFormat, SparseMask, SparseParam, SparseTensor};
use crate::kernel::{SparseConfig, SparseDispatch};

/// Sparse linear layer: Y = W_sparse @ X + b
///
/// Stores weights as SparseTensor (default CSR format).
/// The sparsity structure is fixed during training; only values are updated.
///
/// # Example
///
/// ```ignore
/// use burn_sparse::prelude::*;
/// use burn::nn::Linear;
///
/// // Start with dense linear
/// let dense_linear = Linear::new(512, 256);
///
/// // Prune with algorithm
/// let mask = Wanda::new(config).prune(&dense_linear.weight, &calibration_data);
///
/// // Convert to sparse
/// let sparse_linear = SparseLinear::from_linear(dense_linear, mask);
///
/// // Use in training
/// let output = sparse_linear.forward(input);
/// ```
#[derive(Debug, Module)]
pub struct SparseLinear<B: Backend> {
    /// Sparse weight tensor [d_output, d_input]
    pub weight: SparseParam<B>,

    /// Dense bias [d_output]
    pub bias: Option<burn::module::Param<Tensor<B, 1>>>,

}

impl<B: Backend> SparseLinear<B> {
    /// Create sparse linear from existing Linear layer and mask
    ///
    /// # Arguments
    /// * `linear` - Dense linear layer to convert
    /// * `mask` - Sparsity mask (from Wanda, Magnitude, etc.)
    ///
    /// # Example
    /// ```ignore
    /// let sparse_linear = SparseLinear::from_linear(dense_linear, mask);
    /// ```
    pub fn from_linear(
        linear: Linear<B>,
        mask: SparseMask<B>,
    ) -> Self {
        let weight_data = linear.weight.val();

        // Apply mask and convert to sparse format (CSR by default)
        let sparse_weight = SparseTensor::from_mask(&mask, &weight_data)
            .expect("Failed to convert mask to sparse tensor")
            .to_format(SparseFormat::CSR)
            .expect("Failed to convert to CSR");

        Self {
            weight: SparseParam::from_sparse(sparse_weight),
            bias: linear.bias,
            
        }
    }

    /// Forward pass: Y = W @ X + b
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight.val();

        // Sparse matmul: Y = W_sparse @ X
        let output = SparseDispatch::spmm(&weight, &input, &SparseConfig::default())
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
        let weight = self.weight.val();
        let dense_weight = weight.to_dense();

        Linear {
            weight: burn::module::Param::from_tensor(dense_weight),
            bias: self.bias.clone(),
        }
    }

    /// Update sparsity structure (for dynamic sparse training)
    ///
    /// Reconstructs the layer with a new sparsity pattern.
    /// Used by RigL, MEST, and other dynamic sparse training methods.
    pub fn update_from_mask(&mut self, new_mask: SparseMask<B>) {
        // Get current dense weights
        let dense = self.weight.val().to_dense();

        // Apply new mask and convert to sparse
        let new_sparse = SparseTensor::from_mask(&new_mask, &dense)
            .expect("Failed to update mask")
            .to_format(SparseFormat::CSR)
            .expect("Failed to convert to CSR");

        self.weight = SparseParam::from_sparse(new_sparse);
    }

    /// Get current sparsity ratio
    pub fn sparsity(&self) -> f32 {
        self.weight.val().sparsity()
    }

    /// Get number of non-zero parameters
    pub fn nnz(&self) -> usize {
        self.weight.val().nnz()
    }

    /// Get total number of trainable parameters
    pub fn num_params(&self) -> usize {
        let weight_params = self.nnz(); // Only non-zero values are parameters
        let bias_params = self.bias.as_ref().map(|b| b.val().dims()[0]).unwrap_or(0);
        weight_params + bias_params
    }

    /// Get weight shape
    pub fn shape(&self) -> [usize; 2] {
        self.weight.shape()
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
            weight: burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::from([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
                &device,
            )),
            bias: Some(burn::module::Param::from_tensor(Tensor::from_data(
                [0.1, 0.2],
                &device,
            ))),
        };

        // Create mask (keep non-zeros)
        let mask_data = dense_linear
            .weight
            .val()
            .clone()
            .not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        // Convert to sparse
        let sparse_linear = SparseLinear::from_linear(dense_linear, mask);

        // Check properties
        let sparsity = sparse_linear.sparsity();
        assert!((sparsity - 0.5).abs() < 1e-5, "Sparsity mismatch: {}", sparsity); // 3/6 pruned
        assert_eq!(sparse_linear.num_params(), 3 + 2); // 3 active weights + 2 bias
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

        let dense_linear = Linear::<TestBackend> {
            weight: burn::module::Param::from_tensor(weight),
            bias: Some(burn::module::Param::from_tensor(bias)),
        };

        // Convert to sparse
        let sparse_linear = SparseLinear::from_linear(dense_linear, mask);

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
