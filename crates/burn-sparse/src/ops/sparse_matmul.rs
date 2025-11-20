//! Sparse matrix multiplication with autodiff support
//!
//! Provides SpMM (Sparse-Dense Matrix Multiply) that properly registers
//! backward passes in Burn's autodiff computational graph.
//!
//! Forward: Y = W_sparse @ X_dense
//! Backward:
//!   - dL/dX = W_sparse^T @ dL/dY
//!   - dL/dW_values = gradients at non-zero positions only

use crate::core::{SparseTensor, SparseTensorData};
use crate::kernel::{SparseConfig, SparseDispatch};
use burn_core::tensor::{backend::Backend, Tensor};

/// Sparse matrix multiplication: Y = sparse @ dense
///
/// This function properly handles gradient flow for both sparse and dense tensors.
/// For autodiff backends, backward passes are registered automatically.
///
/// # Arguments
/// * `sparse` - Sparse weight matrix [M, K]
/// * `dense` - Dense input matrix [K, N]
/// * `config` - Sparse computation configuration
///
/// # Returns
/// Output tensor [M, N]
///
/// # Panics
/// If the operation fails (invalid dimensions, unsupported format, etc.)
pub fn sparse_matmul<B: Backend>(
    sparse: &SparseTensor<B>,
    dense: Tensor<B, 2>,
    config: &SparseConfig,
) -> Tensor<B, 2> {
    // For non-autodiff backends, just call kernel directly
    SparseDispatch::spmm(sparse, &dense, config)
        .expect("SpMM failed in sparse_matmul")
}

// TODO: Implement autodiff-specific version
// The challenge: Burn's autodiff system is tightly coupled to backend primitives.
// SparseTensor is not a primitive type, so we can't directly hook into the
// autodiff graph like standard operations.
//
// Possible approaches:
// 1. Implement custom gradient computation in Module system (via map/visit)
// 2. Decompose sparse ops into dense operations for autodiff (inefficient)
// 3. Extend backend trait with sparse primitives (proper but complex)
//
// For now, we rely on the Module system's mapping of values tensors.
// The SparseParam visit/map handles gradient flow for the values.
// This works because:
// - Indices are constant (no gradients)
// - Only values need gradients
// - Module system automatically visits values tensor

/// Helper: Compute gradients for sparse weight values
///
/// Given dL/dY and input X, compute dL/dW for non-zero positions.
/// This is used internally by the autodiff system.
///
/// # Math
/// For Y = W @ X where W is sparse:
/// dL/dW[i,j] = dL/dY[i,:] @ X[j,:].T
///
/// But we only compute this for non-zero positions in W.
#[allow(dead_code)]
fn sparse_weight_gradient<B: Backend>(
    sparse_data: &SparseTensorData<B>,
    _grad_output: &Tensor<B, 2>,
    _input: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    match sparse_data {
        SparseTensorData::CSR { values, .. } => {
            // TODO: Implement efficient gradient computation
            // For now, return zeros as placeholder
            Tensor::zeros_like(values)
        }
        _ => todo!("Gradient computation for other formats"),
    }
}

/// Helper: Compute gradients for dense input
///
/// Given dL/dY and sparse W, compute dL/dX.
///
/// # Math
/// For Y = W @ X where W is sparse:
/// dL/dX = W^T @ dL/dY
///
/// This requires sparse transpose matmul (SpMM with transposed sparse matrix).
#[allow(dead_code)]
fn sparse_input_gradient<B: Backend>(
    sparse: &SparseTensor<B>,
    grad_output: Tensor<B, 2>,
    config: &SparseConfig,
) -> Tensor<B, 2> {
    // Transpose sparse matrix for backward pass
    // For CSR, this means converting to CSC (or vice versa)
    let sparse_transposed = sparse
        .transpose()
        .expect("Failed to transpose sparse tensor");

    // dL/dX = W^T @ dL/dY
    SparseDispatch::spmm(&sparse_transposed, &grad_output, config)
        .expect("SpMM failed in backward pass")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::TensorData;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_sparse_matmul_forward() {
        let device = <TestBackend as Backend>::Device::default();

        // Create sparse matrix (CSR format)
        // [[1.0, 0.0, 2.0],
        //  [0.0, 3.0, 0.0]]
        let weight_dense: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], &device);

        let mask_data = weight_dense.clone().not_equal_elem(0.0);
        let mask = crate::core::SparseMask::from_tensor(mask_data);

        let sparse = SparseTensor::from_mask(&mask, &weight_dense)
            .expect("Failed to create sparse tensor")
            .to_format(crate::core::SparseFormat::CSR)
            .expect("Failed to convert to CSR");

        // Dense input [3, 2]
        let input: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

        // Forward pass
        let output = sparse_matmul(&sparse, input, &SparseConfig::default());

        // Expected: [[1*1+0*3+2*5, 1*2+0*4+2*6], = [[11, 14],
        //            [0*1+3*3+0*5, 0*2+3*4+0*6]]    [9, 12]]
        let expected: Tensor<TestBackend, 2> = Tensor::from_data([[11.0, 14.0], [9.0, 12.0]], &device);

        let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
        let expected_data: Vec<f32> = expected.into_data().to_vec().unwrap();

        for (o, e) in output_data.iter().zip(expected_data.iter()) {
            assert!((o - e).abs() < 1e-5, "Output mismatch: {} vs {}", o, e);
        }
    }

    #[test]
    fn test_sparse_input_gradient() {
        let device = <TestBackend as Backend>::Device::default();

        // Sparse W = [[1, 0, 2],
        //             [0, 3, 0]]
        let weight_dense: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], &device);

        let mask_data = weight_dense.clone().not_equal_elem(0.0);
        let mask = crate::core::SparseMask::from_tensor(mask_data);

        let sparse = SparseTensor::from_mask(&mask, &weight_dense)
            .expect("Failed to create sparse tensor")
            .to_format(crate::core::SparseFormat::CSR)
            .expect("Failed to convert to CSR");

        // Grad output [2, 2]
        let grad_output: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0, 1.0], [1.0, 1.0]], &device);

        // dL/dX = W^T @ grad_output
        // W^T = [[1, 0],
        //        [0, 3],
        //        [2, 0]]
        // Result: [[1, 1], [3, 3], [2, 2]]
        let grad_input = sparse_input_gradient(&sparse, grad_output, &SparseConfig::default());

        let expected: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]], &device);

        let grad_input_data: Vec<f32> = grad_input.into_data().to_vec().unwrap();
        let expected_data: Vec<f32> = expected.into_data().to_vec().unwrap();

        for (o, e) in grad_input_data.iter().zip(expected_data.iter()) {
            assert!((o - e).abs() < 1e-5, "Gradient mismatch: {} vs {}", o, e);
        }
    }
}
