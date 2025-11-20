//! Test autodiff gradient flow through sparse operations

use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn_sparse::prelude::*;

type TestBackend = NdArray<f32>;
type TestAutodiffBackend = Autodiff<TestBackend>;

#[test]
fn test_sparse_linear_gradient_flow() {
    let device = Default::default();

    // Create a simple sparse weight matrix [2, 3] with pattern:
    // [[1.0, 0.0, 2.0],
    //  [0.0, 3.0, 0.0]]
    let weight_dense_inner: Tensor<TestBackend, 2> =
        Tensor::from_data([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], &device);

    // Create mask (non-zeros)
    let mask_data = weight_dense_inner.clone().not_equal_elem(0.0);
    let mask = SparseMask::from_tensor(mask_data);

    // Convert to sparse tensor (on inner backend)
    let weight_sparse = SparseTensor::from_mask(&mask, &weight_dense_inner)
        .expect("Failed to create sparse tensor")
        .to_format(SparseFormat::CSR)
        .expect("Failed to convert to CSR");

    // Enable gradients
    let weight_param = SparseParam::from_sparse(weight_sparse);

    // Input [3, 2] batch
    let input: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).require_grad();

    // Forward pass through sparse linear
    let output = {
        let weight = weight_param.val();
        SparseDispatch::spmm(&weight, &input, &SparseConfig::default())
            .expect("SpMM failed")
    };

    // Expected output: [2, 2]
    // [[1*1 + 0*3 + 2*5, 1*2 + 0*4 + 2*6], = [[11, 14],
    //  [0*1 + 3*3 + 0*5, 0*2 + 3*4 + 0*6]]    [9, 12]]
    println!("Output shape: {:?}", output.shape());
    println!("Output: {:?}", output.to_data());

    // Compute loss (sum of outputs)
    let loss = output.sum();
    println!("Loss: {:?}", loss.to_data());

    // Backward pass
    let grads = loss.backward();

    // Check if input has gradients
    let input_grad = input.grad(&grads);
    match input_grad {
        Some(grad) => {
            println!("✓ Input gradient exists!");
            println!("Input grad shape: {:?}", grad.shape());
            println!("Input grad: {:?}", grad.to_data());
        }
        None => {
            panic!("✗ Input gradient is None - autodiff not working!");
        }
    }

    // TODO: Once we implement backward pass, check weight gradients here
    println!("\nNote: Weight gradient extraction not yet implemented");
    println!("This test verifies structure is ready for autodiff");

    // For now, just verify basic properties
    println!("Weight format: {:?}", weight_param.val().format());
    println!("Weight nnz: {}", weight_param.val().nnz());
    println!("Weight sparsity: {:.2}%", weight_param.val().sparsity() * 100.0);
}

#[test]
fn test_dense_linear_gradient_baseline() {
    // Baseline: verify dense linear gradients work
    let device = Default::default();

    let weight: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).require_grad();

    let input: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).require_grad();

    // Y = W @ X (clone to keep references for grad extraction)
    let output = weight.clone().matmul(input.clone());
    let loss = output.sum();

    let grads = loss.backward();

    // Both should have gradients
    assert!(
        input.grad(&grads).is_some(),
        "Dense input should have gradients"
    );
    assert!(
        weight.grad(&grads).is_some(),
        "Dense weight should have gradients"
    );

    println!("✓ Dense baseline: gradients flow correctly");
}
