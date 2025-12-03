//! JEPA-specific loss functions.
//!
//! This module defines the L2-normalized Mean Squared Error (MSE) loss
//! used in the Joint-Embedding Predictive Architecture, with optional
//! variance regularization to prevent representation collapse.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// # Computes the JEPA L2 Loss with Variance Regularization.
///
/// This loss function calculates the Mean Squared Error (MSE) between
/// L2-normalized predicted and target latent representations, with optional
/// variance regularization to prevent representation collapse.
///
/// Variance regularization encourages the model to use all dimensions of the
/// embedding space by penalizing low variance along any dimension.
///
/// # Arguments
///
/// * `predictions`: Output from the predictor, shape `[B, N_tgt, D]`.
/// * `targets`: Ground truth target representations from the teacher encoder, shape `[B, N_tgt, D]`.
///              These targets are assumed to have `stop_gradient` applied to them.
/// * `var_reg_weight`: Weight for variance regularization loss (0.0 = disabled). Typical: 1.0-10.0.
///
/// # Returns
///
/// * A scalar tensor representing the combined loss over the batch.
pub fn jepa_loss<B: Backend>(
    predictions: Tensor<B, 3>,
    targets: Tensor<B, 3>,
    var_reg_weight: f32,
) -> Tensor<B, 1> {
    // 1. L2 normalize both `predictions` and `targets` along the last dimension (embedding dimension).
    let pred_norm = l2_normalize(predictions.clone());
    let tgt_norm = l2_normalize(targets.clone());

    // 2. Compute the squared difference between the normalized predictions and targets.
    let diff = pred_norm - tgt_norm;
    let sq_diff = diff.powf_scalar(2.0);

    // 3. Compute the mean MSE loss over all elements.
    let mse_loss = sq_diff.mean();

    // 4. Add variance regularization if enabled
    let total_loss = if var_reg_weight > 0.0 {
        // Compute variance regularization on predictions
        let var_loss = variance_loss(predictions);

        // Combine MSE and variance regularization
        mse_loss + var_loss.mul_scalar(var_reg_weight)
    } else {
        mse_loss
    };

    // 5. Return the scalar loss as a 1D tensor.
    total_loss.reshape([1])
}

/// # Variance Regularization Loss.
///
/// Computes a variance regularization loss that prevents representation collapse
/// by encouraging the model to use all dimensions of the embedding space.
///
/// The loss penalizes low variance along any embedding dimension by computing:
/// `loss = -log(std(x, dim=0) + eps)` averaged over all dimensions.
///
/// This encourages each dimension to have non-zero variance across the batch.
///
/// # Arguments
///
/// * `tensor`: Input tensor with shape `[B, N, D]`.
///
/// # Returns
///
/// * A scalar tensor representing the variance regularization loss.
fn variance_loss<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 1> {
    let [batch_size, n_patches, embed_dim] = tensor.dims();

    // Reshape to [B*N, D] to compute variance across all samples and patches
    let flattened = tensor.reshape([batch_size * n_patches, embed_dim]);

    // Compute mean along dim 0: [D]
    let mean = flattened.clone().mean_dim(0);

    // Compute variance along dim 0: [D]
    // Broadcasting works automatically - mean [D] will broadcast to [B*N, D]
    let diff = flattened - mean.clone();
    let variance = diff.powf_scalar(2.0).mean_dim(0);

    // Compute standard deviation with epsilon for numerical stability: [D]
    let std = variance.sqrt().clamp_min(1e-4);

    // Loss = -log(std + eps) encourages high variance
    // We use mean across dimensions to get a scalar
    let loss = std.log().neg().mean();

    loss
}

/// # L2 Normalizes a Tensor.
///
/// Normalizes a tensor along its last dimension to have a unit L2 norm.
/// A small epsilon is added to the norm to prevent division by zero.
///
/// # Arguments
///
/// * `tensor`: The input tensor to normalize, with `D` dimensions.
///
/// # Returns
///
/// * The L2-normalized tensor, with the same shape as the input.
pub fn l2_normalize<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    // 1. Compute the L2 norm (Euclidean norm) along the last dimension.
    let norm = tensor.clone().powf_scalar(2.0).sum_dim(D - 1).sqrt();

    // 2. Add a small epsilon for numerical stability.
    let norm = norm.clamp_min(1e-6);

    // 3. Divide the tensor by its norm. We need to unsqueeze the norm to match
    //    the dimensions of the tensor for broadcasting.
    let norm_unsqueezed = norm.unsqueeze();
    let normalized_tensor = tensor.div(norm_unsqueezed);

    // 4. Return the normalized tensor.
    normalized_tensor
}