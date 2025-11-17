/// Format conversion logic
///
/// All conversions route through a canonical path:
/// - CSR is the "hub" format
/// - X → CSR → Y for most conversions
/// - Special cases: Mask ↔ CSR (direct), CSR ↔ COO (direct)
///
/// This simplifies implementation: only need 2N converters (to/from CSR for N formats),
/// not N² converters.

use burn_core::tensor::{backend::Backend, Bool, Int, Tensor, TensorData, Shape};

use crate::core::{SparseTensor, SparseTensorData, SparseFormat, SparseResult, SparseError};

/// Convert sparse tensor to different format
pub fn convert_format<B: Backend>(
    tensor: &SparseTensor<B>,
    target: SparseFormat,
) -> SparseResult<SparseTensor<B>> {
    if tensor.format() == target {
        return Ok(tensor.clone());
    }

    // Route through CSR hub
    let csr = if tensor.format() == SparseFormat::CSR {
        tensor.clone()
    } else {
        to_csr(tensor)?
    };

    if target == SparseFormat::CSR {
        return Ok(csr);
    }

    // CSR → target
    from_csr(&csr, target)
}

/// Convert to dense tensor
pub fn to_dense<B: Backend>(tensor: &SparseTensor<B>) -> Tensor<B, 2> {
    match tensor.data() {
        SparseTensorData::Mask { values, .. } => {
            // Dense values are already stored
            values.clone()
        }
        SparseTensorData::CSR { values, col_indices, row_pointers } => {
            csr_to_dense(values, col_indices, row_pointers, tensor.shape(), &tensor.device())
        }
        SparseTensorData::CSC { values, row_indices, col_pointers } => {
            csc_to_dense(values, row_indices, col_pointers, tensor.shape(), &tensor.device())
        }
        SparseTensorData::COO { values, row_indices, col_indices } => {
            coo_to_dense(values, row_indices, col_indices, tensor.shape(), &tensor.device())
        }
        SparseTensorData::BlockCSR { .. } => {
            // Convert to CSR first, then to dense
            let csr = to_csr(tensor).unwrap();
            to_dense(&csr)
        }
        SparseTensorData::NInM { .. } => {
            // Convert to CSR first, then to dense
            let csr = to_csr(tensor).unwrap();
            to_dense(&csr)
        }
    }
}

// ============================================================================
// Conversion to CSR (hub)
// ============================================================================

fn to_csr<B: Backend>(tensor: &SparseTensor<B>) -> SparseResult<SparseTensor<B>> {
    match tensor.data() {
        SparseTensorData::Mask { mask, values } => {
            mask_to_csr(mask, values, tensor.shape(), &tensor.device())
        }
        SparseTensorData::CSR { .. } => {
            Ok(tensor.clone())
        }
        SparseTensorData::CSC { values, row_indices, col_pointers } => {
            csc_to_csr(values, row_indices, col_pointers, tensor.shape(), &tensor.device())
        }
        SparseTensorData::COO { values, row_indices, col_indices } => {
            coo_to_csr(values, row_indices, col_indices, tensor.shape(), &tensor.device())
        }
        SparseTensorData::BlockCSR { .. } => {
            // TODO: Implement
            Err(SparseError::ConversionFailed {
                from: tensor.format(),
                to: SparseFormat::CSR,
                reason: "BlockCSR → CSR not yet implemented".to_string(),
            })
        }
        SparseTensorData::NInM { .. } => {
            // TODO: Implement
            Err(SparseError::ConversionFailed {
                from: tensor.format(),
                to: SparseFormat::CSR,
                reason: "N:M → CSR not yet implemented".to_string(),
            })
        }
    }
}

// ============================================================================
// Conversion from CSR
// ============================================================================

fn from_csr<B: Backend>(csr: &SparseTensor<B>, target: SparseFormat) -> SparseResult<SparseTensor<B>> {
    match target {
        SparseFormat::Mask => {
            let dense = to_dense(csr);
            Ok(SparseTensor::from_dense(&dense, SparseFormat::Mask, 0.0))
        }
        SparseFormat::CSR => {
            Ok(csr.clone())
        }
        SparseFormat::CSC => {
            // TODO: Implement CSR → CSC
            Err(SparseError::ConversionFailed {
                from: SparseFormat::CSR,
                to: target,
                reason: "CSR → CSC not yet implemented".to_string(),
            })
        }
        SparseFormat::COO => {
            // TODO: Implement CSR → COO
            Err(SparseError::ConversionFailed {
                from: SparseFormat::CSR,
                to: target,
                reason: "CSR → COO not yet implemented".to_string(),
            })
        }
        SparseFormat::BlockCSR { .. } => {
            Err(SparseError::ConversionFailed {
                from: SparseFormat::CSR,
                to: target,
                reason: "CSR → BlockCSR not yet implemented".to_string(),
            })
        }
        SparseFormat::NInM { .. } => {
            Err(SparseError::ConversionFailed {
                from: SparseFormat::CSR,
                to: target,
                reason: "CSR → N:M not yet implemented".to_string(),
            })
        }
    }
}

// ============================================================================
// Format-Specific Conversions (Implementations)
// ============================================================================

fn mask_to_csr<B: Backend>(
    mask: &Tensor<B, 2, Bool>,
    values: &Tensor<B, 2>,
    shape: [usize; 2],
    device: &B::Device,
) -> SparseResult<SparseTensor<B>> {
    // TODO: Implement Mask → CSR conversion
    // For now, return empty CSR
    let [n_rows, _n_cols] = shape;

    Ok(SparseTensor::from_dense(
        values,
        SparseFormat::CSR,
        0.0,
    ))
}

fn csr_to_dense<B: Backend>(
    values: &Tensor<B, 1>,
    col_indices: &Tensor<B, 1, Int>,
    row_pointers: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    // TODO: Implement CSR → dense conversion
    // For now, return zeros
    Tensor::zeros(shape, device)
}

fn csc_to_dense<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_pointers: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    // TODO: Implement CSC → dense
    Tensor::zeros(shape, device)
}

fn coo_to_dense<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_indices: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    // TODO: Implement COO → dense
    Tensor::zeros(shape, device)
}

fn csc_to_csr<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_pointers: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    device: &B::Device,
) -> SparseResult<SparseTensor<B>> {
    // TODO: Implement CSC → CSR
    Err(SparseError::ConversionFailed {
        from: SparseFormat::CSC,
        to: SparseFormat::CSR,
        reason: "Not yet implemented".to_string(),
    })
}

fn coo_to_csr<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_indices: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    device: &B::Device,
) -> SparseResult<SparseTensor<B>> {
    // TODO: Implement COO → CSR
    Err(SparseError::ConversionFailed {
        from: SparseFormat::COO,
        to: SparseFormat::CSR,
        reason: "Not yet implemented".to_string(),
    })
}
