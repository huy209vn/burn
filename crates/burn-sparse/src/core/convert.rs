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

pub(crate) fn mask_to_csr<B: Backend>(
    mask: &Tensor<B, 2, Bool>,
    values: &Tensor<B, 2>,
    shape: [usize; 2],
    device: &B::Device,
) -> SparseResult<SparseTensor<B>> {
    let [n_rows, n_cols] = shape;

    // Move data to CPU for processing
    let mask_data: Vec<bool> = mask.clone().into_data().to_vec().unwrap();
    let values_data: Vec<f32> = values.clone().into_data().to_vec().unwrap();

    // Build CSR format on CPU
    let mut row_pointers: Vec<i64> = Vec::with_capacity(n_rows + 1);
    let mut col_indices: Vec<i64> = Vec::new();
    let mut csr_values: Vec<f32> = Vec::new();

    row_pointers.push(0);

    for row in 0..n_rows {
        let row_start = row * n_cols;
        let row_end = row_start + n_cols;

        for col in 0..n_cols {
            let idx = row_start + col;
            if mask_data[idx] {
                // This position is active (not pruned)
                col_indices.push(col as i64);
                csr_values.push(values_data[idx]);
            }
        }

        // row_pointers[i+1] = cumulative count of non-zeros up to row i
        row_pointers.push(col_indices.len() as i64);
    }

    let nnz = csr_values.len();

    // Create Burn tensors from CSR data
    let csr_values_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(csr_values, Shape::new([nnz])),
        device,
    );

    let col_indices_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::new(col_indices, Shape::new([nnz])),
        device,
    );

    let row_pointers_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::new(row_pointers, Shape::new([n_rows + 1])),
        device,
    );

    // Build SparseTensor with CSR format
    Ok(SparseTensor::from_csr(
        csr_values_tensor,
        col_indices_tensor,
        row_pointers_tensor,
        shape,
        device.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend as TB;

    #[test]
    fn test_mask_to_csr_conversion() {
        let device = Default::default();

        // Create a simple 3x3 sparse matrix
        // [1.0  0.0  2.0]
        // [0.0  3.0  0.0]
        // [4.0  0.0  5.0]
        let dense = Tensor::<TB, 2>::from_data(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
                [4.0, 0.0, 5.0],
            ],
            &device,
        );

        // Create mask (true for non-zero)
        let mask = Tensor::<TB, 2, Bool>::from_data(
            [
                [true, false, true],
                [false, true, false],
                [true, false, true],
            ],
            &device,
        );

        // Convert to CSR
        let csr = mask_to_csr(&mask, &dense, [3, 3], &device).unwrap();

        // Verify format
        assert_eq!(csr.format(), SparseFormat::CSR);
        assert_eq!(csr.shape(), [3, 3]);
        assert_eq!(csr.nnz(), 5);

        // Extract CSR data and verify structure
        match csr.data() {
            SparseTensorData::CSR { values, col_indices, row_pointers } => {
                // Check values
                let vals: Vec<f32> = values.clone().into_data().to_vec().unwrap();
                assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

                // Check col_indices
                let cols: Vec<i64> = col_indices.clone().into_data().to_vec().unwrap();
                assert_eq!(cols, vec![0, 2, 1, 0, 2]);

                // Check row_pointers
                let ptrs: Vec<i64> = row_pointers.clone().into_data().to_vec().unwrap();
                assert_eq!(ptrs, vec![0, 2, 3, 5]);
            }
            _ => panic!("Expected CSR format"),
        }
    }

    #[test]
    fn test_mask_to_csr_empty_rows() {
        let device = Default::default();

        // Matrix with an empty row
        // [1.0  2.0]
        // [0.0  0.0]
        // [3.0  4.0]
        let dense = Tensor::<TB, 2>::from_data(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [3.0, 4.0],
            ],
            &device,
        );

        let mask = Tensor::<TB, 2, Bool>::from_data(
            [
                [true, true],
                [false, false],
                [true, true],
            ],
            &device,
        );

        let csr = mask_to_csr(&mask, &dense, [3, 2], &device).unwrap();

        assert_eq!(csr.nnz(), 4);

        match csr.data() {
            SparseTensorData::CSR { row_pointers, .. } => {
                let ptrs: Vec<i64> = row_pointers.clone().into_data().to_vec().unwrap();
                // Empty row 1 should have same pointer value
                assert_eq!(ptrs, vec![0, 2, 2, 4]);
            }
            _ => panic!("Expected CSR format"),
        }
    }
}
