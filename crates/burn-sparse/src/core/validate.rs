//! Validation logic for sparse tensor invariants
//!
//! This module provides validation functions for all sparse formats.
//! Validation is performed once at construction time to ensure
//! that sparse tensors are always in a valid state.

use burn_core::tensor::{backend::Backend, Bool, Int, Tensor};

use crate::core::{SparseError, SparseResult, SparseTensor, SparseTensorData};

/// Validate entire sparse tensor (dispatch to format-specific validation)
pub fn validate_sparse_tensor<B: Backend>(tensor: &SparseTensor<B>) -> SparseResult<()> {
    match tensor.data() {
        SparseTensorData::Mask { .. } => {
            // Mask format has no invariants beyond shape
            Ok(())
        }
        SparseTensorData::CSR {
            values,
            col_indices,
            row_pointers,
        } => validate_csr(values, col_indices, row_pointers, tensor.shape()),
        SparseTensorData::CSC {
            values,
            row_indices,
            col_pointers,
        } => validate_csc(values, row_indices, col_pointers, tensor.shape()),
        SparseTensorData::COO {
            values,
            row_indices,
            col_indices,
        } => validate_coo(values, row_indices, col_indices, tensor.shape(), false),
        SparseTensorData::BlockCSR { .. } => {
            // TODO: Implement BlockCSR validation
            Ok(())
        }
        SparseTensorData::NInM { .. } => {
            // TODO: Implement N:M validation
            Ok(())
        }
    }
}

/// Validate CSR format invariants
///
/// Checks:
/// - Row pointers are monotonically increasing
/// - Row pointers[0] == 0
/// - Row pointers[n_rows] == nnz
/// - Column indices are sorted within each row
/// - Column indices are within bounds [0, n_cols)
/// - No duplicate (row, col) pairs
pub fn validate_csr<B: Backend>(
    values: &Tensor<B, 1>,
    col_indices: &Tensor<B, 1, Int>,
    row_pointers: &Tensor<B, 1, Int>,
    shape: [usize; 2],
) -> SparseResult<()> {
    let [n_rows, n_cols] = shape;
    let nnz = values.dims()[0];

    // Check dimensions match
    if col_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSR col_indices length {} doesn't match values length {}",
                col_indices.dims()[0],
                nnz
            ),
        });
    }

    if row_pointers.dims()[0] != n_rows + 1 {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSR row_pointers length {} doesn't match n_rows + 1 = {}",
                row_pointers.dims()[0],
                n_rows + 1
            ),
        });
    }

    // Convert to CPU for validation
    let col_data: Vec<i64> = col_indices.to_data().convert::<i64>().to_vec().unwrap();
    let row_data: Vec<i64> = row_pointers.to_data().convert::<i64>().to_vec().unwrap();

    // Check row_pointers[0] == 0
    if row_data[0] != 0 {
        return Err(SparseError::InvalidTensor {
            reason: format!("CSR row_pointers[0] must be 0, got {}", row_data[0]),
        });
    }

    // Check row_pointers[n_rows] == nnz
    if row_data[n_rows] != nnz as i64 {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSR row_pointers[{}] must be {}, got {}",
                n_rows, nnz, row_data[n_rows]
            ),
        });
    }

    // Check monotonicity and sorted indices within rows
    for i in 0..n_rows {
        let row_start = row_data[i] as usize;
        let row_end = row_data[i + 1] as usize;

        // Check monotonic
        if row_end < row_start {
            return Err(SparseError::InvalidTensor {
                reason: format!(
                    "CSR row_pointers not monotonic: row_pointers[{}] = {}, row_pointers[{}] = {}",
                    i, row_start, i + 1, row_end
                ),
            });
        }

        // Check sorted within row and bounds
        let mut prev_col = -1i64;
        for j in row_start..row_end {
            let col = col_data[j];

            // Check bounds
            if col < 0 || col >= n_cols as i64 {
                return Err(SparseError::InvalidTensor {
                    reason: format!(
                        "CSR col_indices[{}] = {} out of bounds [0, {})",
                        j, col, n_cols
                    ),
                });
            }

            // Check sorted (also catches duplicates)
            if col <= prev_col {
                return Err(SparseError::InvalidTensor {
                    reason: format!(
                        "CSR col_indices not sorted in row {}: col_indices[{}] = {} <= previous {}",
                        i, j, col, prev_col
                    ),
                });
            }

            prev_col = col;
        }
    }

    Ok(())
}

/// Validate COO format invariants
///
/// Checks:
/// - All three tensors have same length
/// - Row and column indices are within bounds
/// - No duplicate (row, col) pairs (optional, expensive)
pub fn validate_coo<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_indices: &Tensor<B, 1, Int>,
    shape: [usize; 2],
    _check_duplicates: bool, // TODO: implement duplicate check
) -> SparseResult<()> {
    let [n_rows, n_cols] = shape;
    let nnz = values.dims()[0];

    // Check dimensions match
    if row_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "COO row_indices length {} doesn't match values length {}",
                row_indices.dims()[0],
                nnz
            ),
        });
    }

    if col_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "COO col_indices length {} doesn't match values length {}",
                col_indices.dims()[0],
                nnz
            ),
        });
    }

    // Check bounds
    let row_data: Vec<i64> = row_indices.to_data().convert::<i64>().to_vec().unwrap();
    let col_data: Vec<i64> = col_indices.to_data().convert::<i64>().to_vec().unwrap();

    for i in 0..nnz {
        let row = row_data[i];
        let col = col_data[i];

        if row < 0 || row >= n_rows as i64 {
            return Err(SparseError::InvalidTensor {
                reason: format!(
                    "COO row_indices[{}] = {} out of bounds [0, {})",
                    i, row, n_rows
                ),
            });
        }

        if col < 0 || col >= n_cols as i64 {
            return Err(SparseError::InvalidTensor {
                reason: format!(
                    "COO col_indices[{}] = {} out of bounds [0, {})",
                    i, col, n_cols
                ),
            });
        }
    }

    // TODO: Check for duplicates if requested (expensive)

    Ok(())
}

/// Validate CSC format invariants (similar to CSR but for columns)
pub fn validate_csc<B: Backend>(
    values: &Tensor<B, 1>,
    row_indices: &Tensor<B, 1, Int>,
    col_pointers: &Tensor<B, 1, Int>,
    shape: [usize; 2],
) -> SparseResult<()> {
    let [n_rows, n_cols] = shape;
    let nnz = values.dims()[0];

    // Check dimensions match
    if row_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSC row_indices length {} doesn't match values length {}",
                row_indices.dims()[0],
                nnz
            ),
        });
    }

    if col_pointers.dims()[0] != n_cols + 1 {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSC col_pointers length {} doesn't match n_cols + 1 = {}",
                col_pointers.dims()[0],
                n_cols + 1
            ),
        });
    }

    // Convert to CPU for validation
    let row_data: Vec<i64> = row_indices.to_data().convert::<i64>().to_vec().unwrap();
    let col_data: Vec<i64> = col_pointers.to_data().convert::<i64>().to_vec().unwrap();

    // Check col_pointers[0] == 0
    if col_data[0] != 0 {
        return Err(SparseError::InvalidTensor {
            reason: format!("CSC col_pointers[0] must be 0, got {}", col_data[0]),
        });
    }

    // Check col_pointers[n_cols] == nnz
    if col_data[n_cols] != nnz as i64 {
        return Err(SparseError::InvalidTensor {
            reason: format!(
                "CSC col_pointers[{}] must be {}, got {}",
                n_cols, nnz, col_data[n_cols]
            ),
        });
    }

    // Check monotonicity and sorted indices within columns
    for i in 0..n_cols {
        let col_start = col_data[i] as usize;
        let col_end = col_data[i + 1] as usize;

        // Check monotonic
        if col_end < col_start {
            return Err(SparseError::InvalidTensor {
                reason: format!(
                    "CSC col_pointers not monotonic: col_pointers[{}] = {}, col_pointers[{}] = {}",
                    i, col_start, i + 1, col_end
                ),
            });
        }

        // Check sorted within column and bounds
        let mut prev_row = -1i64;
        for j in col_start..col_end {
            let row = row_data[j];

            // Check bounds
            if row < 0 || row >= n_rows as i64 {
                return Err(SparseError::InvalidTensor {
                    reason: format!(
                        "CSC row_indices[{}] = {} out of bounds [0, {})",
                        j, row, n_rows
                    ),
                });
            }

            // Check sorted (also catches duplicates)
            if row <= prev_row {
                return Err(SparseError::InvalidTensor {
                    reason: format!(
                        "CSC row_indices not sorted in column {}: row_indices[{}] = {} <= previous {}",
                        i, j, row, prev_row
                    ),
                });
            }

            prev_row = row;
        }
    }

    Ok(())
}

/// Validate N:M structured sparsity pattern
///
/// Checks that every M consecutive positions have exactly N non-zeros.
pub fn validate_nm_pattern<B: Backend>(
    mask: &Tensor<B, 2, Bool>,
    n: usize,
    m: usize,
    dim: usize, // 0 for row-wise, 1 for column-wise
) -> SparseResult<()> {
    let shape = mask.dims();
    let mask_data: Vec<i64> = mask.clone().int().to_data().convert::<i64>().to_vec().unwrap();

    let (outer_dim, inner_dim) = if dim == 0 {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };

    // Check alignment
    if inner_dim % m != 0 {
        return Err(SparseError::NInMViolation {
            details: format!(
                "Dimension {} not aligned to M={}: {} % {} != 0",
                inner_dim, m, inner_dim, m
            ),
        });
    }

    // Check N:M pattern
    for i in 0..outer_dim {
        for block_start in (0..inner_dim).step_by(m) {
            let mut count = 0;
            for j in block_start..(block_start + m).min(inner_dim) {
                let idx = if dim == 0 {
                    i * inner_dim + j
                } else {
                    j * outer_dim + i
                };

                if mask_data[idx] != 0 {
                    count += 1;
                }
            }

            if count != n {
                return Err(SparseError::NInMViolation {
                    details: format!(
                        "{}:M violation at {}={}, block starting at {}: found {} non-zeros, expected {}",
                        n, if dim == 0 { "row" } else { "col" }, i, block_start, count, n
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{Shape, Tensor, TensorData};

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_validate_csr_valid() {
        // Valid CSR: [[1, 0], [0, 2]]
        let values = Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 2.0]), &Default::default());
        let col_indices = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0i64, 1i64]), &Default::default());
        let row_pointers = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0i64, 1i64, 2i64]), &Default::default());

        assert!(validate_csr(&values, &col_indices, &row_pointers, [2, 2]).is_ok());
    }

    #[test]
    fn test_validate_csr_unsorted() {
        // Invalid: col_indices not sorted in row 0
        let values = Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 2.0]), &Default::default());
        let col_indices = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([1i64, 0i64]), &Default::default());
        let row_pointers = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0i64, 2i64]), &Default::default());

        assert!(validate_csr(&values, &col_indices, &row_pointers, [1, 2]).is_err());
    }

    #[test]
    fn test_validate_csr_out_of_bounds() {
        // Invalid: col_indices[0] = 5 > n_cols
        let values = Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0]), &Default::default());
        let col_indices = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([5i64]), &Default::default());
        let row_pointers = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0i64, 1i64]), &Default::default());

        assert!(validate_csr(&values, &col_indices, &row_pointers, [1, 2]).is_err());
    }
}
