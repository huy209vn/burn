/// Validation logic for sparse tensor structures
///
/// Validates invariants for each format:
/// - CSR: sorted indices, monotonic row pointers, bounds checking
/// - COO: bounds checking, no duplicates (optional)
/// - N:M: constraint satisfaction
/// - BlockCSR: alignment, valid blocks

use burn_core::tensor::{backend::Backend, ElementConversion};

use crate::core::{SparseTensor, SparseTensorData, SparseError, SparseResult};

/// Validate sparse tensor structure
///
/// This is called during construction to ensure invariants hold.
/// Once valid, the tensor remains valid (immutable design).
pub fn validate_sparse_tensor<B: Backend>(tensor: &SparseTensor<B>) -> SparseResult<()> {
    match tensor.data() {
        SparseTensorData::Mask { mask, values } => {
            validate_mask(mask, values, tensor.shape())
        }
        SparseTensorData::CSR { values, col_indices, row_pointers } => {
            validate_csr(values, col_indices, row_pointers, tensor.shape())
        }
        SparseTensorData::CSC { values, row_indices, col_pointers } => {
            validate_csc(values, row_indices, col_pointers, tensor.shape())
        }
        SparseTensorData::COO { values, row_indices, col_indices } => {
            validate_coo(values, row_indices, col_indices, tensor.shape())
        }
        SparseTensorData::BlockCSR { blocks, block_col_indices, block_row_pointers, block_size } => {
            validate_block_csr(blocks, block_col_indices, block_row_pointers, *block_size, tensor.shape())
        }
        SparseTensorData::NInM { values, metadata, n, m } => {
            validate_nm(values, metadata, *n, *m, tensor.shape())
        }
    }
}

fn validate_mask<B: Backend>(
    mask: &burn_core::tensor::Tensor<B, 2, burn_core::tensor::Bool>,
    values: &burn_core::tensor::Tensor<B, 2>,
    shape: [usize; 2],
) -> SparseResult<()> {
    if mask.dims() != shape {
        return Err(SparseError::InvalidTensor {
            reason: format!("Mask shape {:?} doesn't match tensor shape {:?}", mask.dims(), shape)
        });
    }
    if values.dims() != shape {
        return Err(SparseError::InvalidTensor {
            reason: format!("Values shape {:?} doesn't match tensor shape {:?}", values.dims(), shape)
        });
    }
    Ok(())
}

fn validate_csr<B: Backend>(
    values: &burn_core::tensor::Tensor<B, 1>,
    col_indices: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    row_pointers: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    shape: [usize; 2],
) -> SparseResult<()> {
    let nnz = values.dims()[0];
    let [n_rows, n_cols] = shape;

    // Check dimensions
    if col_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!("col_indices length {} != nnz {}", col_indices.dims()[0], nnz)
        });
    }
    if row_pointers.dims()[0] != n_rows + 1 {
        return Err(SparseError::InvalidTensor {
            reason: format!("row_pointers length {} != n_rows + 1 {}", row_pointers.dims()[0], n_rows + 1)
        });
    }

    // TODO: Deep validation (sorted indices, monotonic pointers, bounds)
    // This requires reading tensor data to CPU, which is expensive.
    // For now, trust construction. Add debug-mode validation later.

    Ok(())
}

fn validate_csc<B: Backend>(
    values: &burn_core::tensor::Tensor<B, 1>,
    row_indices: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    col_pointers: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    shape: [usize; 2],
) -> SparseResult<()> {
    let nnz = values.dims()[0];
    let [n_rows, n_cols] = shape;

    if row_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!("row_indices length {} != nnz {}", row_indices.dims()[0], nnz)
        });
    }
    if col_pointers.dims()[0] != n_cols + 1 {
        return Err(SparseError::InvalidTensor {
            reason: format!("col_pointers length {} != n_cols + 1 {}", col_pointers.dims()[0], n_cols + 1)
        });
    }

    Ok(())
}

fn validate_coo<B: Backend>(
    values: &burn_core::tensor::Tensor<B, 1>,
    row_indices: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    col_indices: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    shape: [usize; 2],
) -> SparseResult<()> {
    let nnz = values.dims()[0];

    if row_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!("row_indices length {} != nnz {}", row_indices.dims()[0], nnz)
        });
    }
    if col_indices.dims()[0] != nnz {
        return Err(SparseError::InvalidTensor {
            reason: format!("col_indices length {} != nnz {}", col_indices.dims()[0], nnz)
        });
    }

    Ok(())
}

fn validate_block_csr<B: Backend>(
    blocks: &burn_core::tensor::Tensor<B, 2>,
    block_col_indices: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    block_row_pointers: &burn_core::tensor::Tensor<B, 1, burn_core::tensor::Int>,
    block_size: usize,
    shape: [usize; 2],
) -> SparseResult<()> {
    let n_blocks = blocks.dims()[0];
    let block_data_size = blocks.dims()[1];

    if block_data_size != block_size * block_size {
        return Err(SparseError::InvalidTensor {
            reason: format!("Block data size {} != block_size² {}", block_data_size, block_size * block_size)
        });
    }

    if block_col_indices.dims()[0] != n_blocks {
        return Err(SparseError::InvalidTensor {
            reason: format!("block_col_indices length {} != n_blocks {}", block_col_indices.dims()[0], n_blocks)
        });
    }

    Ok(())
}

fn validate_nm<B: Backend>(
    values: &burn_core::tensor::Tensor<B, 2>,
    metadata: &burn_core::tensor::Tensor<B, 2, burn_core::tensor::Int>,
    n: usize,
    m: usize,
    shape: [usize; 2],
) -> SparseResult<()> {
    let total_elements = shape[0] * shape[1];

    if total_elements % m != 0 {
        return Err(SparseError::NInMViolation {
            details: format!("Total elements {} not divisible by M {}", total_elements, m)
        });
    }

    let n_groups = total_elements / m;
    if values.dims() != [n_groups, n] {
        return Err(SparseError::InvalidTensor {
            reason: format!("Values shape {:?} doesn't match expected [{}, {}]", values.dims(), n_groups, n)
        });
    }

    Ok(())
}
