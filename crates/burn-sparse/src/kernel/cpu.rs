//! CPU reference implementation of sparse kernels
//!
//! This module provides correct but not necessarily optimized
//! implementations of sparse operations for CPU backends.

use burn_core::tensor::{backend::Backend, Int, Tensor, TensorData};

use crate::core::{SparseError, SparseFormat, SparseResult, SparseTensor, SparseTensorData};
use crate::kernel::api::{KernelSupport, SparseKernel};

/// CPU sparse kernel implementation
pub struct CpuKernel;

impl CpuKernel {
    /// CSR Sparse-Dense Matrix Multiply
    fn spmm_csr<B: Backend>(
        values: &Tensor<B, 1>,
        col_indices: &Tensor<B, 1, Int>,
        row_pointers: &Tensor<B, 1, Int>,
        b: &Tensor<B, 2>,
        shape: [usize; 2],
    ) -> SparseResult<Tensor<B, 2>> {
        let [n_rows, _n_cols] = shape;
        let k = b.dims()[1];

        // Convert to CPU data for processing
        let val_data: Vec<f32> = values.to_data().to_vec().unwrap();
        let col_data: Vec<i64> = col_indices.to_data().convert::<i64>().to_vec().unwrap();
        let row_data: Vec<i64> = row_pointers.to_data().convert::<i64>().to_vec().unwrap();
        let b_data: Vec<f32> = b.to_data().to_vec().unwrap();
        let b_cols = b.dims()[1];

        let mut result = vec![0.0; n_rows * k];

        // Naive CSR SpMM: Y[i, :] = Σ_j A[i,j] * B[j, :]
        for i in 0..n_rows {
            let row_start = row_data[i] as usize;
            let row_end = row_data[i + 1] as usize;

            for j_idx in row_start..row_end {
                let col = col_data[j_idx] as usize;
                let val = val_data[j_idx];

                // Y[i, :] += val * B[col, :]
                for p in 0..k {
                    result[i * k + p] += val * b_data[col * b_cols + p];
                }
            }
        }

        Ok(Tensor::from_data(
            TensorData::new(result, [n_rows, k]),
            &b.device(),
        ))
    }

    /// COO Sparse-Dense Matrix Multiply
    fn spmm_coo<B: Backend>(
        values: &Tensor<B, 1>,
        row_indices: &Tensor<B, 1, Int>,
        col_indices: &Tensor<B, 1, Int>,
        b: &Tensor<B, 2>,
        shape: [usize; 2],
    ) -> SparseResult<Tensor<B, 2>> {
        let [n_rows, _n_cols] = shape;
        let k = b.dims()[1];
        let nnz = values.dims()[0];

        // Convert to CPU data
        let val_data: Vec<f32> = values.to_data().to_vec().unwrap();
        let row_data: Vec<i64> = row_indices.to_data().convert::<i64>().to_vec().unwrap();
        let col_data: Vec<i64> = col_indices.to_data().convert::<i64>().to_vec().unwrap();
        let b_data: Vec<f32> = b.to_data().to_vec().unwrap();
        let b_cols = b.dims()[1];

        let mut result = vec![0.0; n_rows * k];

        // Naive COO SpMM: accumulate each (i, j, val) contribution
        for idx in 0..nnz {
            let row = row_data[idx] as usize;
            let col = col_data[idx] as usize;
            let val = val_data[idx];

            // Y[row, :] += val * B[col, :]
            for p in 0..k {
                result[row * k + p] += val * b_data[col * b_cols + p];
            }
        }

        Ok(Tensor::from_data(
            TensorData::new(result, [n_rows, k]),
            &b.device(),
        ))
    }
}

impl<B: Backend> SparseKernel<B> for CpuKernel {
    fn spmm(&self, a: &SparseTensor<B>, b: &Tensor<B, 2>) -> SparseResult<Tensor<B, 2>> {
        match &a.data() {
            SparseTensorData::CSR {
                values,
                col_indices,
                row_pointers,
            } => Self::spmm_csr(values, col_indices, row_pointers, b, a.shape()),

            SparseTensorData::COO {
                values,
                row_indices,
                col_indices,
            } => Self::spmm_coo(values, row_indices, col_indices, b, a.shape()),

            _ => {
                // Convert to CSR and retry
                let csr = a.to_format(SparseFormat::CSR)?;
                self.spmm(&csr, b)
            }
        }
    }

    fn sddmm(&self,
        _a: &Tensor<B, 2>,
        _b: &Tensor<B, 2>,
        _mask: &SparseTensor<B>,
    ) -> SparseResult<SparseTensor<B>> {
        // TODO: Implement SddMM
        Err(SparseError::UnsupportedOperation {
            operation: "sddmm".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn to_format(&self,
        a: &SparseTensor<B>,
        target: SparseFormat,
    ) -> SparseResult<SparseTensor<B>> {
        a.to_format(target)
    }

    fn to_dense(&self, a: &SparseTensor<B>) -> Tensor<B, 2> {
        a.to_dense()
    }

    fn supports(&self, format: SparseFormat) -> KernelSupport {
        match format {
            SparseFormat::CSR => KernelSupport::Supported,
            SparseFormat::COO => KernelSupport::Supported,
            SparseFormat::Mask => KernelSupport::SupportedWithConversion(SparseFormat::CSR),
            SparseFormat::CSC => KernelSupport::SupportedWithConversion(SparseFormat::CSR),
            SparseFormat::BlockCSR { .. } => {
                KernelSupport::SupportedWithConversion(SparseFormat::CSR)
            }
            SparseFormat::NInM { .. } => KernelSupport::SupportedWithConversion(SparseFormat::CSR),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::Tensor;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_cpu_spmm_csr() {
        // TODO: Add comprehensive tests
    }
}
