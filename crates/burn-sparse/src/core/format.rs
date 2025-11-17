/// Sparse tensor storage formats
///
/// Each format is optimized for different workloads:
/// - **Mask**: Boolean mask, no compression, easy manipulation (algorithm layer)
/// - **CSR**: Compressed Sparse Row, row-major SpMM, CPU/GPU standard
/// - **CSC**: Compressed Sparse Column, column-major SpMM
/// - **COO**: Coordinate list, easy construction, element-wise ops
/// - **BlockCSR**: Block-aligned CSR for GPU tensor cores
/// - **NInM**: N:M structured sparsity (e.g., 2:4 for NVIDIA hardware)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SparseFormat {
    /// Boolean mask (no compression, used for algorithms)
    Mask,

    /// Compressed Sparse Row (row-major, standard format)
    CSR,

    /// Compressed Sparse Column (column-major)
    CSC,

    /// Coordinate list (easiest construction)
    COO,

    /// Block-aligned CSR (GPU-optimized, power-of-2 block sizes)
    BlockCSR { block_size: usize },

    /// N:M structured sparsity (hardware-accelerated on NVIDIA GPUs)
    /// For example: N=2, M=4 means 2 non-zeros per 4 elements (50% sparsity)
    NInM { n: usize, m: usize },
}

impl SparseFormat {
    /// Check if format is structured (N:M)
    pub fn is_structured(&self) -> bool {
        matches!(self, Self::NInM { .. })
    }

    /// Check if format is unstructured (any pattern allowed)
    pub fn is_unstructured(&self) -> bool {
        matches!(self, Self::Mask | Self::CSR | Self::CSC | Self::COO | Self::BlockCSR { .. })
    }

    /// Get compression ratio estimate (bytes used / bytes in dense)
    ///
    /// # Arguments
    /// * `sparsity` - Fraction of pruned weights (0.0 = dense, 1.0 = all zeros)
    ///
    /// # Returns
    /// Estimated memory ratio vs dense (e.g., 0.5 = half the memory)
    pub fn compression_ratio(&self, sparsity: f32) -> f32 {
        match self {
            // Boolean mask: 1 bit per element (packed), 4 bytes per float
            Self::Mask => 1.0 / 32.0,

            // CSR/CSC: values (4 bytes) + col_indices (4 bytes) per nnz
            // Plus row_pointers (4 bytes × n_rows, negligible for large matrices)
            Self::CSR | Self::CSC => {
                let nnz_fraction = 1.0 - sparsity;
                nnz_fraction * (4.0 + 4.0) / 4.0  // (val + idx) / dense
            }

            // COO: values + row_indices + col_indices (3 × 4 bytes per nnz)
            Self::COO => {
                let nnz_fraction = 1.0 - sparsity;
                nnz_fraction * (4.0 + 4.0 + 4.0) / 4.0
            }

            // BlockCSR: slight overhead for block pointers
            Self::BlockCSR { .. } => {
                let nnz_fraction = 1.0 - sparsity;
                nnz_fraction * 1.1  // 10% overhead estimate
            }

            // N:M: exactly N/M ratio (e.g., 2:4 = 0.5)
            Self::NInM { n, m } => (*n as f32) / (*m as f32),
        }
    }

    /// Check if block size is valid (power of 2, reasonable range)
    pub fn validate_block_size(block_size: usize) -> Result<(), String> {
        if !block_size.is_power_of_two() {
            return Err(format!("Block size must be power of 2, got {}", block_size));
        }
        if block_size < 4 || block_size > 128 {
            return Err(format!("Block size must be in [4, 128], got {}", block_size));
        }
        Ok(())
    }

    /// Check if N:M parameters are valid
    pub fn validate_nm(n: usize, m: usize) -> Result<(), String> {
        if n == 0 || m == 0 {
            return Err(format!("N and M must be non-zero, got N={}, M={}", n, m));
        }
        if n > m {
            return Err(format!("N must be <= M, got N={}, M={}", n, m));
        }
        // Common patterns: 2:4, 4:8, 1:2, etc.
        // Don't restrict too much, but warn if uncommon
        Ok(())
    }
}

impl core::fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Mask => write!(f, "Mask"),
            Self::CSR => write!(f, "CSR"),
            Self::CSC => write!(f, "CSC"),
            Self::COO => write!(f, "COO"),
            Self::BlockCSR { block_size } => write!(f, "BlockCSR({})", block_size),
            Self::NInM { n, m } => write!(f, "{}:{}", n, m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_ratios() {
        // CSR at 50% sparsity: 0.5 nnz × 8 bytes (val+idx) / 4 bytes = 1.0 (no savings)
        assert!((SparseFormat::CSR.compression_ratio(0.5) - 1.0).abs() < 0.01);

        // CSR at 90% sparsity: 0.1 nnz × 2 = 0.2 (5× smaller)
        assert!((SparseFormat::CSR.compression_ratio(0.9) - 0.2).abs() < 0.01);

        // 2:4 is always 0.5 regardless of sparsity
        assert!((SparseFormat::NInM { n: 2, m: 4 }.compression_ratio(0.0) - 0.5).abs() < 0.01);
        assert!((SparseFormat::NInM { n: 2, m: 4 }.compression_ratio(1.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_block_size_validation() {
        assert!(SparseFormat::validate_block_size(16).is_ok());
        assert!(SparseFormat::validate_block_size(32).is_ok());
        assert!(SparseFormat::validate_block_size(3).is_err());   // Not power of 2
        assert!(SparseFormat::validate_block_size(256).is_err()); // Too large
    }

    #[test]
    fn test_nm_validation() {
        assert!(SparseFormat::validate_nm(2, 4).is_ok());
        assert!(SparseFormat::validate_nm(4, 8).is_ok());
        assert!(SparseFormat::validate_nm(5, 3).is_err()); // N > M
        assert!(SparseFormat::validate_nm(0, 4).is_err()); // N = 0
    }
}
