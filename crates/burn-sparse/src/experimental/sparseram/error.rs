//! Error types for SparseRAM operations

use core::fmt;

#[cfg(feature = "std")]
use std::error::Error;

/// Result type for SparseRAM operations
pub type SparseRAMResult<T> = Result<T, SparseRAMError>;

/// Errors that can occur during SparseRAM operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseRAMError {
    /// Invalid configuration
    InvalidConfig {
        reason: alloc::string::String,
    },

    /// Invalid block size
    InvalidBlockSize {
        size: usize,
        reason: alloc::string::String,
    },

    /// Dimension mismatch
    DimensionMismatch {
        expected: [usize; 2],
        actual: [usize; 2],
    },

    /// Operation not allowed in current lifecycle mode
    LifecycleViolation {
        operation: alloc::string::String,
        mode: alloc::string::String,
    },

    /// Pruned storage not available
    PrunedStorageUnavailable {
        operation: alloc::string::String,
    },

    /// Block not found
    BlockNotFound {
        row: usize,
        col: usize,
    },

    /// I/O error (disk operations)
    #[cfg(feature = "std")]
    IoError {
        message: alloc::string::String,
    },

    /// Residency engine error
    ResidencyError {
        message: alloc::string::String,
    },

    /// Conversion error
    ConversionError {
        reason: alloc::string::String,
    },

    /// Serialization error
    SerializationError {
        reason: alloc::string::String,
    },
}

impl fmt::Display for SparseRAMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseRAMError::InvalidConfig { reason } => {
                write!(f, "Invalid SparseRAM configuration: {}", reason)
            }
            SparseRAMError::InvalidBlockSize { size, reason } => {
                write!(f, "Invalid block size {}: {}", size, reason)
            }
            SparseRAMError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            SparseRAMError::LifecycleViolation { operation, mode } => {
                write!(
                    f,
                    "Operation '{}' not allowed in {} mode",
                    operation, mode
                )
            }
            SparseRAMError::PrunedStorageUnavailable { operation } => {
                write!(
                    f,
                    "Operation '{}' requires pruned storage, but PrunedStorage::None is set",
                    operation
                )
            }
            SparseRAMError::BlockNotFound { row, col } => {
                write!(f, "Block not found at position ({}, {})", row, col)
            }
            #[cfg(feature = "std")]
            SparseRAMError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            SparseRAMError::ResidencyError { message } => {
                write!(f, "Residency engine error: {}", message)
            }
            SparseRAMError::ConversionError { reason } => {
                write!(f, "Conversion error: {}", reason)
            }
            SparseRAMError::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
        }
    }
}

#[cfg(feature = "std")]
impl Error for SparseRAMError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for SparseRAMError {
    fn from(err: std::io::Error) -> Self {
        SparseRAMError::IoError {
            message: err.to_string(),
        }
    }
}
