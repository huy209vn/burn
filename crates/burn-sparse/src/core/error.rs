/// Errors that can occur in burn-sparse operations
#[derive(Debug, thiserror::Error)]
pub enum SparseError {
    /// Format not supported by backend
    #[error("Unsupported format {format} on backend {backend}")]
    UnsupportedFormat {
        backend: String,
        format: crate::core::format::SparseFormat,
    },

    /// Invalid sparse tensor structure
    #[error("Invalid sparse tensor: {reason}")]
    InvalidTensor { reason: String },

    /// Shape mismatch in operation
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: [usize; 2],
        got: [usize; 2],
    },

    /// Format conversion failed
    #[error("Format conversion failed: {from} → {to}: {reason}")]
    ConversionFailed {
        from: crate::core::format::SparseFormat,
        to: crate::core::format::SparseFormat,
        reason: String,
    },

    /// N:M constraint violation
    #[error("N:M constraint violation: {details}")]
    NInMViolation { details: String },

    /// Backend-specific error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Device mismatch
    #[error("Device mismatch: tensors on different devices")]
    DeviceMismatch,

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Result type for burn-sparse operations
pub type SparseResult<T> = Result<T, SparseError>;
