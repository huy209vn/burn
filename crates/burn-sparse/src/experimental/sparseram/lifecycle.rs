//! Lifecycle management for SparseRAM weights
//!
//! Separates training mode (reversible) from inference mode (irreversible).

use core::fmt;

/// Lifecycle mode for SparseRAM weight
///
/// SparseRAM weights exist in one of two lifecycle states:
///
/// # Training Mode
/// - Pruned blocks are kept in memory (RAM or Disk)
/// - Operations allowed:
///   - `.to_dense()` - reconstruct full dense tensor
///   - Regrowth (RigL, MEST, DSnoT)
///   - RESU-style optimization of pruned coordinates
///   - Mask refinement
/// - Memory cost: Full model size (active + pruned blocks)
///
/// # Inference Mode (Finalized)
/// - Pruned blocks permanently deleted
/// - Operations allowed:
///   - Sparse inference (forward pass)
///   - Sparse kernel execution
///   - Save/load model
/// - Operations **prohibited** (will panic):
///   - `.to_dense()` - no pruned data to restore
///   - Regrowth - cannot restore pruned weights
///   - RESU - pruned weights don't exist
/// - Memory cost: Minimal (only active blocks)
///
/// # Finalization
///
/// Transitioning from Training → Inference is **irreversible**:
///
/// ```ignore
/// let training_weight = SparseRAM::enable()
///     .pruned_storage(PrunedStorageConfig::Ram)  // Keep pruned blocks
///     .apply(dense, mask)?;
///
/// // Can do regrowth, .to_dense(), etc.
/// let dense = training_weight.to_dense();
///
/// // Finalize for deployment (irreversible!)
/// let inference_weight = training_weight.finalize_inference()?;
///
/// // inference_weight.to_dense() -> PANIC!
/// ```
///
/// The API makes finalization explicit to prevent accidental data loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleMode {
    /// Training mode - pruned blocks available
    ///
    /// Created when `pruned_storage` is `Ram` or `Disk`.
    Training,

    /// Inference mode - pruned blocks deleted
    ///
    /// Created when:
    /// - `pruned_storage` is `None` (direct creation)
    /// - `.finalize_inference()` called (explicit finalization)
    Inference,
}

impl LifecycleMode {
    /// Check if weight is in training mode
    pub fn is_training(self) -> bool {
        matches!(self, LifecycleMode::Training)
    }

    /// Check if weight is in inference mode
    pub fn is_inference(self) -> bool {
        matches!(self, LifecycleMode::Inference)
    }

    /// Get mode name as string
    pub fn name(self) -> &'static str {
        match self {
            LifecycleMode::Training => "Training",
            LifecycleMode::Inference => "Inference",
        }
    }

    /// Check if operation is allowed in this mode
    pub fn allows_operation(self, op: LifecycleOperation) -> bool {
        match (self, op) {
            // Inference mode only allows inference operations
            (LifecycleMode::Inference, LifecycleOperation::SparseInference) => true,
            (LifecycleMode::Inference, LifecycleOperation::SaveLoad) => true,
            (LifecycleMode::Inference, _) => false,

            // Training mode allows everything
            (LifecycleMode::Training, _) => true,
        }
    }
}

impl fmt::Display for LifecycleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Operations that may be lifecycle-restricted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleOperation {
    /// Sparse inference (forward pass)
    SparseInference,

    /// Convert to dense tensor
    ToDense,

    /// Regrow pruned weights (RigL, MEST)
    Regrowth,

    /// RESU optimization of pruned coordinates
    ResuOptimization,

    /// Update sparsity mask
    MaskUpdate,

    /// Save/load model
    SaveLoad,
}

impl LifecycleOperation {
    /// Get operation name
    pub fn name(self) -> &'static str {
        match self {
            LifecycleOperation::SparseInference => "sparse_inference",
            LifecycleOperation::ToDense => "to_dense",
            LifecycleOperation::Regrowth => "regrowth",
            LifecycleOperation::ResuOptimization => "resu_optimization",
            LifecycleOperation::MaskUpdate => "mask_update",
            LifecycleOperation::SaveLoad => "save_load",
        }
    }
}

/// Check if operation is allowed, return error if not
pub fn check_lifecycle(
    mode: LifecycleMode,
    operation: LifecycleOperation,
) -> Result<(), crate::experimental::sparseram::error::SparseRAMError> {
    if mode.allows_operation(operation) {
        Ok(())
    } else {
        Err(crate::experimental::sparseram::error::SparseRAMError::LifecycleViolation {
            operation: alloc::string::String::from(operation.name()),
            mode: alloc::string::String::from(mode.name()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_mode_checks() {
        assert!(LifecycleMode::Training.is_training());
        assert!(!LifecycleMode::Training.is_inference());

        assert!(!LifecycleMode::Inference.is_training());
        assert!(LifecycleMode::Inference.is_inference());
    }

    #[test]
    fn test_training_mode_allows_all() {
        let mode = LifecycleMode::Training;

        assert!(mode.allows_operation(LifecycleOperation::SparseInference));
        assert!(mode.allows_operation(LifecycleOperation::ToDense));
        assert!(mode.allows_operation(LifecycleOperation::Regrowth));
        assert!(mode.allows_operation(LifecycleOperation::ResuOptimization));
        assert!(mode.allows_operation(LifecycleOperation::MaskUpdate));
        assert!(mode.allows_operation(LifecycleOperation::SaveLoad));
    }

    #[test]
    fn test_inference_mode_restrictions() {
        let mode = LifecycleMode::Inference;

        // Allowed
        assert!(mode.allows_operation(LifecycleOperation::SparseInference));
        assert!(mode.allows_operation(LifecycleOperation::SaveLoad));

        // Prohibited
        assert!(!mode.allows_operation(LifecycleOperation::ToDense));
        assert!(!mode.allows_operation(LifecycleOperation::Regrowth));
        assert!(!mode.allows_operation(LifecycleOperation::ResuOptimization));
        assert!(!mode.allows_operation(LifecycleOperation::MaskUpdate));
    }

    #[test]
    fn test_check_lifecycle_ok() {
        let result = check_lifecycle(
            LifecycleMode::Training,
            LifecycleOperation::ToDense,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_lifecycle_error() {
        let result = check_lifecycle(
            LifecycleMode::Inference,
            LifecycleOperation::ToDense,
        );
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(e.to_string().contains("to_dense"));
            assert!(e.to_string().contains("Inference"));
        }
    }

    #[test]
    fn test_mode_display() {
        assert_eq!(format!("{}", LifecycleMode::Training), "Training");
        assert_eq!(format!("{}", LifecycleMode::Inference), "Inference");
    }
}
