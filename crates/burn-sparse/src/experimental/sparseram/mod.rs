//! # SparseRAM: Runtime Memory Tiering for Sparse Models
//!
//! SparseRAM is a runtime extension that ensures only **active (non-zero) blocks** of a sparse
//! weight matrix reside in VRAM, while **pruned (zero) blocks** are stored in RAM, on Disk, or
//! erased entirely.
//!
//! This achieves **real VRAM reduction proportional to sparsity** and enables inference of
//! extremely large sparse models on consumer GPUs.
//!
//! ## What SparseRAM Does
//!
//! - **Memory Tiering**: GPU → RAM → Disk → None
//! - **Block Management**: Only non-zero blocks occupy VRAM
//! - **Residency Policies**: Eager (all GPU) / Paged (LRU cache) / Streaming (disk I/O)
//! - **Lifecycle Management**: Training (reversible) vs Inference (irreversible)
//!
//! ## What Sparse RAM Does NOT Do
//!
//! - Sparsification (use Wanda, Magnitude, RigL, MEST from `burn-sparse`)
//! - Sparse kernels (uses existing `burn-sparse` kernels)
//! - Sparse training algorithms (handled by `burn-sparse::methods`)
//!
//! ## Quick Start
//!
//! ### Inference-Only (Irreversible)
//!
//! ```ignore
//! use burn_sparse::experimental::sparseram::SparseRAM;
//! use burn_sparse::prelude::*;
//!
//! // Step 1: Prune with Wanda
//! let mask = Wanda::new(config).prune(&weights, &calibration_data);
//!
//! // Step 2: Convert to SparseRAM (Eager policy)
//! let sparse_weight = SparseRAM::enable()
//!     .pruned_storage(PrunedStorageConfig::None)  // Discard pruned blocks
//!     .policy(SparsePolicy::Eager)                 // All on GPU
//!     .apply(weights, mask)?;
//!
//! // VRAM reduced by sparsity percentage!
//! println!("VRAM usage: {} MB", sparse_weight.vram_mb());
//! ```
//!
//! ### Training Mode (Reversible)
//!
//! ```ignore
//! // Keep pruned blocks for RESU training / regrowth
//! let sparse_weight = SparseRAM::enable()
//!     .pruned_storage(PrunedStorageConfig::Ram)  // Keep in RAM
//!     .policy(SparsePolicy::Eager)
//!     .apply(weights, mask)?;
//!
//! // Can reconstruct dense tensor
//! let dense = sparse_weight.to_dense();
//!
//! // Finalize for deployment (irreversible!)
//! let inference_weight = sparse_weight.finalize_inference()?;
//! ```
//!
//! ### Large Models (Paged Cache)
//!
//! ```ignore
//! // Model slightly exceeds VRAM - use LRU cache
//! let sparse_weight = SparseRAM::enable()
//!     .policy(SparsePolicy::Paged {
//!         cache_size: 1000,  // Keep 1000 blocks in GPU cache
//!     })
//!     .apply(weights, mask)?;
//! ```
//!
//! ### Ultra-Large Models (Streaming)
//!
//! ```ignore
//! // 70B+ model - stream from disk
//! let sparse_weight = SparseRAM::enable()
//!     .policy(SparsePolicy::Streaming {
//!         prefetch: 10,  // Prefetch 10 blocks ahead
//!     })
//!     .apply(weights, mask)?;
//! ```
//!
//! ## Memory Footprint
//!
//! | Sparsity | Dense (14GB) | SparseRAM Eager | SparseRAM Streaming |
//! |----------|--------------|-----------------|---------------------|
//! | 50%      | 14 GB        | 7 GB            | ~100 MB             |
//! | 70%      | 14 GB        | 4.2 GB          | ~100 MB             |
//! | 90%      | 14 GB        | 1.4 GB          | ~100 MB             |
//!
//! ## Architecture
//!
//! ```text
//! Dense Weights (14 GB)
//!        ↓ [Apply mask]
//! ┌──────────────┬──────────────┐
//! │ Active (30%) │ Pruned (70%) │
//! └──────────────┴──────────────┘
//!        ↓                ↓
//!    [GPU/RAM/Disk]   [None/RAM/Disk]
//!        ↓
//! SparseRAMWeight (4.2 GB VRAM)
//! ```
//!
//! ## Implementation Status
//!
//! - ✅ **Core Infrastructure**: Tiers, Blocks, IndexMap
//! - ✅ **Eager Policy**: Zero overhead, all blocks on GPU
//! - ⚠️ **Paged Policy**: LRU cache implemented, needs integration testing
//! - ⚠️ **Streaming Policy**: Stub implemented, needs disk I/O completion
//! - ✅ **Lifecycle Management**: Training vs Inference modes
//! - ⚠️ **Serialization**: Format defined, needs implementation
//!
//! ## Feature Flags
//!
//! - `experimental` - Enable SparseRAM (required)
//! - `std` - Enable disk storage and streaming (required for Paged/Streaming)
//!
//! ## Safety
//!
//! - `PrunedStorage::None` is **irreversible** - pruned blocks permanently deleted
//! - `.finalize_inference()` is **irreversible** - converts Training → Inference mode
//! - Operations requiring pruned blocks will **panic** in Inference mode

pub mod config;
pub mod convert;
pub mod error;
pub mod residency;
pub mod storage;
pub mod weight;

#[cfg(feature = "std")]
pub mod io;

// Re-exports for convenient access
pub use config::{PrunedStorageConfig, SparsePolicy, SparseRAMBuilder, SparseRAMConfig};
pub use error::{SparseRAMError, SparseRAMResult};
pub use storage::{Block, BlockData, BlockIndexMap, BlockLocation, PrunedStorage, Tier};
pub use weight::SparseRAMWeight;

// Re-export residency engines
pub use residency::{EagerEngine, ResidencyEngine};

#[cfg(feature = "std")]
pub use residency::{PagedCache, StreamingEngine};

/// Entry point for SparseRAM API
///
/// # Example
///
/// ```ignore
/// use burn_sparse::experimental::sparseram::SparseRAM;
///
/// let sparse_weight = SparseRAM::enable()
///     .policy(SparsePolicy::Eager)
///     .apply(weights, mask)?;
/// ```
pub struct SparseRAM;

impl SparseRAM {
    /// Start building a SparseRAM configuration
    ///
    /// Returns a builder with default settings:
    /// - Format: BlockCSR { block_size: 16 }
    /// - Active tier: GPU
    /// - Pruned storage: None
    /// - Policy: Eager
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SparseRAM::enable()
    ///     .block_size(32)
    ///     .policy(SparsePolicy::Paged { cache_size: 500 })
    ///     .build();
    /// ```
    pub fn enable<B>() -> SparseRAMBuilder<B> {
        SparseRAMBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_ram_enable() {
        let _builder = SparseRAM::enable::<burn_ndarray::NdArray<f32>>();
        // Just test that we can create the builder
    }
}
