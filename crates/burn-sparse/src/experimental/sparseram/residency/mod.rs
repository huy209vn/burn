//! Residency engines for managing block placement across memory tiers
//!
//! Each engine implements a different strategy for keeping blocks on GPU:
//! - [`EagerEngine`]: All blocks on GPU at all times (zero overhead)
//! - [`PagedCache`]: LRU cache with on-demand loading (moderate overhead)
//! - [`StreamingEngine`]: Sequential streaming from disk (high latency, minimal VRAM)

use crate::core::SparseTensor;
use crate::experimental::sparseram::error::SparseRAMResult;
use burn_core::tensor::backend::Backend;
use core::fmt::Debug;

pub mod eager;

#[cfg(feature = "std")]
pub mod cache;

#[cfg(feature = "std")]
pub mod paged;

#[cfg(feature = "std")]
pub mod streaming;

// Re-exports
pub use eager::EagerEngine;

#[cfg(feature = "std")]
pub use cache::LRUCache;

#[cfg(feature = "std")]
pub use paged::PagedCache;

#[cfg(feature = "std")]
pub use streaming::StreamingEngine;

/// Trait for block residency management
///
/// Residency engines control where blocks physically reside and when they move.
/// Each engine provides different trade-offs between VRAM usage, latency, and throughput.
///
/// # Lifecycle
///
/// 1. **Construction**: Engine created with initial block configuration
/// 2. **Execution**: `execute()` called for each forward pass
/// 3. **Movement**: Engine ensures required blocks are on GPU before execution
/// 4. **Cleanup**: Engine dropped, resources released
///
/// # Thread Safety
///
/// Engines must be `Send + Sync` to support multi-threaded training.
/// Implementations use interior mutability (Mutex, RwLock) where needed.
pub trait ResidencyEngine<B: Backend>: Send + Sync + Debug {
    /// Get reference to active sparse tensor on GPU
    ///
    /// The engine guarantees that all required blocks are on GPU.
    /// For Eager: returns immediately.
    /// For Paged/Streaming: ensures blocks loaded before returning.
    ///
    /// # Returns
    /// Reference to SparseTensor with active blocks on GPU
    fn get_sparse(&mut self) -> SparseRAMResult<&SparseTensor<B>>;

    /// Get current VRAM usage in bytes
    ///
    /// Returns approximate memory footprint of blocks currently on GPU.
    fn vram_usage(&self) -> usize;

    /// Get current RAM usage in bytes
    ///
    /// Returns approximate memory footprint of blocks currently in RAM.
    /// For Eager engine, this is always 0.
    fn ram_usage(&self) -> usize {
        0
    }

    /// Prefetch blocks (hint for Paged/Streaming engines)
    ///
    /// Signals that certain blocks will be needed soon. Paged/Streaming
    /// engines can use this to load blocks asynchronously.
    ///
    /// # Arguments
    /// * `block_ids` - Indices of blocks to prefetch
    ///
    /// # Note
    /// This is a hint only - implementations may ignore it.
    fn prefetch(&mut self, _block_ids: &[usize]) {
        // Default: no-op (Eager doesn't need prefetch)
    }

    /// Clone the engine (for model cloning)
    ///
    /// Required because `Clone` trait can't be made object-safe.
    fn clone_engine(&self) -> Box<dyn ResidencyEngine<B>>;

    /// Get engine name (for debugging)
    fn name(&self) -> &'static str;

    /// Get estimated latency per forward pass in microseconds
    fn latency_us(&self) -> f32 {
        1.0 // Default: assume negligible overhead
    }
}

/// Helper trait to make ResidencyEngine cloneable
pub trait CloneableResidencyEngine<B: Backend>: ResidencyEngine<B> {
    /// Clone this engine
    fn clone_box(&self) -> Box<dyn ResidencyEngine<B>>;
}

impl<B: Backend, T> CloneableResidencyEngine<B> for T
where
    T: ResidencyEngine<B> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ResidencyEngine<B>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that ResidencyEngine is object-safe
    #[allow(dead_code)]
    fn assert_object_safe<B: Backend>(_engine: Box<dyn ResidencyEngine<B>>) {}
}
