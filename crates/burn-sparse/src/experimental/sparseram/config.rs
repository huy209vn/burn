//! Configuration and builder API for SparseRAM

use crate::experimental::sparseram::storage::Tier;
use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::path::PathBuf;

/// Residency policy for active (non-zero) blocks
///
/// Determines how active blocks move between memory tiers during execution.
///
/// # Policies
///
/// - **Eager**: All active blocks on GPU at initialization (fastest, highest VRAM)
/// - **Paged**: LRU cache with on-demand loading (moderate speed, controlled VRAM)
/// - **Streaming**: Sequential streaming from disk with prefetch (slowest, minimal VRAM)
#[derive(Debug, Clone, PartialEq)]
pub enum SparsePolicy {
    /// All active blocks on GPU at all times
    ///
    /// **Characteristics:**
    /// - Zero runtime overhead (no block movement)
    /// - Highest VRAM usage
    /// - Best throughput
    ///
    /// **Use when:**
    /// - Sparse model fits in VRAM after compression
    /// - Latency-critical inference
    /// - Small-medium models (1B-13B at 50%+ sparsity)
    Eager,

    /// LRU cache with on-demand block loading
    ///
    /// **Characteristics:**
    /// - Moderate overhead (~5-15% slower than Eager)
    /// - VRAM usage capped at cache_size
    /// - Requires RAM = full model size
    ///
    /// **Use when:**
    /// - Model slightly exceeds VRAM
    /// - Locality exploitable (sequential layers)
    /// - RAM available for backing store
    ///
    /// # Arguments
    /// * `cache_size` - Number of blocks in GPU cache
    Paged {
        /// Maximum number of blocks in GPU cache
        cache_size: usize,
    },

    /// Sequential streaming from disk with prefetch
    ///
    /// **Characteristics:**
    /// - High latency (5-50ms per layer)
    /// - Minimal VRAM (only prefetch buffer)
    /// - Requires fast disk (NVMe recommended)
    ///
    /// **Use when:**
    /// - Model exceeds both VRAM and RAM
    /// - Autoregressive generation (predictable access)
    /// - 70B-200B sparse models on consumer GPUs
    ///
    /// # Arguments
    /// * `prefetch` - Number of blocks to prefetch ahead
    Streaming {
        /// Number of blocks to prefetch ahead of current position
        prefetch: usize,
    },
}

impl Default for SparsePolicy {
    fn default() -> Self {
        SparsePolicy::Eager
    }
}

impl SparsePolicy {
    /// Get policy name as string
    pub fn name(&self) -> &'static str {
        match self {
            SparsePolicy::Eager => "Eager",
            SparsePolicy::Paged { .. } => "Paged",
            SparsePolicy::Streaming { .. } => "Streaming",
        }
    }

    /// Check if policy requires RAM backing store
    pub fn requires_ram(&self) -> bool {
        matches!(self, SparsePolicy::Paged { .. })
    }

    /// Check if policy requires disk storage
    pub fn requires_disk(&self) -> bool {
        matches!(self, SparsePolicy::Streaming { .. })
    }

    /// Get expected VRAM usage ratio relative to Eager
    ///
    /// Returns approximate ratio: 1.0 = full VRAM, 0.1 = 10% VRAM
    pub fn vram_ratio(&self, total_blocks: usize) -> f32 {
        match self {
            SparsePolicy::Eager => 1.0,
            SparsePolicy::Paged { cache_size } => {
                if total_blocks == 0 {
                    0.0
                } else {
                    (*cache_size as f32 / total_blocks as f32).min(1.0)
                }
            }
            SparsePolicy::Streaming { prefetch } => {
                if total_blocks == 0 {
                    0.0
                } else {
                    (*prefetch as f32 / total_blocks as f32).min(0.1)
                }
            }
        }
    }
}

/// Complete SparseRAM configuration
///
/// Encapsulates all settings for converting a dense model to SparseRAM format.
///
/// # R4 Architecture Note
///
/// SparseRAM no longer specifies sparse format - burn-sparse automatically
/// chooses the optimal format (CSR/COO/BlockCSR) based on the sparsity pattern.
#[derive(Debug, Clone)]
pub struct SparseRAMConfig {
    /// Where active blocks reside
    pub active_tier: Tier,

    /// Storage policy for pruned blocks
    pub pruned_storage: PrunedStorageConfig,

    /// Residency policy for active blocks
    pub policy: SparsePolicy,

    /// Block size (B in B×B) - kept for compatibility, not used in R4
    pub block_size: usize,
}

/// Configuration for pruned block storage
#[derive(Debug, Clone)]
pub enum PrunedStorageConfig {
    /// Discard pruned blocks
    None,

    /// Store in RAM
    Ram,

    /// Store on disk at path
    #[cfg(feature = "std")]
    Disk { path: PathBuf },
}

impl Default for SparseRAMConfig {
    fn default() -> Self {
        Self {
            active_tier: Tier::GPU,
            pruned_storage: PrunedStorageConfig::None,
            policy: SparsePolicy::Eager,
            block_size: 16, // Kept for compatibility
        }
    }
}

impl SparseRAMConfig {
    /// Validate configuration
    ///
    /// # R4 Architecture Note
    ///
    /// Format validation removed - burn-sparse chooses format automatically.
    pub fn validate(&self) -> Result<(), alloc::string::String> {
        // Active tier must be GPU (for now)
        if self.active_tier != Tier::GPU {
            return Err(alloc::format!(
                "Active tier must be GPU, got {:?}",
                self.active_tier
            ));
        }

        // Note: block_size validation removed in R4 (blocks are no longer used)
        // Field kept for backwards compatibility only

        Ok(())
    }
}

/// Builder for SparseRAM configuration
///
/// Provides fluent API for constructing SparseRAM weights.
///
/// # R4 Architecture
///
/// SparseRAM is a memory tier manager, NOT a format selector.
/// burn-sparse automatically chooses the optimal format (CSR/COO/BlockCSR)
/// based on the sparsity pattern.
///
/// # Example
///
/// ```ignore
/// use burn_sparse::experimental::sparseram::SparseRAM;
///
/// let sparse_weight = SparseRAM::enable()
///     .policy(SparsePolicy::Eager)
///     .pruned_storage(PrunedStorageConfig::None)
///     .apply(dense_weights, mask)?;
/// ```
pub struct SparseRAMBuilder<B> {
    config: SparseRAMConfig,
    _phantom: PhantomData<B>,
}

impl<B> Default for SparseRAMBuilder<B> {
    fn default() -> Self {
        Self {
            config: SparseRAMConfig::default(),
            _phantom: PhantomData,
        }
    }
}

impl<B> SparseRAMBuilder<B> {
    /// Create new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set active block tier (default: GPU)
    ///
    /// # Arguments
    /// * `tier` - Where active blocks reside
    ///
    /// # Note
    /// Currently only `Tier::GPU` is supported. Future versions may support
    /// multi-tier active storage.
    pub fn active_in(mut self, tier: Tier) -> Self {
        self.config.active_tier = tier;
        self
    }

    /// Set pruned block storage policy (default: None)
    ///
    /// # Arguments
    /// * `storage` - Storage policy for pruned blocks
    ///
    /// # Example
    /// ```ignore
    /// // Inference-only (irreversible)
    /// builder.pruned_storage(PrunedStorageConfig::None)
    ///
    /// // Training mode (keeps pruned blocks in RAM)
    /// builder.pruned_storage(PrunedStorageConfig::Ram)
    /// ```
    pub fn pruned_storage(mut self, storage: PrunedStorageConfig) -> Self {
        self.config.pruned_storage = storage;
        self
    }

    /// Set residency policy (default: Eager)
    ///
    /// # Arguments
    /// * `policy` - How active blocks move between tiers
    ///
    /// # Example
    /// ```ignore
    /// // All blocks on GPU
    /// builder.policy(SparsePolicy::Eager)
    ///
    /// // LRU cache with 1000 blocks
    /// builder.policy(SparsePolicy::Paged { cache_size: 1000 })
    ///
    /// // Streaming with 10-block prefetch
    /// builder.policy(SparsePolicy::Streaming { prefetch: 10 })
    /// ```
    pub fn policy(mut self, policy: SparsePolicy) -> Self {
        self.config.policy = policy;
        self
    }

    /// Set block size (default: 16)
    ///
    /// # Arguments
    /// * `size` - Block size (must be power of 2, between 4 and 64)
    ///
    /// # Panics
    /// Panics if size is not valid (checked during `.apply()`)
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Get immutable reference to config
    pub fn config(&self) -> &SparseRAMConfig {
        &self.config
    }

    /// Consume builder and return config
    pub fn build(self) -> SparseRAMConfig {
        self.config
    }

    /// Apply SparseRAM conversion to dense tensor
    ///
    /// Consumes the builder and returns a SparseRAMWeight.
    ///
    /// # Arguments
    /// * `dense` - Dense weight tensor [n_rows, n_cols]
    /// * `mask` - Sparsity mask indicating active elements
    ///
    /// # Returns
    /// SparseRAMWeight ready for inference
    ///
    /// # Example
    /// ```ignore
    /// let sparse_weight = SparseRAM::enable()
    ///     .policy(SparsePolicy::Eager)
    ///     .apply(weights, mask)?;
    /// ```
    pub fn apply(
        self,
        dense: burn_core::tensor::Tensor<B, 2>,
        mask: crate::core::SparseMask<B>,
    ) -> crate::experimental::sparseram::SparseRAMResult<
        crate::experimental::sparseram::SparseRAMWeight<B>,
    >
    where
        B: burn_core::tensor::backend::Backend,
    {
        use crate::experimental::sparseram::SparseRAMWeight;
        SparseRAMWeight::from_builder(self, dense, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_policy_names() {
        assert_eq!(SparsePolicy::Eager.name(), "Eager");
        assert_eq!(
            SparsePolicy::Paged { cache_size: 100 }.name(),
            "Paged"
        );
        assert_eq!(
            SparsePolicy::Streaming { prefetch: 10 }.name(),
            "Streaming"
        );
    }

    #[test]
    fn test_sparse_policy_requirements() {
        assert!(!SparsePolicy::Eager.requires_ram());
        assert!(SparsePolicy::Paged { cache_size: 100 }.requires_ram());
        assert!(!SparsePolicy::Streaming { prefetch: 10 }.requires_ram());

        assert!(!SparsePolicy::Eager.requires_disk());
        assert!(!SparsePolicy::Paged { cache_size: 100 }.requires_disk());
        assert!(SparsePolicy::Streaming { prefetch: 10 }.requires_disk());
    }

    #[test]
    fn test_vram_ratio() {
        assert_eq!(SparsePolicy::Eager.vram_ratio(1000), 1.0);

        let paged = SparsePolicy::Paged { cache_size: 200 };
        assert!((paged.vram_ratio(1000) - 0.2).abs() < 1e-5);

        let streaming = SparsePolicy::Streaming { prefetch: 50 };
        assert!((streaming.vram_ratio(1000) - 0.05).abs() < 1e-5);
    }

    #[test]
    fn test_config_validation() {
        let config = SparseRAMConfig::default();
        assert!(config.validate().is_ok());

        // R4: Block size validation removed
        // Format is chosen automatically by burn-sparse
        let mut config2 = SparseRAMConfig::default();
        config2.block_size = 32; // No longer validated
        assert!(config2.validate().is_ok());

        // Only active_tier is validated
        let mut config3 = SparseRAMConfig::default();
        config3.active_tier = Tier::RAM;
        assert!(config3.validate().is_err()); // RAM not supported yet
    }

    #[test]
    fn test_builder_defaults() {
        let builder = SparseRAMBuilder::<()>::new();
        let config = builder.build();

        assert_eq!(config.block_size, 16);
        assert_eq!(config.active_tier, Tier::GPU);
        assert!(matches!(config.policy, SparsePolicy::Eager));
        assert!(matches!(config.pruned_storage, PrunedStorageConfig::None));
    }

    #[test]
    fn test_builder_fluent_api() {
        let config = SparseRAMBuilder::<()>::new()
            .block_size(32)
            .policy(SparsePolicy::Paged { cache_size: 500 })
            .pruned_storage(PrunedStorageConfig::Ram)
            .build();

        assert_eq!(config.block_size, 32);
        assert!(matches!(
            config.policy,
            SparsePolicy::Paged { cache_size: 500 }
        ));
        assert!(matches!(config.pruned_storage, PrunedStorageConfig::Ram));
    }
}
