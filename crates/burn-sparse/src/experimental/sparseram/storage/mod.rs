//! Storage layer for SparseRAM
//!
//! This module defines the physical storage of blocks across memory tiers:
//! - [`Tier`]: GPU / RAM / Disk / None
//! - [`Block`]: Single B×B block with metadata
//! - [`BlockData`]: Block with actual tensor data
//! - [`BlockIndexMap`]: Fast block lookup structure
//! - [`PrunedStorage`]: Storage policy for pruned blocks

pub mod block;
pub mod decompose;
pub mod index;
pub mod pruned;
pub mod tier;

pub use block::{Block, BlockData, BlockLocation};
pub use decompose::{decompose_sparse_tensor, reconstruct_from_blocks};
pub use index::BlockIndexMap;
pub use pruned::{PrunedLocation, PrunedStorage};
pub use tier::Tier;

#[cfg(feature = "std")]
pub use pruned::DiskBackedStorage;
