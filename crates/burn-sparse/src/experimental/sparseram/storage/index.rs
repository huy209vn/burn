//! Block indexing and fast lookup structures

use super::block::Block;
use crate::experimental::sparseram::error::{SparseRAMError, SparseRAMResult};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Fast lookup structure for block metadata
///
/// Provides O(log n) lookup of blocks by coordinates.
/// Uses BTreeMap for no_std compatibility.
#[derive(Debug, Clone)]
pub struct BlockIndexMap {
    /// Map from (block_row, block_col) → Block
    index: BTreeMap<(usize, usize), Block>,

    /// Total number of active (non-zero) blocks
    num_active: usize,

    /// Total number of pruned (zero) blocks
    num_pruned: usize,

    /// Block size (B in B×B)
    block_size: usize,

    /// Original tensor shape [n_rows, n_cols] in element units
    tensor_shape: [usize; 2],
}

impl BlockIndexMap {
    /// Create new empty block index
    pub fn new(block_size: usize, tensor_shape: [usize; 2]) -> Self {
        Self {
            index: BTreeMap::new(),
            num_active: 0,
            num_pruned: 0,
            block_size,
            tensor_shape,
        }
    }

    /// Insert a block into the index
    pub fn insert(&mut self, block: Block) {
        let coords = block.coords();

        // Update counters
        if block.is_active {
            self.num_active += 1;
        } else {
            self.num_pruned += 1;
        }

        self.index.insert(coords, block);
    }

    /// Get block by coordinates (read-only)
    pub fn get(&self, row: usize, col: usize) -> Option<&Block> {
        self.index.get(&(row, col))
    }

    /// Get mutable block by coordinates
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Block> {
        self.index.get_mut(&(row, col))
    }

    /// Get block by coordinates or error
    pub fn get_or_err(&self, row: usize, col: usize) -> SparseRAMResult<&Block> {
        self.get(row, col).ok_or(SparseRAMError::BlockNotFound { row, col })
    }

    /// Check if block exists at coordinates
    pub fn contains(&self, row: usize, col: usize) -> bool {
        self.index.contains_key(&(row, col))
    }

    /// Get all active blocks
    pub fn active_blocks(&self) -> Vec<&Block> {
        self.index
            .values()
            .filter(|b| b.is_active)
            .collect()
    }

    /// Get all pruned blocks
    pub fn pruned_blocks(&self) -> Vec<&Block> {
        self.index
            .values()
            .filter(|b| !b.is_active)
            .collect()
    }

    /// Get all blocks (active + pruned)
    pub fn all_blocks(&self) -> Vec<&Block> {
        self.index.values().collect()
    }

    /// Get number of active blocks
    pub fn num_active(&self) -> usize {
        self.num_active
    }

    /// Get number of pruned blocks
    pub fn num_pruned(&self) -> usize {
        self.num_pruned
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> usize {
        self.index.len()
    }

    /// Get block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get tensor shape in element units
    pub fn tensor_shape(&self) -> [usize; 2] {
        self.tensor_shape
    }

    /// Get tensor shape in block units
    pub fn block_shape(&self) -> [usize; 2] {
        [
            (self.tensor_shape[0] + self.block_size - 1) / self.block_size,
            (self.tensor_shape[1] + self.block_size - 1) / self.block_size,
        ]
    }

    /// Calculate sparsity at block level
    ///
    /// Returns fraction of blocks that are pruned (zero).
    pub fn block_sparsity(&self) -> f32 {
        if self.total_blocks() == 0 {
            return 0.0;
        }
        self.num_pruned as f32 / self.total_blocks() as f32
    }

    /// Calculate element-level sparsity
    ///
    /// Returns fraction of elements that are pruned (zero).
    pub fn element_sparsity(&self) -> f32 {
        let total_elements = self.tensor_shape[0] * self.tensor_shape[1];
        let pruned_elements = self.num_pruned * self.block_size * self.block_size;

        pruned_elements as f32 / total_elements as f32
    }

    /// Iterate over blocks in row-major order
    pub fn iter_row_major(&self) -> impl Iterator<Item = &Block> {
        self.index.values()
    }

    /// Clear all blocks
    pub fn clear(&mut self) {
        self.index.clear();
        self.num_active = 0;
        self.num_pruned = 0;
    }

    /// Get memory footprint of the index structure itself (bytes)
    pub fn index_memory_bytes(&self) -> usize {
        // Approximate: each entry has two usizes for key + Block struct
        // Block struct is approximately 64 bytes (conservative estimate)
        self.total_blocks() * (2 * core::mem::size_of::<usize>() + 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experimental::sparseram::storage::block::BlockLocation;

    #[test]
    fn test_block_index_creation() {
        let index = BlockIndexMap::new(16, [256, 512]);

        assert_eq!(index.block_size(), 16);
        assert_eq!(index.tensor_shape(), [256, 512]);
        assert_eq!(index.block_shape(), [16, 32]);
        assert_eq!(index.total_blocks(), 0);
    }

    #[test]
    fn test_block_insertion_and_lookup() {
        let mut index = BlockIndexMap::new(16, [256, 512]);

        let block1 = Block {
            row: 0,
            col: 0,
            size: 16,
            is_active: true,
            location: BlockLocation::GPU { offset: 0 },
        };

        let block2 = Block {
            row: 1,
            col: 2,
            size: 16,
            is_active: false,
            location: BlockLocation::None,
        };

        index.insert(block1.clone());
        index.insert(block2.clone());

        assert_eq!(index.total_blocks(), 2);
        assert_eq!(index.num_active(), 1);
        assert_eq!(index.num_pruned(), 1);

        let retrieved = index.get(0, 0).unwrap();
        assert_eq!(retrieved.row, 0);
        assert_eq!(retrieved.col, 0);
        assert!(retrieved.is_active);
    }

    #[test]
    fn test_active_pruned_filtering() {
        let mut index = BlockIndexMap::new(16, [256, 512]);

        for i in 0..10 {
            let block = Block {
                row: i,
                col: 0,
                size: 16,
                is_active: i % 2 == 0, // Even rows active, odd rows pruned
                location: BlockLocation::None,
            };
            index.insert(block);
        }

        assert_eq!(index.active_blocks().len(), 5);
        assert_eq!(index.pruned_blocks().len(), 5);
    }

    #[test]
    fn test_sparsity_calculation() {
        let mut index = BlockIndexMap::new(16, [256, 512]);

        // Insert 8 blocks: 2 active, 6 pruned
        for i in 0..8 {
            let block = Block {
                row: i,
                col: 0,
                size: 16,
                is_active: i < 2,
                location: BlockLocation::None,
            };
            index.insert(block);
        }

        // Block sparsity: 6/8 = 0.75
        assert!((index.block_sparsity() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_block_not_found_error() {
        let index = BlockIndexMap::new(16, [256, 512]);

        let result = index.get_or_err(999, 999);
        assert!(result.is_err());

        match result {
            Err(SparseRAMError::BlockNotFound { row, col }) => {
                assert_eq!(row, 999);
                assert_eq!(col, 999);
            }
            _ => panic!("Expected BlockNotFound error"),
        }
    }
}
