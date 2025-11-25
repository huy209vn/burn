//! Block representation and metadata

use super::tier::Tier;
use burn_core::tensor::{backend::Backend, TensorData};

/// Physical location of block data in a tier
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockLocation {
    /// Block resides on GPU
    ///
    /// `offset` is the linear index into the SparseTensor's values array
    GPU { offset: usize },

    /// Block resides in RAM
    ///
    /// `index` is the index into the RAM backing store
    RAM { index: usize },

    /// Block resides on disk
    ///
    /// `offset` is the byte offset in the memory-mapped file
    #[cfg(feature = "std")]
    Disk { offset: u64 },

    /// Block has been deleted (no storage)
    None,
}

impl BlockLocation {
    /// Get the tier this location corresponds to
    pub fn tier(&self) -> Tier {
        match self {
            BlockLocation::GPU { .. } => Tier::GPU,
            BlockLocation::RAM { .. } => Tier::RAM,
            #[cfg(feature = "std")]
            BlockLocation::Disk { .. } => Tier::Disk,
            BlockLocation::None => Tier::None,
        }
    }

    /// Check if location has actual storage
    pub fn has_storage(&self) -> bool {
        !matches!(self, BlockLocation::None)
    }
}

/// Single B×B block with metadata
///
/// Blocks are the fundamental unit of memory management in SparseRAM.
/// Each block tracks:
/// - Its logical position in the weight matrix (row, col)
/// - Its size (typically 16×16)
/// - Where it physically resides (GPU/RAM/Disk/None)
/// - Whether it's active (non-zero) or pruned (zero)
#[derive(Debug, Clone)]
pub struct Block {
    /// Block row coordinate (in block units, not element units)
    pub row: usize,

    /// Block column coordinate (in block units, not element units)
    pub col: usize,

    /// Block size (B in B×B)
    pub size: usize,

    /// Whether this block is active (non-zero) or pruned (zero)
    pub is_active: bool,

    /// Current physical location
    pub location: BlockLocation,
}

impl Block {
    /// Create a new block
    pub fn new(row: usize, col: usize, size: usize, is_active: bool) -> Self {
        Self {
            row,
            col,
            size,
            is_active,
            location: BlockLocation::None,
        }
    }

    /// Get the tier where this block resides
    pub fn tier(&self) -> Tier {
        self.location.tier()
    }

    /// Get block coordinates as tuple
    pub fn coords(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// Calculate number of elements in this block
    pub fn num_elements(&self) -> usize {
        self.size * self.size
    }

    /// Calculate memory footprint in bytes for a given element size
    ///
    /// # Arguments
    /// * `elem_size` - Size of each element in bytes (4 for f32, 2 for f16)
    pub fn memory_bytes(&self, elem_size: usize) -> usize {
        if self.location.has_storage() {
            self.num_elements() * elem_size
        } else {
            0
        }
    }

    /// Get element range for this block in the original tensor
    ///
    /// Returns (row_start, row_end, col_start, col_end)
    pub fn element_range(&self) -> (usize, usize, usize, usize) {
        let row_start = self.row * self.size;
        let row_end = row_start + self.size;
        let col_start = self.col * self.size;
        let col_end = col_start + self.size;

        (row_start, row_end, col_start, col_end)
    }
}

/// Block data storage (type-erased for cross-backend compatibility)
///
/// Stores the actual values of a block in a backend-agnostic way.
/// This allows blocks to be stored in RAM/disk regardless of the
/// active backend (CUDA, WGPU, etc.)
#[derive(Debug, Clone)]
pub struct BlockData {
    /// Block metadata
    pub block: Block,

    /// Raw tensor data (can be transferred to any backend)
    pub data: TensorData,
}

impl BlockData {
    /// Create new block data
    pub fn new(block: Block, data: TensorData) -> Self {
        Self { block, data }
    }

    /// Get memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        // TensorData stores values as Vec<E>, calculate from shape
        let num_elements = self.data.shape.iter().product::<usize>();
        // Assuming f32 for now (can be extended for dtype tracking)
        num_elements * 4
    }

    /// Convert to tensor on a specific backend
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> burn_core::tensor::Tensor<B, 2> {
        burn_core::tensor::Tensor::from_data(self.data.clone(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = Block::new(0, 0, 16, true);
        assert_eq!(block.coords(), (0, 0));
        assert_eq!(block.num_elements(), 256);
        assert!(block.is_active);
        assert_eq!(block.tier(), Tier::None);
    }

    #[test]
    fn test_block_element_range() {
        let block = Block::new(2, 3, 16, true);
        let (r_start, r_end, c_start, c_end) = block.element_range();

        assert_eq!(r_start, 32);
        assert_eq!(r_end, 48);
        assert_eq!(c_start, 48);
        assert_eq!(c_end, 64);
    }

    #[test]
    fn test_block_memory_calculation() {
        let block = Block {
            row: 0,
            col: 0,
            size: 16,
            is_active: true,
            location: BlockLocation::RAM { index: 0 },
        };

        // 16x16 block with f32 (4 bytes) = 1024 bytes
        assert_eq!(block.memory_bytes(4), 1024);

        // Same block with f16 (2 bytes) = 512 bytes
        assert_eq!(block.memory_bytes(2), 512);
    }

    #[test]
    fn test_block_location_tier() {
        assert_eq!(BlockLocation::GPU { offset: 0 }.tier(), Tier::GPU);
        assert_eq!(BlockLocation::RAM { index: 0 }.tier(), Tier::RAM);
        #[cfg(feature = "std")]
        assert_eq!(BlockLocation::Disk { offset: 0 }.tier(), Tier::Disk);
        assert_eq!(BlockLocation::None.tier(), Tier::None);
    }
}
