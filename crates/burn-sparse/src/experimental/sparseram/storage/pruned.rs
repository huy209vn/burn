//! Pruned value storage strategies

use crate::experimental::sparseram::error::{SparseRAMError, SparseRAMResult};
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::path::PathBuf;

/// Dynamic location of pruned values at runtime
///
/// Tracks WHERE pruned values currently reside (can change during execution).
///
/// # Distinction from PrunedStorage
///
/// - **PrunedStorage** (static): Policy for where pruned values are STORED long-term
/// - **PrunedLocation** (dynamic): Where they are RIGHT NOW at runtime
///
/// # Use Case: Inference ↔ Training Transitions
///
/// ```ignore
/// // Deploy: Sparse 70B with Eager + Ram
/// let weight = SparseRAM::enable()
///     .policy(SparsePolicy::Eager)
///     .pruned_storage(PrunedStorageConfig::Ram)
///     .apply(dense, mask)?;
///
/// // Inference mode: pruned values in RAM (unused)
/// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
/// let output = weight.forward(input); // Full speed, VRAM reduced
///
/// // Need to fine-tune? Load pruned values to GPU
/// weight.load_pruned_to_gpu()?;
/// assert_eq!(weight.pruned_location(), PrunedLocation::Gpu);
/// // Now can do RESU training, regrowth, etc.
///
/// // Done training? Offload back to RAM
/// weight.offload_pruned()?;
/// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
/// // Back to inference mode
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunedLocation {
    /// No pruned values exist (PrunedStorage::None)
    None,

    /// Pruned values currently in host RAM
    Ram,

    /// Pruned values currently on disk (memory-mapped)
    #[cfg(feature = "std")]
    Disk,

    /// Pruned values temporarily loaded to GPU
    ///
    /// Enables training operations (RESU, regrowth, .to_dense()).
    /// Can be offloaded back to Ram/Disk when done.
    Gpu,
}

/// Storage strategy for pruned (zero) values
///
/// SparseRAM stores non-zeros in SparseTensor (CSR format).
/// Pruned values are the zero elements that were masked out.
///
/// # Storage Options
///
/// ## `None` (Inference-Only)
/// - Pruned values permanently discarded
/// - Minimum memory footprint
/// - Irreversible: `.to_dense()` and regrowth operations will panic
/// - Use for: Production deployment
///
/// ## `Ram` (Training Mode)
/// - Pruned values kept in system RAM
/// - Enables: RESU training, regrowth, `.to_dense()`
/// - Memory cost: Proportional to number of pruned elements
/// - Use for: Training with dynamic sparsity
///
/// ## `Disk` (Archival Mode)
/// - Pruned values stored on disk (memory-mapped)
/// - Enables: Same as `Ram`, but slower access
/// - Memory cost: Minimal (OS page cache)
/// - Use for: Huge models where pruned values exceed RAM
#[derive(Debug)]
pub enum PrunedStorage {
    /// Discard pruned values entirely
    ///
    /// **Warning**: This is irreversible. Operations requiring pruned values
    /// (`.to_dense()`, regrowth) will panic.
    None,

    /// Store pruned values in RAM
    ///
    /// Values stored as `Vec<(row, col, value)>` for fast access.
    Ram {
        /// Pruned value storage: (row, col, value) tuples
        values: Vec<(usize, usize, f32)>,
    },

    /// Store pruned values on disk (memory-mapped file)
    ///
    /// Requires `std` feature.
    #[cfg(feature = "std")]
    Disk {
        /// Path to memory-mapped file
        path: PathBuf,

        /// Memory-mapped file handle
        /// Using Box to avoid large enum variants
        mmap: Box<DiskBackedStorage>,
    },
}

/// Disk-backed storage for pruned values (memory-mapped)
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct DiskBackedStorage {
    /// Memory-mapped file
    mmap: memmap2::MmapMut,

    /// Number of values stored
    num_values: usize,

    /// Size of each entry in bytes
    /// (usize + usize + f32 = 8 + 8 + 4 = 20 bytes per entry)
    entry_size_bytes: usize,
}

#[cfg(feature = "std")]
impl DiskBackedStorage {
    /// Create new disk-backed storage
    pub fn create(
        path: PathBuf,
        num_values: usize,
        entry_size_bytes: usize,
    ) -> SparseRAMResult<Self> {
        use std::fs::OpenOptions;

        let total_bytes = num_values * entry_size_bytes;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| SparseRAMError::IoError {
                message: format!("Failed to create disk storage at {:?}: {}", path, e),
            })?;

        file.set_len(total_bytes as u64)
            .map_err(|e| SparseRAMError::IoError {
                message: format!("Failed to set file size: {}", e),
            })?;

        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file).map_err(|e| SparseRAMError::IoError {
                message: format!("Failed to mmap file: {}", e),
            })?
        };

        Ok(Self {
            mmap,
            num_values,
            entry_size_bytes,
        })
    }

    /// Write value entry at index
    pub fn write_value(&mut self, index: usize, row: usize, col: usize, value: f32) -> SparseRAMResult<()> {
        if index >= self.num_values {
            return Err(SparseRAMError::ConversionError {
                reason: format!("Value index {} out of bounds", index),
            });
        }

        let offset = index * self.entry_size_bytes;

        // Write row (8 bytes)
        let row_bytes = row.to_le_bytes();
        self.mmap[offset..offset + 8].copy_from_slice(&row_bytes);

        // Write col (8 bytes)
        let col_bytes = col.to_le_bytes();
        self.mmap[offset + 8..offset + 16].copy_from_slice(&col_bytes);

        // Write value (4 bytes)
        let value_bytes = value.to_le_bytes();
        self.mmap[offset + 16..offset + 20].copy_from_slice(&value_bytes);

        Ok(())
    }

    /// Read value entry at index
    pub fn read_value(&self, index: usize) -> SparseRAMResult<(usize, usize, f32)> {
        if index >= self.num_values {
            return Err(SparseRAMError::ConversionError {
                reason: format!("Value index {} out of bounds", index),
            });
        }

        let offset = index * self.entry_size_bytes;

        // Read row
        let mut row_bytes = [0u8; 8];
        row_bytes.copy_from_slice(&self.mmap[offset..offset + 8]);
        let row = usize::from_le_bytes(row_bytes);

        // Read col
        let mut col_bytes = [0u8; 8];
        col_bytes.copy_from_slice(&self.mmap[offset + 8..offset + 16]);
        let col = usize::from_le_bytes(col_bytes);

        // Read value
        let mut value_bytes = [0u8; 4];
        value_bytes.copy_from_slice(&self.mmap[offset + 16..offset + 20]);
        let value = f32::from_le_bytes(value_bytes);

        Ok((row, col, value))
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> SparseRAMResult<()> {
        self.mmap.flush().map_err(|e| SparseRAMError::IoError {
            message: format!("Failed to flush mmap: {}", e),
        })
    }
}

impl PrunedStorage {
    /// Create empty RAM storage
    pub fn new_ram() -> Self {
        PrunedStorage::Ram { values: Vec::new() }
    }

    /// Create disk storage at path
    #[cfg(feature = "std")]
    pub fn new_disk(
        path: PathBuf,
        num_values: usize,
        entry_size_bytes: usize,
    ) -> SparseRAMResult<Self> {
        let disk = DiskBackedStorage::create(path.clone(), num_values, entry_size_bytes)?;

        Ok(PrunedStorage::Disk {
            path,
            mmap: Box::new(disk),
        })
    }

    /// Check if pruned values are available
    pub fn has_storage(&self) -> bool {
        !matches!(self, PrunedStorage::None)
    }

    /// Get number of pruned values stored
    pub fn num_values(&self) -> usize {
        match self {
            PrunedStorage::None => 0,
            PrunedStorage::Ram { values } => values.len(),
            #[cfg(feature = "std")]
            PrunedStorage::Disk { mmap, .. } => mmap.num_values,
        }
    }

    /// Set pruned values (bulk operation)
    pub fn set_values(&mut self, pruned_values: Vec<(usize, usize, f32)>) -> SparseRAMResult<()> {
        match self {
            PrunedStorage::None => {
                // Silently ignore - values are discarded
                Ok(())
            }
            PrunedStorage::Ram { values } => {
                *values = pruned_values;
                Ok(())
            }
            #[cfg(feature = "std")]
            PrunedStorage::Disk { mmap, .. } => {
                // Write all values to disk
                for (index, (row, col, value)) in pruned_values.iter().enumerate() {
                    mmap.write_value(index, *row, *col, *value)?;
                }
                mmap.flush()?;
                Ok(())
            }
        }
    }

    /// Get pruned value by index
    pub fn get_value(&self, index: usize) -> SparseRAMResult<(usize, usize, f32)> {
        match self {
            PrunedStorage::None => Err(SparseRAMError::PrunedStorageUnavailable {
                operation: "get_value".into(),
            }),
            PrunedStorage::Ram { values } => values.get(index).copied().ok_or_else(|| {
                SparseRAMError::ConversionError {
                    reason: format!("Value index {} out of bounds", index),
                }
            }),
            #[cfg(feature = "std")]
            PrunedStorage::Disk { mmap, .. } => mmap.read_value(index),
        }
    }

    /// Calculate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            PrunedStorage::None => 0,
            PrunedStorage::Ram { values } => {
                // Each entry: (usize, usize, f32) = 8 + 8 + 4 = 20 bytes
                values.len() * 20
            }
            #[cfg(feature = "std")]
            PrunedStorage::Disk { .. } => {
                // Disk-backed storage uses minimal RAM (OS page cache)
                // Return approximate metadata size
                1024
            }
        }
    }

    /// Get initial location based on storage type
    ///
    /// Returns where pruned values start (before any load_pruned_to_gpu calls).
    pub fn initial_location(&self) -> PrunedLocation {
        match self {
            PrunedStorage::None => PrunedLocation::None,
            PrunedStorage::Ram { .. } => PrunedLocation::Ram,
            #[cfg(feature = "std")]
            PrunedStorage::Disk { .. } => PrunedLocation::Disk,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruned_storage_none() {
        let storage = PrunedStorage::None;

        assert!(!storage.has_storage());
        assert_eq!(storage.num_values(), 0);
        assert_eq!(storage.memory_bytes(), 0);
    }

    #[test]
    fn test_pruned_storage_ram() {
        let mut storage = PrunedStorage::new_ram();

        assert!(storage.has_storage());
        assert_eq!(storage.num_values(), 0);

        // Add some pruned values
        let pruned_values = vec![
            (0, 1, 0.0),
            (1, 0, 0.0),
            (2, 3, 0.0),
        ];

        storage.set_values(pruned_values).unwrap();

        assert_eq!(storage.num_values(), 3);
        assert_eq!(storage.memory_bytes(), 3 * 20); // 3 values × 20 bytes each
    }

    #[test]
    fn test_pruned_storage_none_discard() {
        let mut storage = PrunedStorage::None;

        let pruned_values = vec![
            (0, 1, 0.0),
            (1, 0, 0.0),
        ];

        // Should succeed but discard the values
        storage.set_values(pruned_values).unwrap();
        assert_eq!(storage.num_values(), 0);
    }

    #[test]
    fn test_pruned_storage_get_error() {
        let storage = PrunedStorage::None;

        let result = storage.get_value(0);
        assert!(result.is_err());

        match result {
            Err(SparseRAMError::PrunedStorageUnavailable { operation }) => {
                assert_eq!(operation, "get_value");
            }
            _ => panic!("Expected PrunedStorageUnavailable error"),
        }
    }

    #[test]
    fn test_pruned_storage_ram_get() {
        let mut storage = PrunedStorage::new_ram();

        let pruned_values = vec![
            (0, 1, 1.5),
            (1, 0, 2.5),
            (2, 3, 3.5),
        ];

        storage.set_values(pruned_values).unwrap();

        // Get values back
        let val0 = storage.get_value(0).unwrap();
        assert_eq!(val0, (0, 1, 1.5));

        let val1 = storage.get_value(1).unwrap();
        assert_eq!(val1, (1, 0, 2.5));

        let val2 = storage.get_value(2).unwrap();
        assert_eq!(val2, (2, 3, 3.5));
    }
}
