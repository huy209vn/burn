//! Memory tier definitions for SparseRAM

use core::fmt;

/// Physical memory tier where blocks reside
///
/// SparseRAM partitions weight blocks across different memory tiers
/// based on their usage pattern and residency policy.
///
/// # Tier Hierarchy (fastest → slowest)
///
/// 1. `GPU` - VRAM (1-10 GB/s bandwidth)
/// 2. `RAM` - System memory (50-100 GB/s bandwidth)
/// 3. `Disk` - SSD/HDD storage (3-7 GB/s for NVMe)
/// 4. `None` - Deleted (metadata only)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// GPU VRAM (fastest access, limited capacity)
    ///
    /// Typical capacity: 4-80 GB
    /// Access latency: ~microseconds
    /// Best for: Active blocks during inference
    GPU,

    /// System RAM (fast access, larger capacity)
    ///
    /// Typical capacity: 16-256 GB
    /// Access latency: ~100 microseconds (PCIe transfer)
    /// Best for: Paged cache backing store, pruned blocks for RESU
    RAM,

    /// Disk storage (slowest access, unlimited capacity)
    ///
    /// Typical capacity: 500 GB - multiple TB
    /// Access latency: ~1-10 milliseconds
    /// Best for: Streaming huge models, archival of pruned blocks
    Disk,

    /// Deleted - block data erased, only metadata remains
    ///
    /// Memory usage: ~0 bytes per block
    /// Best for: Inference-only deployment
    None,
}

impl Tier {
    /// Check if tier requires physical storage
    pub fn has_storage(self) -> bool {
        !matches!(self, Tier::None)
    }

    /// Check if tier is on GPU
    pub fn is_gpu(self) -> bool {
        matches!(self, Tier::GPU)
    }

    /// Check if tier is in system memory (RAM or Disk)
    pub fn is_host(self) -> bool {
        matches!(self, Tier::RAM | Tier::Disk)
    }

    /// Get typical access latency in microseconds (approximate)
    pub fn latency_us(self) -> f32 {
        match self {
            Tier::GPU => 1.0,        // Direct VRAM access
            Tier::RAM => 100.0,      // PCIe transfer overhead
            Tier::Disk => 5000.0,    // NVMe read latency
            Tier::None => 0.0,       // No access possible
        }
    }

    /// Get typical bandwidth in GB/s (approximate)
    pub fn bandwidth_gbps(self) -> f32 {
        match self {
            Tier::GPU => 500.0,   // HBM bandwidth (A100/H100)
            Tier::RAM => 50.0,    // PCIe Gen4 x16
            Tier::Disk => 5.0,    // NVMe Gen4
            Tier::None => 0.0,    // No bandwidth
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::GPU => write!(f, "GPU"),
            Tier::RAM => write!(f, "RAM"),
            Tier::Disk => write!(f, "Disk"),
            Tier::None => write!(f, "None"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_properties() {
        assert!(Tier::GPU.has_storage());
        assert!(Tier::RAM.has_storage());
        assert!(Tier::Disk.has_storage());
        assert!(!Tier::None.has_storage());

        assert!(Tier::GPU.is_gpu());
        assert!(!Tier::RAM.is_gpu());

        assert!(Tier::RAM.is_host());
        assert!(Tier::Disk.is_host());
        assert!(!Tier::GPU.is_host());
    }

    #[test]
    fn test_tier_ordering() {
        // GPU should be fastest
        assert!(Tier::GPU.latency_us() < Tier::RAM.latency_us());
        assert!(Tier::RAM.latency_us() < Tier::Disk.latency_us());

        // GPU should have highest bandwidth
        assert!(Tier::GPU.bandwidth_gbps() > Tier::RAM.bandwidth_gbps());
        assert!(Tier::RAM.bandwidth_gbps() > Tier::Disk.bandwidth_gbps());
    }
}
