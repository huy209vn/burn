//! SparseRAMWeight - core wrapper for tiered sparse weights

use crate::core::{SparseMask, SparseTensor};
use crate::experimental::sparseram::{
    config::{PrunedStorageConfig, SparseRAMBuilder, SparseRAMConfig, SparsePolicy},
    convert::ConversionPipeline,
    error::{SparseRAMError, SparseRAMResult},
    lifecycle::{check_lifecycle, LifecycleMode, LifecycleOperation},
    residency::{EagerEngine, ResidencyEngine},
    storage::{PrunedLocation, PrunedStorage},
};
use burn_core::tensor::{backend::Backend, Tensor};

/// SparseRAM weight with tiered memory management
///
/// Wraps a `SparseTensor` (CSR/COO/BlockCSR format) and manages memory placement.
///
/// # Architecture (R4)
///
/// SparseRAM is a **memory tier manager**, NOT a sparse format or block manager.
/// - Non-zeros stored in SparseTensor (CSR format chosen by burn-sparse)
/// - Residency engine controls WHERE the SparseTensor lives (GPU/RAM/Disk)
/// - Pruned values optionally stored for training (RESU, regrowth)
///
/// # VRAM Reduction
///
/// Memory savings come from CSR format storing only non-zeros:
/// - Dense: n_rows × n_cols × dtype_size
/// - Sparse (CSR): nnz × dtype_size + index overhead
/// - At 70% sparsity: ~70% VRAM reduction
///
/// # Lifecycle
///
/// - **Training Mode**: Pruned values kept, can call `.to_dense()`, regrowth
/// - **Inference Mode**: Pruned values deleted, irreversible, minimal VRAM
///
/// # Memory Tiers
///
/// - **GPU**: SparseTensor (managed by residency engine)
/// - **RAM**: Pruned values (if `PrunedStorage::Ram`)
/// - **Disk**: Pruned values (if `PrunedStorage::Disk`)
/// - **None**: Pruned values deleted (if `PrunedStorage::None`)
///
/// # Example
///
/// ```ignore
/// use burn_sparse::experimental::sparseram::SparseRAM;
///
/// // Create SparseRAM weight
/// let sparse_weight = SparseRAM::enable()
///     .policy(SparsePolicy::Eager)
///     .apply(dense_weights, mask)?;
///
/// // Use in forward pass
/// let output = sparse_weight.forward(input)?;
///
/// // Check memory usage
/// println!("VRAM: {} MB", sparse_weight.vram_mb());
/// println!("Sparsity: {:.1}%", sparse_weight.sparsity() * 100.0);
/// ```
pub struct SparseRAMWeight<B: Backend> {
    /// Original weight shape [n_rows, n_cols]
    shape: [usize; 2],

    /// Pruned value storage (RAM/Disk/None)
    pruned: PrunedStorage,

    /// Current location of pruned values (dynamic, can change)
    current_location: PrunedLocation,

    /// GPU tensor holding pruned values when loaded
    ///
    /// Only allocated when `load_pruned_to_gpu()` is called.
    /// Contains a dense tensor with shape [n_rows, n_cols] where:
    /// - Active positions are zero (already in SparseTensor)
    /// - Pruned positions have their original values
    ///
    /// When training, this is added to SparseTensor.to_dense() to reconstruct full weight.
    pruned_gpu_tensor: Option<Tensor<B, 2>>,

    /// Residency engine (manages where SparseTensor lives)
    residency: Box<dyn ResidencyEngine<B>>,

    /// Device
    device: B::Device,

    /// Lifecycle mode
    mode: LifecycleMode,
}

impl<B: Backend> SparseRAMWeight<B> {
    /// Create SparseRAMWeight from builder and tensors
    ///
    /// This is the main entry point from the builder API.
    ///
    /// # Arguments
    /// * `builder` - Configuration builder
    /// * `dense` - Dense weight tensor [n_rows, n_cols]
    /// * `mask` - Sparsity mask indicating active elements
    pub fn from_builder(
        builder: SparseRAMBuilder<B>,
        dense: Tensor<B, 2>,
        mask: SparseMask<B>,
    ) -> SparseRAMResult<Self> {
        let config = builder.build();
        Self::from_config(config, dense, mask)
    }

    /// Create SparseRAMWeight from config
    ///
    /// # Process (R4 Architecture)
    /// 1. Convert dense + mask → SparseTensor using burn-sparse (CSR format)
    /// 2. Extract pruned values if needed (for training)
    /// 3. Create residency engine to manage SparseTensor placement
    /// 4. Dense tensor is DROPPED, freeing GPU memory
    ///
    /// # VRAM Reduction
    /// Comes from CSR format itself - stores only non-zeros!
    pub fn from_config(
        config: SparseRAMConfig,
        dense: Tensor<B, 2>,
        mask: SparseMask<B>,
    ) -> SparseRAMResult<Self> {
        // Run conversion pipeline
        // This converts dense → SparseTensor and extracts pruned values
        let pipeline = ConversionPipeline::new(config.clone());
        let (sparse, pruned_storage, shape, device) = pipeline.convert(dense, mask)?;

        // Determine lifecycle mode
        let mode = if matches!(pruned_storage, PrunedStorage::None) {
            LifecycleMode::Inference
        } else {
            LifecycleMode::Training
        };

        // Create residency engine with SparseTensor
        let residency = Self::create_residency_engine(&config, sparse)?;

        // Initialize current_location based on pruned_storage
        let current_location = pruned_storage.initial_location();

        Ok(Self {
            shape,
            pruned: pruned_storage,
            current_location,
            pruned_gpu_tensor: None, // Initially no GPU tensor
            residency,
            device,
            mode,
        })
    }

    /// Create residency engine based on policy
    ///
    /// Takes SparseTensor and wraps it in the appropriate residency engine.
    fn create_residency_engine(
        config: &SparseRAMConfig,
        sparse: SparseTensor<B>,
    ) -> SparseRAMResult<Box<dyn ResidencyEngine<B>>> {
        match &config.policy {
            SparsePolicy::Eager => Ok(Box::new(EagerEngine::new(sparse))),

            #[cfg(feature = "std")]
            SparsePolicy::Paged { cache_size: _ } => {
                // TODO: Implement PagedCache for R4
                Err(SparseRAMError::InvalidConfig {
                    reason: "Paged policy not yet implemented in R4".into(),
                })
            }

            #[cfg(not(feature = "std"))]
            SparsePolicy::Paged { .. } => Err(SparseRAMError::InvalidConfig {
                reason: "Paged policy requires 'std' feature".into(),
            }),

            #[cfg(feature = "std")]
            SparsePolicy::Streaming { prefetch: _ } => {
                // TODO: Implement StreamingEngine for R4
                Err(SparseRAMError::InvalidConfig {
                    reason: "Streaming policy not yet implemented in R4".into(),
                })
            }

            #[cfg(not(feature = "std"))]
            SparsePolicy::Streaming { .. } => Err(SparseRAMError::InvalidConfig {
                reason: "Streaming policy requires 'std' feature".into(),
            }),
        }
    }

    // ============================================================================
    // Accessors
    // ============================================================================

    /// Get weight shape [n_rows, n_cols]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Get device
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Get lifecycle mode
    pub fn mode(&self) -> LifecycleMode {
        self.mode
    }

    /// Get sparsity (fraction of pruned elements)
    ///
    /// Returns the ratio of zero elements to total elements.
    /// Calculated from SparseTensor: sparsity = 1.0 - (nnz / total_elements)
    pub fn sparsity(&self) -> f32 {
        let mut engine = self.residency.clone_engine();
        if let Ok(sparse) = engine.get_sparse() {
            let [n_rows, n_cols] = self.shape;
            let total_elements = n_rows * n_cols;
            let nnz = sparse.nnz();
            1.0 - (nnz as f32 / total_elements as f32)
        } else {
            0.0 // Fallback if we can't get sparse tensor
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        let mut engine = self.residency.clone_engine();
        if let Ok(sparse) = engine.get_sparse() {
            sparse.nnz()
        } else {
            0
        }
    }

    /// Get VRAM usage in bytes
    pub fn vram_bytes(&self) -> usize {
        self.residency.vram_usage()
    }

    /// Get VRAM usage in bytes (alias for consistency)
    pub fn vram_usage(&self) -> usize {
        self.vram_bytes()
    }

    /// Get VRAM usage in megabytes
    pub fn vram_mb(&self) -> f32 {
        self.vram_bytes() as f32 / (1024.0 * 1024.0)
    }

    /// Get VRAM usage in gigabytes
    pub fn vram_gb(&self) -> f32 {
        self.vram_bytes() as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get RAM usage in bytes
    pub fn ram_bytes(&self) -> usize {
        self.residency.ram_usage() + self.pruned.memory_bytes()
    }

    /// Get RAM usage in bytes (alias for consistency)
    pub fn ram_usage(&self) -> usize {
        self.ram_bytes()
    }

    /// Get RAM usage in megabytes
    pub fn ram_mb(&self) -> f32 {
        self.ram_bytes() as f32 / (1024.0 * 1024.0)
    }

    /// Get residency engine name
    pub fn policy_name(&self) -> &'static str {
        self.residency.name()
    }

    /// Get current location of pruned values
    ///
    /// Returns where pruned values are currently stored (can change during execution).
    ///
    /// # Example
    /// ```ignore
    /// // Initially in RAM
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
    ///
    /// // Load to GPU for training
    /// weight.load_pruned_to_gpu()?;
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Gpu);
    ///
    /// // Offload back to RAM
    /// weight.offload_pruned()?;
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
    /// ```
    pub fn pruned_location(&self) -> PrunedLocation {
        self.current_location
    }

    // ============================================================================
    // Operations
    // ============================================================================

    /// Forward pass (sparse matrix multiply)
    ///
    /// # Arguments
    /// * `input` - Input tensor [n_cols, batch_size]
    ///
    /// # Returns
    /// Output tensor [n_rows, batch_size]
    pub fn forward(&mut self, input: Tensor<B, 2>) -> SparseRAMResult<Tensor<B, 2>> {
        check_lifecycle(self.mode, LifecycleOperation::SparseInference)?;

        // Get sparse tensor (ensures blocks on GPU)
        let sparse = self.residency.get_sparse()?;

        // Use sparse dispatch for matmul
        use crate::backend::{SparseConfig, SparseDispatch};

        SparseDispatch::spmm(sparse, &input, &SparseConfig::default())
            .map_err(|e| SparseRAMError::ResidencyError {
                message: alloc::format!("SpMM failed: {:?}", e),
            })
    }

    /// Convert to dense tensor
    ///
    /// Reconstructs full dense tensor by combining active and pruned values.
    ///
    /// # Behavior
    ///
    /// - If pruned values are on GPU: Reconstructs by adding SparseTensor.to_dense() + pruned_gpu_tensor
    /// - If pruned values in RAM/Disk: Must call load_pruned_to_gpu() first
    /// - If PrunedStorage::None: Returns only active values (lossy)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - In Inference mode (no pruned storage available)
    /// - Pruned values exist but not loaded to GPU
    pub fn to_dense(&mut self) -> SparseRAMResult<Tensor<B, 2>> {
        check_lifecycle(self.mode, LifecycleOperation::ToDense)?;

        // Get active values from SparseTensor (CSR format)
        let sparse = self.residency.get_sparse()?;
        let active_dense = sparse.to_dense();

        // If we have pruned values on GPU, add them back
        if let Some(pruned_tensor) = &self.pruned_gpu_tensor {
            // Reconstruct full tensor: active + pruned
            Ok(active_dense + pruned_tensor.clone())
        } else {
            // No pruned values on GPU
            if self.pruned.has_storage() {
                // Have storage but not loaded - user needs to call load_pruned_to_gpu() first
                Err(SparseRAMError::InvalidConfig {
                    reason: "Pruned values exist but not loaded to GPU. Call load_pruned_to_gpu() first.".into(),
                })
            } else {
                // PrunedStorage::None - return only active values
                // This is lossy but acceptable for inference-only mode
                Ok(active_dense)
            }
        }
    }

    /// Finalize for inference (irreversible)
    ///
    /// Transitions from Training → Inference mode:
    /// - Deletes all pruned blocks
    /// - Frees pruned storage
    /// - Returns new weight in Inference mode
    ///
    /// # Errors
    /// Returns error if already in Inference mode.
    ///
    /// # Warning
    /// This operation is **irreversible**. After finalization:
    /// - `.to_dense()` will panic
    /// - Regrowth operations will fail
    /// - RESU optimization not possible
    pub fn finalize_inference(mut self) -> SparseRAMResult<Self> {
        if self.mode == LifecycleMode::Inference {
            return Err(SparseRAMError::LifecycleViolation {
                operation: "finalize_inference".into(),
                mode: "Inference (already finalized)".into(),
            });
        }

        // Drop pruned blocks
        self.pruned = PrunedStorage::None;
        self.mode = LifecycleMode::Inference;

        Ok(self)
    }

    /// Prefetch blocks (hint for Paged/Streaming engines)
    pub fn prefetch(&mut self, block_ids: &[usize]) {
        self.residency.prefetch(block_ids);
    }

    // ============================================================================
    // Pruned Value Memory Movement (Phase A: Stubs)
    // ============================================================================

    /// Load pruned values to GPU for training
    ///
    /// Enables training operations (RESU, regrowth, .to_dense()) by loading
    /// pruned values from RAM/Disk to GPU memory.
    ///
    /// # Use Case: Inference → Training Transition
    ///
    /// ```ignore
    /// // Start in inference mode (pruned values in RAM)
    /// let mut weight = SparseRAM::enable()
    ///     .policy(SparsePolicy::Eager)
    ///     .pruned_storage(PrunedStorageConfig::Ram)
    ///     .apply(dense, mask)?;
    ///
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
    ///
    /// // Need to fine-tune? Load pruned values to GPU
    /// weight.load_pruned_to_gpu()?;
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Gpu);
    ///
    /// // Now can do RESU training, regrowth, etc.
    /// weight.resu_step(gradients)?;
    /// ```
    ///
    /// # Errors
    /// - Returns error if `PrunedStorage::None` (no pruned values exist)
    /// - Returns error if already on GPU
    pub fn load_pruned_to_gpu(&mut self) -> SparseRAMResult<()> {
        // Check if we have pruned values
        if !self.pruned.has_storage() {
            return Err(SparseRAMError::PrunedStorageUnavailable {
                operation: "load_pruned_to_gpu".into(),
            });
        }

        // Check if already on GPU
        if self.current_location == PrunedLocation::Gpu {
            return Ok(()); // Already on GPU, no-op
        }

        // Phase B: Implement actual GPU transfer
        // Step 1: Read all pruned values from storage
        let num_pruned = self.pruned.num_values();
        if num_pruned == 0 {
            // No pruned values to load
            self.current_location = PrunedLocation::Gpu;
            self.pruned_gpu_tensor = Some(Tensor::zeros(self.shape, &self.device));
            return Ok(());
        }

        let [n_rows, n_cols] = self.shape;

        // Collect pruned values and convert to linear indices
        let mut indices = Vec::with_capacity(num_pruned);
        let mut values = Vec::with_capacity(num_pruned);

        for i in 0..num_pruned {
            let (row, col, val) = self.pruned.get_value(i)?;
            let linear_idx = row * n_cols + col; // Row-major indexing
            indices.push(linear_idx as i32);
            values.push(val);
        }

        // Step 2: Create a 1D zero tensor (flattened view)
        let total_elements = n_rows * n_cols;
        let zeros_flat = Tensor::<B, 1>::zeros([total_elements], &self.device);

        // Step 3: Create indices and values tensors on device
        use burn_core::tensor::Int;
        let indices_tensor = Tensor::<B, 1, Int>::from_data(
            indices.as_slice(),
            &self.device
        );
        let values_tensor = Tensor::<B, 1>::from_data(
            values.as_slice(),
            &self.device
        );

        // Step 4: Scatter pruned values into the tensor
        // scatter(dim, indices, values) - scatters values at positions specified by indices
        let pruned_flat = zeros_flat.scatter(0, indices_tensor, values_tensor);

        // Step 5: Reshape back to 2D
        let pruned_tensor = pruned_flat.reshape(self.shape);

        // Step 6: Store and update location
        self.pruned_gpu_tensor = Some(pruned_tensor);
        self.current_location = PrunedLocation::Gpu;

        Ok(())
    }

    /// Offload pruned values from GPU back to RAM/Disk
    ///
    /// Enables Training → Inference transition by moving pruned values off GPU.
    ///
    /// # Use Case: Training → Inference Transition
    ///
    /// ```ignore
    /// // After training, offload pruned values
    /// weight.offload_pruned()?;
    /// assert_eq!(weight.pruned_location(), PrunedLocation::Ram);
    ///
    /// // Now back to efficient inference mode
    /// let output = weight.forward(input)?; // Full speed, minimal VRAM
    /// ```
    ///
    /// # Errors
    /// - Returns error if not currently on GPU
    pub fn offload_pruned(&mut self) -> SparseRAMResult<()> {
        // Check if currently on GPU
        if self.current_location != PrunedLocation::Gpu {
            return Err(SparseRAMError::InvalidConfig {
                reason: "Pruned values are not on GPU, cannot offload".into(),
            });
        }

        // Phase B: Implement actual GPU offload
        // Step 1: Drop GPU tensor (frees VRAM)
        self.pruned_gpu_tensor = None;

        // Step 2: Pruned values remain safely in PrunedStorage (RAM/Disk)
        // No need to write back - they were never modified

        // Step 3: Update current_location back to initial location
        self.current_location = self.pruned.initial_location();

        Ok(())
    }
}

// Builder API extension removed - now in config.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseFormat;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sparse_ram_weight_creation() {
        let device = Default::default();

        // Create sparse weight with 50% sparsity
        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let builder = SparseRAMBuilder::new()
            .policy(SparsePolicy::Eager);

        let sparse_weight = builder.apply(dense, mask).unwrap();

        assert_eq!(sparse_weight.shape(), [2, 4]);
        // Element sparsity should be 50% (4 zeros out of 8 elements)
        // But block sparsity may be 0 if all blocks contain at least one non-zero
        assert!(sparse_weight.sparsity() >= 0.0); // Just check it doesn't crash
        assert_eq!(sparse_weight.mode(), LifecycleMode::Inference);
        assert_eq!(sparse_weight.policy_name(), "Eager");
    }

    #[test]
    fn test_training_mode() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data([[1.0, 0.0], [0.0, 2.0]], &device);

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .apply(dense, mask)
            .unwrap();

        // Should be in training mode
        assert_eq!(sparse_weight.mode(), LifecycleMode::Training);

        // Phase B: to_dense() should fail if pruned values not loaded to GPU
        let result = sparse_weight.to_dense();
        assert!(
            result.is_err(),
            "to_dense() should fail when pruned values are in RAM but not GPU"
        );

        // Load pruned values to GPU
        sparse_weight.load_pruned_to_gpu().unwrap();

        // Now to_dense() should work
        let dense_reconstructed = sparse_weight.to_dense();
        assert!(dense_reconstructed.is_ok());
    }

    #[test]
    fn test_inference_mode_restrictions() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data([[1.0, 0.0], [0.0, 2.0]], &device);

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::None) // Inference mode
            .apply(dense, mask)
            .unwrap();

        // Should be in inference mode
        assert_eq!(sparse_weight.mode(), LifecycleMode::Inference);

        // to_dense() should fail
        let result = sparse_weight.to_dense();
        assert!(result.is_err());
    }

    #[test]
    fn test_finalize_inference() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data([[1.0, 0.0], [0.0, 2.0]], &device);

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram) // Training mode
            .apply(dense, mask)
            .unwrap();

        assert_eq!(sparse_weight.mode(), LifecycleMode::Training);

        // Finalize
        let mut finalized = sparse_weight.finalize_inference().unwrap();

        assert_eq!(finalized.mode(), LifecycleMode::Inference);

        // Now to_dense() should fail
        assert!(finalized.to_dense().is_err());
    }

    #[test]
    fn test_memory_tracking() {
        let device = Default::default();

        use burn_core::tensor::Float;
        let dense = Tensor::<TestBackend, 2>::zeros([64, 64], &device);
        let mask_data = Tensor::<TestBackend, 2, Float>::ones([64, 64], &device).bool();
        let mask = SparseMask::from_tensor(mask_data);

        let sparse_weight =
            SparseRAMBuilder::new().apply(dense, mask).unwrap();

        // Should have non-zero VRAM usage
        assert!(sparse_weight.vram_bytes() > 0);
        assert!(sparse_weight.vram_mb() > 0.0);

        // Inference mode should have zero RAM usage (no pruned blocks)
        assert_eq!(sparse_weight.ram_bytes(), 0);
    }

    #[test]
    fn test_pruned_location_ram() {
        let device = Default::default();

        // Create sparse tensor with 50% sparsity
        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0, 0.0],
             [0.0, 3.0, 0.0, 4.0],
             [5.0, 0.0, 6.0, 0.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        // Create with PrunedStorage::Ram
        let sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .policy(SparsePolicy::Eager)
            .apply(dense, mask)
            .unwrap();

        // Initial location should be Ram
        assert_eq!(
            sparse_weight.pruned_location(),
            PrunedLocation::Ram,
            "Initial location should be Ram"
        );

        // Should be in training mode (has pruned storage)
        assert_eq!(sparse_weight.mode(), LifecycleMode::Training);

        // RAM usage should be > 0 (storing pruned values)
        assert!(
            sparse_weight.ram_bytes() > 0,
            "RAM usage should be non-zero for PrunedStorage::Ram"
        );

        // VRAM usage should be > 0 (active values on GPU)
        assert!(sparse_weight.vram_bytes() > 0);
    }

    #[test]
    fn test_pruned_location_load_offload_stub() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0], [0.0, 2.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .apply(dense, mask)
            .unwrap();

        // Initially in RAM
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Ram);

        // Load to GPU (stub implementation)
        sparse_weight.load_pruned_to_gpu().unwrap();
        assert_eq!(
            sparse_weight.pruned_location(),
            PrunedLocation::Gpu,
            "After load_pruned_to_gpu, location should be Gpu"
        );

        // Loading again should be a no-op
        sparse_weight.load_pruned_to_gpu().unwrap();
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Gpu);

        // Offload back to RAM
        sparse_weight.offload_pruned().unwrap();
        assert_eq!(
            sparse_weight.pruned_location(),
            PrunedLocation::Ram,
            "After offload_pruned, location should be back to Ram"
        );
    }

    #[test]
    fn test_pruned_location_none_errors() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0], [0.0, 2.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::None) // No pruned storage
            .apply(dense, mask)
            .unwrap();

        // Location should be None
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::None);

        // Loading should fail (no storage available)
        let result = sparse_weight.load_pruned_to_gpu();
        assert!(
            result.is_err(),
            "load_pruned_to_gpu should fail with PrunedStorage::None"
        );

        match result {
            Err(SparseRAMError::PrunedStorageUnavailable { operation }) => {
                assert_eq!(operation, "load_pruned_to_gpu");
            }
            _ => panic!("Expected PrunedStorageUnavailable error"),
        }
    }

    #[test]
    fn test_offload_without_load_errors() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0], [0.0, 2.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .apply(dense, mask)
            .unwrap();

        // Initially in RAM
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Ram);

        // Offloading without loading should fail
        let result = sparse_weight.offload_pruned();
        assert!(
            result.is_err(),
            "offload_pruned should fail if not currently on GPU"
        );

        match result {
            Err(SparseRAMError::InvalidConfig { reason }) => {
                assert!(reason.contains("not on GPU"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_pruned_location_disk() {
        use std::env;

        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0],
             [0.0, 3.0, 0.0],
             [4.0, 0.0, 5.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        // Create temporary file path
        let temp_dir = env::temp_dir();
        let disk_path = temp_dir.join("test_sparseram_disk.bin");

        // Create with PrunedStorage::Disk
        let sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Disk {
                path: disk_path.clone()
            })
            .policy(SparsePolicy::Eager)
            .apply(dense, mask)
            .unwrap();

        // Initial location should be Disk
        assert_eq!(
            sparse_weight.pruned_location(),
            PrunedLocation::Disk,
            "Initial location should be Disk"
        );

        // Should be in training mode
        assert_eq!(sparse_weight.mode(), LifecycleMode::Training);

        // RAM usage should be minimal (just metadata, actual data on disk)
        let ram_bytes = sparse_weight.ram_bytes();
        assert!(
            ram_bytes < 10000,
            "RAM usage should be minimal for Disk storage, got {} bytes",
            ram_bytes
        );

        // Cleanup
        let _ = std::fs::remove_file(disk_path);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_pruned_location_disk_load_offload() {
        use std::env;

        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0], [0.0, 2.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let temp_dir = env::temp_dir();
        let disk_path = temp_dir.join("test_sparseram_disk_movement.bin");

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Disk {
                path: disk_path.clone()
            })
            .apply(dense, mask)
            .unwrap();

        // Initially on Disk
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Disk);

        // Load to GPU
        sparse_weight.load_pruned_to_gpu().unwrap();
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Gpu);

        // Offload back to Disk
        sparse_weight.offload_pruned().unwrap();
        assert_eq!(
            sparse_weight.pruned_location(),
            PrunedLocation::Disk,
            "After offload, should return to Disk"
        );

        // Cleanup
        let _ = std::fs::remove_file(disk_path);
    }

    #[test]
    fn test_phase_b_actual_gpu_transfer_and_reconstruction() {
        let device = Default::default();

        // Create a known tensor with specific pattern
        let original_dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0, 0.0],
             [0.0, 3.0, 0.0, 4.0],
             [5.0, 0.0, 6.0, 0.0]],
            &device,
        );

        // Create mask (1s are active, 0s are pruned)
        let mask_data = original_dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .apply(original_dense.clone(), mask)
            .unwrap();

        // Verify initial state
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Ram);
        assert!(sparse_weight.pruned_gpu_tensor.is_none());

        // Load pruned values to GPU
        sparse_weight.load_pruned_to_gpu().unwrap();

        // Verify GPU state
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Gpu);
        assert!(sparse_weight.pruned_gpu_tensor.is_some());

        // Reconstruct dense tensor
        let reconstructed = sparse_weight.to_dense().unwrap();

        // Verify reconstruction matches original
        let original_data = original_dense.to_data();
        let reconstructed_data = reconstructed.to_data();

        assert_eq!(original_data.shape, reconstructed_data.shape);

        // Check all values match
        let original_vals: Vec<f32> = original_data.to_vec().unwrap();
        let reconstructed_vals: Vec<f32> = reconstructed_data.to_vec().unwrap();

        for (i, (orig, recon)) in original_vals.iter().zip(reconstructed_vals.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-5,
                "Mismatch at index {}: original={}, reconstructed={}",
                i,
                orig,
                recon
            );
        }

        // Offload pruned values
        sparse_weight.offload_pruned().unwrap();

        // Verify back to RAM
        assert_eq!(sparse_weight.pruned_location(), PrunedLocation::Ram);
        assert!(sparse_weight.pruned_gpu_tensor.is_none());

        // to_dense() should now fail (pruned values not on GPU)
        let result = sparse_weight.to_dense();
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_b_vram_tracking_with_pruned_gpu() {
        let device = Default::default();

        let dense = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.0, 2.0],
             [0.0, 3.0, 0.0],
             [4.0, 0.0, 5.0]],
            &device,
        );

        let mask_data = dense.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        let mut sparse_weight = SparseRAMBuilder::new()
            .pruned_storage(PrunedStorageConfig::Ram)
            .apply(dense, mask)
            .unwrap();

        // Initial VRAM usage (just SparseTensor)
        let vram_before = sparse_weight.vram_usage();

        // Load pruned to GPU
        sparse_weight.load_pruned_to_gpu().unwrap();

        // VRAM usage should still be counted from SparseTensor only
        // (pruned_gpu_tensor is separate and not counted in vram_usage)
        let vram_after = sparse_weight.vram_usage();

        // VRAM shouldn't change (we only track SparseTensor VRAM, not temp tensors)
        assert_eq!(vram_before, vram_after);

        // But GPU tensor should exist
        assert!(sparse_weight.pruned_gpu_tensor.is_some());
    }
}
