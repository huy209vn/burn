//! Integration test for SparseRAM
//!
//! Tests the complete pipeline:
//! 1. Create dense tensor
//! 2. Apply Wanda pruning
//! 3. Convert to SparseRAM
//! 4. Verify VRAM reduction
//! 5. Test forward pass

#![cfg(feature = "experimental")]

mod tests {
    use burn_sparse::{
        core::CalibrationData,
        experimental::sparseram::{SparsePolicy, SparseRAM},
        methods::static_pruning::{Wanda, WandaConfig},
    };
    use burn_core::tensor::Tensor;

    #[cfg(feature = "cuda")]
    use burn_cuda::{Cuda, CudaDevice};
    #[cfg(feature = "cuda")]
    type Backend = Cuda<f32>;
    #[cfg(feature = "cuda")]
    fn default_device() -> CudaDevice {
        CudaDevice::default()
    }

    #[cfg(not(feature = "cuda"))]
    use burn_ndarray::NdArray;
    #[cfg(not(feature = "cuda"))]
    type Backend = NdArray<f32>;
    #[cfg(not(feature = "cuda"))]
    fn default_device() -> <Backend as burn_core::tensor::backend::Backend>::Device {
        Default::default()
    }

    #[test]
    fn test_sparseram_eager_none_high_sparsity() {
        let device = default_device();

        // Create 128×128 weight matrix
        let weights = Tensor::<Backend, 2>::random(
            [128, 128],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        let dense_elements = 128 * 128; // 16384
        let dense_bytes = dense_elements * 4; // 65536 bytes

        // Apply Wanda pruning (90% sparsity - high enough to create zero blocks)
        // With block size 16, need high element sparsity to get block sparsity
        let n_calibration = 32;
        let calibration_data: CalibrationData<Backend> = CalibrationData::from_samples(
            (0..n_calibration)
                .map(|_| {
                    Tensor::random(
                        [1, 128],
                        burn_core::tensor::Distribution::Normal(0.0, 1.0),
                        &device,
                    )
                })
                .collect(),
        );

        let wanda_config = WandaConfig {
            sparsity: 0.9, // 90% element sparsity
            n_calibration,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        let actual_sparsity = mask.actual_sparsity();
        println!("Element-level actual sparsity: {:.1}%", actual_sparsity * 100.0);

        // Convert to SparseRAM with Eager + None
        // Format (CSR/COO/BlockCSR) chosen automatically by burn-sparse
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            // PrunedStorage::None is default
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM");

        // Check VRAM usage
        let vram_bytes = sparse_weight.vram_usage();
        let vram_mb = sparse_weight.vram_mb();

        println!("Dense: {} bytes ({:.2} MB)", dense_bytes, dense_bytes as f32 / (1024.0 * 1024.0));
        println!("SparseRAM: {} bytes ({:.2} MB)", vram_bytes, vram_mb);

        // R4 Architecture: Element-level sparsity via CSR format
        // VRAM reduction proportional to sparsity (no blocks)
        let nnz = sparse_weight.nnz();
        let total_elements = 128 * 128;

        println!("\n=== R4 Element-Level Sparsity ===");
        println!("Non-zero elements: {} / {}", nnz, total_elements);
        println!("Element sparsity: {:.1}%", actual_sparsity * 100.0);

        // VRAM reduction should be proportional to sparsity
        let reduction_ratio = vram_bytes as f32 / dense_bytes as f32;
        println!("VRAM reduction ratio: {:.2} (1.0 = no reduction)", reduction_ratio);

        let reduction_pct = (1.0 - reduction_ratio) * 100.0;
        println!("VRAM reduction: {:.1}%", reduction_pct);

        // RAM usage should be 0 (Eager + None)
        assert_eq!(sparse_weight.ram_usage(), 0, "Eager + None should use 0 RAM");

        // Element sparsity should match Wanda target
        assert!((sparse_weight.sparsity() - actual_sparsity).abs() < 0.05);

        // R4: At 90% sparsity, CSR format should provide significant VRAM reduction
        // Expected: ~70-80% reduction (accounting for CSR index overhead)
        assert!(
            reduction_pct > 60.0,
            "At 90% sparsity, should have >60% VRAM reduction, got {:.1}%",
            reduction_pct
        );

        // Non-zero count should match sparsity
        let expected_nnz = (total_elements as f32 * (1.0 - actual_sparsity)) as usize;
        assert!(
            (nnz as isize - expected_nnz as isize).abs() < 100,
            "NNZ count should match sparsity"
        );
    }

    #[test]
    fn test_sparseram_pruned_storage_ram() {
        let device = default_device();

        // Create test weights
        let weights = Tensor::<Backend, 2>::random(
            [64, 64],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        // Apply 70% sparsity with Wanda
        let n_calibration = 32;
        let calibration_data: CalibrationData<Backend> = CalibrationData::from_samples(
            (0..n_calibration)
                .map(|_| {
                    Tensor::random(
                        [1, 64],
                        burn_core::tensor::Distribution::Normal(0.0, 1.0),
                        &device,
                    )
                })
                .collect(),
        );

        let wanda_config = WandaConfig {
            sparsity: 0.7,
            n_calibration,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        // Convert with PrunedStorage::Ram
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .pruned_storage(burn_sparse::experimental::sparseram::config::PrunedStorageConfig::Ram)
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM with Ram storage");

        // RAM usage should be > 0 (storing pruned values)
        let ram_bytes = sparse_weight.ram_usage();
        println!("RAM usage: {} bytes ({:.2} MB)", ram_bytes, ram_bytes as f32 / (1024.0 * 1024.0));

        // Should have storage available
        assert!(ram_bytes > 0, "PrunedStorage::Ram should use RAM");
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_sparseram_pruned_storage_disk() {
        use std::env;

        let device = default_device();

        // Create test weights
        let weights = Tensor::<Backend, 2>::random(
            [64, 64],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        // Apply 70% sparsity with Wanda
        let n_calibration = 32;
        let calibration_data: CalibrationData<Backend> = CalibrationData::from_samples(
            (0..n_calibration)
                .map(|_| {
                    Tensor::random(
                        [1, 64],
                        burn_core::tensor::Distribution::Normal(0.0, 1.0),
                        &device,
                    )
                })
                .collect(),
        );

        let wanda_config = WandaConfig {
            sparsity: 0.7,
            n_calibration,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        // Create temporary file path
        let temp_dir = env::temp_dir();
        let disk_path = temp_dir.join("test_sparseram_integration_disk.bin");

        // Convert with PrunedStorage::Disk
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .pruned_storage(burn_sparse::experimental::sparseram::config::PrunedStorageConfig::Disk {
                path: disk_path.clone(),
            })
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM with Disk storage");

        // RAM usage should be minimal (just metadata)
        let ram_bytes = sparse_weight.ram_usage();
        println!("RAM usage: {} bytes ({:.2} MB)", ram_bytes, ram_bytes as f32 / (1024.0 * 1024.0));

        // Should have minimal RAM usage (disk-backed)
        assert!(
            ram_bytes < 10000,
            "PrunedStorage::Disk should use minimal RAM, got {} bytes",
            ram_bytes
        );

        // VRAM should still be used for active values
        assert!(sparse_weight.vram_usage() > 0);

        // Cleanup
        let _ = std::fs::remove_file(disk_path);
    }

    #[test]
    fn test_sparseram_different_sparsities() {
        let device = default_device();

        let weights = Tensor::<Backend, 2>::random(
            [128, 128],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        let dense_bytes = 128 * 128 * 4;

        for target_sparsity in [0.3, 0.5, 0.7, 0.9] {
            let calibration_data: CalibrationData<Backend> = CalibrationData::from_samples(
                (0..32)
                    .map(|_| {
                        Tensor::random(
                            [1, 128],
                            burn_core::tensor::Distribution::Normal(0.0, 1.0),
                            &device,
                        )
                    })
                    .collect(),
            );

            let wanda_config = WandaConfig {
                sparsity: target_sparsity,
                n_calibration: 32,
                use_l2: true,
            };

            let mut wanda = Wanda::new(wanda_config);
            let mask = wanda.prune(&weights, &calibration_data);

            let sparse_weight = SparseRAM::enable()
                .policy(SparsePolicy::Eager)
                .apply(weights.clone(), mask)
                .expect("Failed to convert");

            let vram_bytes = sparse_weight.vram_usage();
            let actual_sparsity = sparse_weight.sparsity();

            let reduction = (1.0 - (vram_bytes as f32 / dense_bytes as f32)) * 100.0;

            println!(
                "Sparsity {:.0}%: VRAM = {:.2} MB, Reduction = {:.1}%",
                actual_sparsity * 100.0,
                vram_bytes as f32 / (1024.0 * 1024.0),
                reduction
            );

            // R4: CSR format has overhead, only wins at high sparsity
            // At 70%+ sparsity, should see VRAM reduction
            // At <50% sparsity, CSR overhead may make it larger than dense
            if actual_sparsity >= 0.7 {
                assert!(
                    vram_bytes < dense_bytes,
                    "At {:.0}% sparsity, CSR should reduce VRAM",
                    actual_sparsity * 100.0
                );
            }
        }
    }
}
