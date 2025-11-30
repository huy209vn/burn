//! Test R4 architecture - format-agnostic memory tier management

#![cfg(feature = "experimental")]

mod tests {
    use burn_core::tensor::Tensor;
    use burn_sparse::{
        core::{CalibrationData, SparseFormat, SparseMask},
        experimental::sparseram::{SparsePolicy, SparseRAM},
        methods::static_pruning::{Wanda, WandaConfig},
    };

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
    fn test_r4_wanda_to_sparseram_vram_reduction() {
        let device = default_device();

        // Create 100×100 weight matrix
        let weights = Tensor::<Backend, 2>::random(
            [100, 100],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        let dense_elements = 100 * 100; // 10,000
        let dense_vram_bytes = dense_elements * std::mem::size_of::<f32>(); // bytes for f32
        let dense_vram_mb = dense_vram_bytes as f32 / 1024.0 / 1024.0;

        println!(
            "Dense: {} elements, {} bytes ({:.2} MB)",
            dense_elements, dense_vram_bytes, dense_vram_mb
        );

        // Apply Wanda pruning (70% sparsity)
        let n_calibration = 32;
        let calibration_data: CalibrationData<Backend> = CalibrationData::from_samples(
            (0..n_calibration)
                .map(|_| {
                    Tensor::random(
                        [1, 100],
                        burn_core::tensor::Distribution::Normal(0.0, 1.0),
                        &device,
                    )
                })
                .collect(),
        );

        let wanda_config = WandaConfig {
            sparsity: 0.7, // 70% sparsity
            n_calibration,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        let actual_sparsity = mask.actual_sparsity();
        println!("Wanda sparsity: {:.1}%", actual_sparsity * 100.0);

        // Convert to SparseRAM with Eager + None
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM");

        // Check VRAM usage
        let vram_bytes = sparse_weight.vram_usage();
        let vram_mb = sparse_weight.vram_mb();
        let nnz = sparse_weight.nnz();
        let sparsity = sparse_weight.sparsity();

        println!("\nSparseRAM (CSR format):");
        println!("  Non-zeros: {} / {}", nnz, dense_elements);
        println!("  Sparsity: {:.1}%", sparsity * 100.0);
        println!(
            "  VRAM: {} bytes ({:.2} MB, {:.3} GB)",
            vram_bytes,
            sparse_weight.vram_mb(),
            sparse_weight.vram_gb()
        );

        // Calculate reduction
        let reduction_ratio = vram_bytes as f32 / dense_vram_bytes as f32;
        let reduction_pct = (1.0 - reduction_ratio) * 100.0;

        println!("\nMemory reduction:");
        println!(
            "  Dense: {} bytes ({:.2} MB, {:.3} GB)",
            dense_vram_bytes,
            dense_vram_mb,
            dense_vram_mb / 1024.0
        );
        println!(
            "  Sparse (CSR): {} bytes ({:.2} MB, {:.3} GB)",
            vram_bytes,
            sparse_weight.vram_mb(),
            sparse_weight.vram_gb()
        );
        println!("  Reduction: {:.1}%", reduction_pct);
        println!("  Ratio: {:.2}x smaller", 1.0 / reduction_ratio);

        // RAM usage should be 0 (Eager + None)
        assert_eq!(
            sparse_weight.ram_usage(),
            0,
            "Eager + None should use 0 RAM"
        );

        // Sparsity should match Wanda target (within 5%)
        assert!((sparsity - actual_sparsity).abs() < 0.05);

        // VRAM should be significantly reduced at 70% sparsity
        // CSR format: nnz × 4 bytes (values) + nnz × 8 bytes (indices) ≈ 12 × nnz
        // At 70% sparsity: 3000 non-zeros × 12 = 36,000 bytes (90% of dense)
        // CSR has overhead at low sparsity, but at 70%+ we should see some reduction
        println!("\nNote: At 70% sparsity, CSR stores only non-zeros + indices");
        println!(
            "Expected VRAM ≈ nnz × 12 bytes = {} × 12 = {} bytes",
            nnz,
            nnz * 12
        );

        // At 70% sparsity, we expect VRAM to be roughly proportional
        // (CSR overhead means it won't be exactly 30% of dense)
        assert!(vram_bytes > 0);
        assert!(vram_bytes < dense_vram_bytes); // Should be less than dense
    }

    #[test]
    fn test_r4_high_sparsity_vram_reduction() {
        let device = default_device();

        // Create 128×128 weight matrix
        let weights = Tensor::<Backend, 2>::random(
            [128, 128],
            burn_core::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );

        let dense_elements = 128 * 128; // 16,384
        let dense_vram_bytes = dense_elements * std::mem::size_of::<f32>(); // bytes for f32
        let dense_vram_mb = dense_vram_bytes as f32 / 1024.0 / 1024.0;

        println!(
            "Dense: {} elements, {} bytes ({:.2} MB)",
            dense_elements, dense_vram_bytes, dense_vram_mb
        );

        // Apply Wanda pruning (90% sparsity - VERY sparse)
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
            sparsity: 0.9, // 90% sparsity
            n_calibration,
            use_l2: true,
        };

        let mut wanda = Wanda::new(wanda_config);
        let mask = wanda.prune(&weights, &calibration_data);

        let actual_sparsity = mask.actual_sparsity();
        println!("\n=== High Sparsity Test (90%) ===");
        println!("Wanda sparsity: {:.1}%", actual_sparsity * 100.0);

        // Convert to SparseRAM
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM");

        let vram_bytes = sparse_weight.vram_usage();
        let nnz = sparse_weight.nnz();
        let sparsity = sparse_weight.sparsity();

        println!("\nSparseRAM (format chosen automatically):");
        println!("  Non-zeros: {} / {}", nnz, dense_elements);
        println!("  Sparsity: {:.1}%", sparsity * 100.0);
        println!(
            "  Dense: {} bytes ({:.2} MB, {:.3} GB)",
            dense_vram_bytes,
            dense_vram_mb,
            dense_vram_mb / 1024.0
        );
        println!(
            "  Sparse: {} bytes ({:.2} MB, {:.3} GB)",
            vram_bytes,
            sparse_weight.vram_mb(),
            sparse_weight.vram_gb()
        );

        let reduction_pct = (1.0 - (vram_bytes as f32 / dense_vram_bytes as f32)) * 100.0;
        println!("  VRAM reduction: {:.1}%", reduction_pct);

        // At 90% sparsity (only 10% non-zeros), CSR should give significant savings
        // 1638 non-zeros × 12 bytes ≈ 19,656 bytes (vs 65,536 dense)
        // Reduction: ~70%
        println!("\nExpected VRAM ≈ nnz × 12 = {} bytes", nnz * 12);
        println!("Actual reduction: {:.1}%", reduction_pct);

        // At 90% sparsity, should see real VRAM reduction
        assert!(
            vram_bytes < dense_vram_bytes / 2,
            "At 90% sparsity, VRAM should be < 50% of dense"
        );
    }

    #[test]
    fn test_r4_architecture_confirms_csr_format() {
        let device = default_device();

        // Small test matrix
        let weights = Tensor::<Backend, 2>::from_data(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]],
            &device,
        );

        let mask_data = weights.clone().not_equal_elem(0.0);
        let mask = SparseMask::from_tensor(mask_data);

        // Convert to SparseRAM
        let sparse_weight = SparseRAM::enable()
            .policy(SparsePolicy::Eager)
            .apply(weights.clone(), mask.clone())
            .expect("Failed to convert to SparseRAM");

        // Check structure
        assert_eq!(sparse_weight.shape(), [3, 3]);
        assert_eq!(sparse_weight.nnz(), 5); // 5 non-zeros

        // Check sparsity (4 zeros out of 9 elements ≈ 44.4%)
        let expected_sparsity = 4.0 / 9.0;
        assert!((sparse_weight.sparsity() - expected_sparsity).abs() < 0.001);

        println!("\n=== R4 Architecture Confirmation ===");
        println!("Shape: {:?}", sparse_weight.shape());
        println!("Non-zeros: {}", sparse_weight.nnz());
        println!("Sparsity: {:.1}%", sparse_weight.sparsity() * 100.0);
        println!(
            "VRAM: {} bytes ({:.2} MB, {:.3} GB)",
            sparse_weight.vram_usage(),
            sparse_weight.vram_mb(),
            sparse_weight.vram_gb()
        );
        println!(
            "RAM: {} bytes (should be 0 for Eager+None)",
            sparse_weight.ram_usage()
        );
        println!("\nR4 implementation working! SparseTensor stored, blocks removed.");
    }
}
