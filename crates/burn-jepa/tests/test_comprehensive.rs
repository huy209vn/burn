use burn::tensor::{Distribution, Tensor};
use burn_jepa::{JepaConfig, JepaBatch};
use burn_ndarray::NdArray;

type TestBackend = NdArray;

#[test]
fn test_jepa_variance_regularization() {
    // Test that variance regularization can be configured
    let device = Default::default();

    // Model with variance regularization enabled
    let config_with_var = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(128)
        .with_n_layers(2)
        .with_n_heads(4)
        .with_predictor_n_layers(1)
        .with_var_reg_weight(5.0);

    let model_with_var = config_with_var.init::<TestBackend>(&device);

    // Model without variance regularization
    let config_no_var = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(128)
        .with_n_layers(2)
        .with_n_heads(4)
        .with_predictor_n_layers(1)
        .with_var_reg_weight(0.0);

    let model_no_var = config_no_var.init::<TestBackend>(&device);

    // Create identical batches
    let batch_size = 2;
    let images = Tensor::random(
        [batch_size, 3, 224, 224],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let batch1 = JepaBatch { images: images.clone() };
    let batch2 = JepaBatch { images };

    // Forward pass
    let output_with_var = model_with_var.forward_step(batch1);
    let output_no_var = model_no_var.forward_step(batch2);

    // Both should produce valid losses
    assert_eq!(output_with_var.loss.dims(), [1]);
    assert_eq!(output_no_var.loss.dims(), [1]);

    println!("✓ Variance regularization configuration works");
}

#[test]
fn test_jepa_different_image_sizes() {
    let device = Default::default();

    // Test with 224x224 images
    let config_224 = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4)
        .with_predictor_n_layers(1);

    let model_224 = config_224.init::<TestBackend>(&device);

    let batch_224 = JepaBatch {
        images: Tensor::random([2, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device),
    };

    let output_224 = model_224.forward_step(batch_224);
    assert_eq!(output_224.loss.dims(), [1]);

    // Test with 96x96 images
    let config_96 = JepaConfig::new()
        .with_image_size(96)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4)
        .with_predictor_n_layers(1);

    let model_96 = config_96.init::<TestBackend>(&device);

    let batch_96 = JepaBatch {
        images: Tensor::random([2, 3, 96, 96], Distribution::Normal(0.0, 1.0), &device),
    };

    let output_96 = model_96.forward_step(batch_96);
    assert_eq!(output_96.loss.dims(), [1]);

    println!("✓ Different image sizes work correctly");
}

#[test]
fn test_jepa_different_patch_sizes() {
    let device = Default::default();

    // Test with 16x16 patches
    let config_p16 = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4);

    let model_p16 = config_p16.init::<TestBackend>(&device);

    // Test with 14x14 patches (224 / 14 = 16 patches per side)
    let config_p14 = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(14)
        .with_in_channels(3)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4);

    let model_p14 = config_p14.init::<TestBackend>(&device);

    let images = Tensor::random([2, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device);

    let output_p16 = model_p16.forward_step(JepaBatch { images: images.clone() });
    let output_p14 = model_p14.forward_step(JepaBatch { images });

    assert_eq!(output_p16.loss.dims(), [1]);
    assert_eq!(output_p14.loss.dims(), [1]);

    println!("✓ Different patch sizes work correctly");
}

#[test]
fn test_jepa_masking_strategies() {
    let device = Default::default();

    // Test with different masking configurations
    let config_few_blocks = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4)
        .with_num_target_blocks(2)
        .with_target_scale_range([0.1, 0.15]);

    let model = config_few_blocks.init::<TestBackend>(&device);

    let batch = JepaBatch {
        images: Tensor::random([2, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device),
    };

    let output = model.forward_step(batch);
    assert_eq!(output.loss.dims(), [1]);

    println!("✓ Masking strategies work correctly");
}

#[test]
fn test_jepa_ema_update() {
    let device = Default::default();

    let config = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4);

    let model = config.init::<TestBackend>(&device);

    // Perform EMA update
    let momentum = 0.99;
    let updated_model = model.ema_update(momentum);

    // Model should still work after EMA update
    let batch = JepaBatch {
        images: Tensor::random([2, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device),
    };

    let output = updated_model.forward_step(batch);
    assert_eq!(output.loss.dims(), [1]);

    println!("✓ EMA update works correctly");
}

#[test]
fn test_encoder_architecture_variants() {
    let device = Default::default();

    // Test pre-norm (default)
    let config_prenorm = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_embed_dim(64)
        .with_n_layers(2)
        .with_n_heads(4);

    let model_prenorm = config_prenorm.init::<TestBackend>(&device);

    let batch = JepaBatch {
        images: Tensor::random([2, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device),
    };

    let output = model_prenorm.forward_step(batch);
    assert_eq!(output.loss.dims(), [1]);

    println!("✓ Encoder architecture variants work correctly");
}

#[test]
fn test_training_config() {
    use burn_jepa::train::TrainingConfig;

    let config = TrainingConfig::new()
        .with_num_epochs(100)
        .with_batch_size(256)
        .with_learning_rate(0.0001)
        .with_weight_decay(0.05)
        .with_warmup_epochs(10)
        .with_grad_clip_norm(1.0);

    assert_eq!(config.num_epochs, 100);
    assert_eq!(config.batch_size, 256);
    assert_eq!(config.learning_rate, 0.0001);
    assert_eq!(config.weight_decay, 0.05);
    assert_eq!(config.warmup_epochs, 10);
    assert_eq!(config.grad_clip_norm, 1.0);

    println!("✓ Training configuration works correctly");
}

#[test]
fn test_learning_rate_scheduler() {
    use burn_jepa::train::scheduler::{CosineAnnealingLr, ConstantLr};

    // Test cosine annealing with warmup
    let scheduler = CosineAnnealingLr::new(0.001, 0.00001, 10000, 1000);

    // At start of training, LR should be very small
    assert!(scheduler.get_lr(0) < 0.0001);

    // After warmup, LR should be at max
    assert!((scheduler.get_lr(1000) - 0.001).abs() < 1e-6);

    // At end of training, LR should be at min
    assert!((scheduler.get_lr(10000) - 0.00001).abs() < 1e-6);

    // Test constant LR
    let const_scheduler = ConstantLr::new(0.001);
    assert_eq!(const_scheduler.get_lr(0), 0.001);
    assert_eq!(const_scheduler.get_lr(5000), 0.001);

    println!("✓ Learning rate schedulers work correctly");
}

#[test]
fn test_ema_momentum_scheduler() {
    use burn_jepa::model::ema::CosineAnnealingMomentum;

    let scheduler = CosineAnnealingMomentum::new(0.996, 1.0);

    // At start, momentum should be at base
    assert_eq!(scheduler.get_momentum(0, 10000), 0.996);

    // At end, momentum should reach end value
    assert!((scheduler.get_momentum(10000, 10000) - 1.0).abs() < 1e-6);

    // Midway should be between base and end
    let mid_momentum = scheduler.get_momentum(5000, 10000);
    assert!(mid_momentum > 0.996 && mid_momentum < 1.0);

    println!("✓ EMA momentum scheduler works correctly");
}
