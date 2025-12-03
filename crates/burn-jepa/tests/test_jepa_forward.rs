use burn::tensor::{Distribution, Tensor};
use burn_jepa::{JepaConfig, JepaBatch};
use burn_ndarray::NdArray;

type TestBackend = NdArray;

#[test]
fn test_jepa_forward_pass() {
    // Create a small JEPA model
    let device = Default::default();
    let config = JepaConfig::new()
        .with_image_size(224)
        .with_patch_size(16)
        .with_in_channels(3)
        .with_embed_dim(128)  // small for testing
        .with_n_layers(2)      // small for testing
        .with_n_heads(4)
        .with_dropout(0.0)
        .with_mlp_ratio(4.0)
        .with_predictor_n_layers(1)
        .with_num_target_blocks(2)
        .with_target_scale_range([0.15, 0.2])
        .with_target_aspect_ratio_range([0.75, 1.5])
        .with_ema_momentum_base(0.996)
        .with_ema_momentum_end(1.0);

    let model = config.init::<TestBackend>(&device);

    // Create a dummy batch
    let batch_size = 2;
    let images = Tensor::random(
        [batch_size, 3, 224, 224],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let batch = JepaBatch { images };

    // Forward pass
    let output = model.forward_step(batch);

    // Check that loss is a valid scalar
    let loss_shape = output.loss.dims();
    assert_eq!(loss_shape, [1], "Loss should be a scalar (1D tensor with dim 1)");

    println!("✓ JEPA forward pass successful!");
    println!("  Loss shape: {:?}", loss_shape);
}
