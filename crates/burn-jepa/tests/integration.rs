//! Rigorous and complete integration tests for the `burn-jepa` crate.

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, Backend};
    use burn::module::{Module, ParamId};
    use burn::tensor::{self, Tensor, Data};
    use burn_jepa::model::{Jepa, JepaBatch, JepaConfig};
    use burn_jepa::test_utils::generate_dummy_image;
    use std::collections::HashMap;

    // Trait alias for backend trait bounds
    trait TestBackend: Backend + burn::tensor::backend::AutodiffBackend {}
    impl<B: Backend + burn::tensor::backend::AutodiffBackend> TestBackend for B {}

    // Macro to generate tests for different backends
    macro_rules! test_backend {
        ($backend:ty) => {
            // Forward Pass Test
            #[test]
            fn test_jepa_forward_pass() {
                test_jepa_forward_pass_impl::<$backend>();
            }

            // EMA Update Test
            #[test]
            fn test_jepa_ema_update() {
                test_jepa_ema_update_impl::<$backend>();
            }

            // Training Step Test
            #[test]
            fn test_training_step() {
                test_training_step_impl::<Autodiff<$backend>>();
            }
        };
    }

    // Generic implementation for the forward pass test
    fn test_jepa_forward_pass_impl<B: TestBackend>() {
        let device = Default::default();
        let config = JepaConfig {
            image_size: 32,
            patch_size: 16,
            embed_dim: 64,
            n_layers: 2,
            n_heads: 4,
            ..JepaConfig::new()
        };
        let model = config.init::<B>(&device);
        let dummy_images = generate_dummy_image::<B>(2, 3, 32, 32, &device);
        let batch = JepaBatch {
            images: dummy_images,
        };

        let output = model.forward_step(batch);
        let loss = output.loss;

        assert_eq!(loss.dims(), [1]);
        assert!(loss.into_scalar() > 0.0);
    }

    // Generic implementation for the EMA update test
    fn test_jepa_ema_update_impl<B: TestBackend>() {
        let device = Default::default();
        let config = JepaConfig {
            image_size: 32,
            patch_size: 16,
            embed_dim: 64,
            n_layers: 2,
            n_heads: 4,
            ..JepaConfig::new()
        };
        let model = config.init::<B>(&device);

        // Get initial teacher params
        let initial_teacher_params: HashMap<_, _> =
            model.teacher_encoder.clone().named_parameters().collect();

        // Create a modified student encoder
        let student_encoder_modified = model.student_encoder.clone().map_params(|mut params| {
            params.data = params.data + 1.0;
            params
        });

        // Replace the student encoder and update the teacher
        let model_modified_student = model.clone().with_student_encoder(student_encoder_modified);
        let model_updated = model_modified_student.ema_update(0.996);

        // Get updated teacher params
        let updated_teacher_params: HashMap<_, _> =
            model_updated.teacher_encoder.named_parameters().collect();

        // Ensure teacher params have changed and are not equal to student params
        for (id, initial_param) in initial_teacher_params.iter() {
            let updated_param = updated_teacher_params.get(id).unwrap();

            // Check they are not the same.
            assert_ne!(initial_param.val().to_data(), updated_param.val().to_data());

            // Check that the updated param has moved towards the student param.
            let student_param = model_updated.student_encoder.get_parameter(id).unwrap();
            let initial_diff = (student_param.val() - initial_param.val()).abs().sum().into_scalar();
            let updated_diff = (student_param.val() - updated_param.val()).abs().sum().into_scalar();
            
            assert!(updated_diff < initial_diff);
        }
    }

    // Generic implementation for the training step test
    fn test_training_step_impl<B: TestBackend>() {
        use burn::train::{TrainOutput, TrainStep};

        let device = Default::default();
        let config = JepaConfig {
            image_size: 32,
            patch_size: 16,
            embed_dim: 64,
            n_layers: 2,
            n_heads: 4,
            ..JepaConfig::new()
        };
        let model = config.init::<B>(&device);

        let dummy_images = generate_dummy_image::<B>(2, 3, 32, 32, &device);
        let batch = JepaBatch {
            images: dummy_images,
        };

        // Perform a training step
        let train_output: TrainOutput<B, _, _> = model.step(batch);
        
        // Check that student encoder has grads
        let mut student_grad_norm = 0.0;
        for param in model.student_encoder.parameters() {
            if let Some(grad) = train_output.grads.get(&param.id()) {
                student_grad_norm += grad.abs().sum().into_scalar();
            }
        }
        assert!(student_grad_norm > 0.0);

        // Check that teacher encoder does not have grads
        for param in model.teacher_encoder.parameters() {
            assert!(!train_output.grads.get(&param.id()).is_some());
        }
    }

    // Backend-specific test configurations
    #[cfg(not(feature = "cuda"))]
    mod cpu {
        use super::*;
        use burn::backend::ndarray::NdArray;

        test_backend!(NdArray<f32>);
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use burn::backend::cuda::Cuda;

        type CudaBackend = Cuda<f32, i64>;

        test_backend!(CudaBackend);
    }
}