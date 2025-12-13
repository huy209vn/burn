//! Main JEPA module and configuration, following the `Config::init()` pattern.

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn::train::{TrainOutput, TrainStep, ValidStep};

use super::encoder::{VisionTransformer, VisionTransformerConfig};
use super::loss::jepa_loss;
use super::mask::{sample_block_masks, MaskingConfig, MaskOutput};
use super::patch_embed::{PatchEmbed, PatchEmbedConfig};
use super::predictor::{Predictor, PredictorConfig};


// =================================================================
// --- Configurations ---
// =================================================================

#[derive(Config, Debug)]
pub struct JepaConfig {
    #[config(default = 224)]
    pub image_size: usize,
    #[config(default = 16)]
    pub patch_size: usize,
    #[config(default = 3)]
    pub in_channels: usize,
    #[config(default = 768)]
    pub embed_dim: usize,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
    #[config(default = 6)]
    pub predictor_n_layers: usize,
    #[config(default = 4)]
    pub num_target_blocks: usize,
    #[config(default = "[0.15, 0.2]")]
    pub target_scale_range: [f32; 2],
    #[config(default = "[0.75, 1.5]")]
    pub target_aspect_ratio_range: [f32; 2],
    #[config(default = 0.996)]
    pub ema_momentum_base: f64,
    #[config(default = 1.0)]
    pub ema_momentum_end: f64,
    /// Variance regularization weight (0.0 = disabled, typical: 1.0-10.0)
    #[config(default = 5.0)]
    pub var_reg_weight: f32,
}

impl JepaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Jepa<B> {
        // Initialize patch embedding
        let patch_embed = PatchEmbedConfig {
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
        }
        .init(device);

        // Initialize student encoder
        let student_encoder = VisionTransformerConfig::new(self.embed_dim, self.n_layers, self.n_heads)
            .with_dropout(self.dropout)
            .with_mlp_ratio(self.mlp_ratio)
            .init(device);

        // Initialize teacher encoder (clone of student)
        let teacher_encoder = VisionTransformerConfig::new(self.embed_dim, self.n_layers, self.n_heads)
            .with_dropout(self.dropout)
            .with_mlp_ratio(self.mlp_ratio)
            .init(device);

        // Initialize predictor
        let predictor = PredictorConfig {
            embed_dim: self.embed_dim,
            predictor_embed_dim: self.embed_dim, // Using same dim for simplicity
            n_layers: self.predictor_n_layers,
            n_heads: self.n_heads,
            dropout: self.dropout,
            mlp_ratio: self.mlp_ratio,
        }
        .init(device);

        Jepa {
            patch_embed,
            student_encoder,
            teacher_encoder,
            predictor,
            num_target_blocks: self.num_target_blocks,
            target_scale_range: self.target_scale_range,
            target_aspect_ratio_range: self.target_aspect_ratio_range,
            patch_size: self.patch_size,
            image_size: self.image_size,
            var_reg_weight: self.var_reg_weight,
        }
    }
}


// =================================================================
// --- Model & Step Output ---
// =================================================================

#[derive(Module, Debug, Clone)]
pub struct Jepa<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub student_encoder: VisionTransformer<B>,
    pub teacher_encoder: VisionTransformer<B>,
    pub predictor: Predictor<B>,
    pub num_target_blocks: usize,
    pub target_scale_range: [f32; 2],
    pub target_aspect_ratio_range: [f32; 2],
    pub patch_size: usize,
    pub image_size: usize,
    pub var_reg_weight: f32,
}

#[derive(Debug, Clone)]
pub struct JepaStepOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> Jepa<B> {
    /// Returns a new model with the given student encoder.
    pub fn with_student_encoder(mut self, student_encoder: VisionTransformer<B>) -> Self {
        self.student_encoder = student_encoder;
        self
    }
    
    /// Forward pass for training/validation
    ///
    /// # Arguments
    /// * `batch` - Batch of images
    ///
    /// # Returns
    /// * `JepaStepOutput` containing the loss
    pub fn forward_step(&self, batch: JepaBatch<B>) -> JepaStepOutput<B> {
        let device = batch.images.device();
        let [batch_size, _, height, width] = batch.images.dims();

        // 1. Patchify images: [B, C, H, W] -> [B, N, D]
        let patches = self.patch_embed.forward(batch.images);
        let [_, num_patches, embed_dim] = patches.dims();

        // 2. Generate masks
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let masking_config = MaskingConfig {
            num_target_blocks: self.num_target_blocks,
            target_scale_range: self.target_scale_range,
            target_aspect_ratio_range: self.target_aspect_ratio_range,
        };
        let masks = sample_block_masks(batch_size, grid_h, grid_w, &masking_config, &device);

        // 3. Student encoder: process only context patches
        let context_repr =
            self.student_encoder
                .forward_context(patches.clone(), masks.context_indices.clone());

        // 4. Teacher encoder: process all patches (no grad)
        let full_repr = self.teacher_encoder.forward(patches.clone()).detach();

        // 5. Extract target representations
        let target_repr = self.gather_by_indices(full_repr, masks.target_indices.clone());

        // 6. Get position embeddings for context and target
        let context_pos_embed = self.gather_pos_embed(masks.context_indices.clone());
        let target_pos_embed = self.gather_pos_embed(masks.target_indices.clone());

        // 7. Predictor: predict target representations from context
        let predictions = self.predictor.forward(
            context_repr,
            context_pos_embed,
            target_pos_embed,
        );

        // 8. Compute loss with variance regularization
        let loss = jepa_loss(predictions, target_repr, self.var_reg_weight);

        JepaStepOutput { loss }
    }

    /// Gather position embeddings by indices
    ///
    /// Uses efficient batched gather operation instead of CPU loops.
    ///
    /// # Arguments
    /// * `indices` - Indices of patches to gather [B, N_indices]
    ///
    /// # Returns
    /// * Position embeddings at the specified indices [B, N_indices, D]
    fn gather_pos_embed(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, n_indices] = indices.dims();
        let pos_embed = self.patch_embed.pos_embed.val(); // [1, N, D]
        let embed_dim = pos_embed.dims()[2];

        // Broadcast position embeddings to batch size: [1, N, D] -> [B, N, D]
        let pos_embed_batched = pos_embed.repeat_dim(0, batch_size);

        // Expand indices to match output shape: [B, N_indices] -> [B, N_indices, D]
        // For gather, indices should match the output shape except for the gather dimension
        let indices_expanded = indices.unsqueeze_dim(2).repeat_dim(2, embed_dim);

        // Gather along dimension 1 (the patch dimension)
        // output[i, j, k] = input[i, indices[i, j, k], k] when dim=1
        pos_embed_batched.gather(1, indices_expanded)
    }

    /// Gather tensor values by indices along dimension 1
    ///
    /// Uses efficient batched gather operation instead of CPU loops.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor [B, N, D]
    /// * `indices` - Indices to gather [B, N_indices]
    ///
    /// # Returns
    /// * Gathered tensor values [B, N_indices, D]
    fn gather_by_indices(&self, tensor: Tensor<B, 3>, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch_size, _, embed_dim] = tensor.dims();
        let [_, n_indices] = indices.dims();

        // Expand indices to match output shape: [B, N_indices] -> [B, N_indices, D]
        let indices_expanded = indices.unsqueeze_dim(2).repeat_dim(2, embed_dim);

        // Gather along dimension 1 (the sequence dimension)
        // output[i, j, k] = input[i, indices[i, j, k], k] when dim=1
        tensor.gather(1, indices_expanded)
    }

    /// Update teacher encoder weights using EMA
    ///
    /// # Arguments
    /// * `momentum` - EMA momentum value (between 0 and 1)
    ///
    /// # Implementation
    ///
    /// This performs: `teacher = momentum * teacher + (1 - momentum) * student`
    ///
    /// We use Burn's record system to blend the parameters.
    pub fn ema_update(self, momentum: f64) -> Self {
        // Blend the teacher encoder with student encoder using EMA
        let updated_teacher_encoder = crate::model::ema::ema_update_params(
            &self.student_encoder,
            self.teacher_encoder,
            momentum,
        );

        // Return updated model
        Self {
            teacher_encoder: updated_teacher_encoder,
            ..self
        }
    }
}

impl<B: AutodiffBackend> TrainStep<JepaBatch<B>, JepaStepOutput<B>> for Jepa<B> {
    fn step(&self, batch: JepaBatch<B>) -> TrainOutput<JepaStepOutput<B>> {
        let item = self.forward_step(batch);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<JepaBatch<B>, JepaStepOutput<B>> for Jepa<B> {
    fn step(&self, batch: JepaBatch<B>) -> JepaStepOutput<B> {
        self.forward_step(batch)
    }
}

// JepaBatch moved here to avoid circular dependency
#[derive(Clone, Debug)]
pub struct JepaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
}
