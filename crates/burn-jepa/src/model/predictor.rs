//! The JEPA Predictor module.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use super::encoder::{TransformerBlock, TransformerBlockConfig};

/// # Predictor Configuration
#[derive(Config, Debug)]
pub struct PredictorConfig {
    pub embed_dim: usize,
    #[config(default = 384)]
    pub predictor_embed_dim: usize,
    #[config(default = 6)]
    pub n_layers: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
}

impl PredictorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Predictor<B> {
        // Learnable mask token for target positions
        let mask_token = Tensor::random(
            [1, 1, self.embed_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        // Transformer blocks for the predictor
        let blocks: Vec<TransformerBlock<B>> = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(self.embed_dim, self.n_heads)
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_dropout(self.dropout)
                    .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        Predictor {
            mask_token: Param::from_tensor(mask_token),
            blocks,
            norm,
        }
    }
}

/// # The JEPA Predictor Module.
///
/// Predicts target representations from context representations.
/// Uses a concatenation-based approach where mask tokens at target positions
/// attend to context representations through self-attention.
#[derive(Module, Debug)]
pub struct Predictor<B: Backend> {
    pub mask_token: Param<Tensor<B, 3>>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub norm: LayerNorm<B>,
}

impl<B: Backend> Predictor<B> {
    /// Forward pass: predict target representations from context
    ///
    /// # Arguments
    /// * `context_repr` - Context representations [B, N_ctx, D]
    /// * `context_pos_embed` - Position embeddings for context [B, N_ctx, D]
    /// * `target_pos_embed` - Position embeddings for targets [B, N_tgt, D]
    ///
    /// # Returns
    /// * Predicted target representations [B, N_tgt, D]
    pub fn forward(
        &self,
        context_repr: Tensor<B, 3>,
        context_pos_embed: Tensor<B, 3>,
        target_pos_embed: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch_size, n_ctx, _] = context_repr.dims();
        let n_tgt = target_pos_embed.dims()[1];

        // Add position embeddings to context representations
        let context_with_pos = context_repr + context_pos_embed;

        // Create mask tokens for target positions: [B, N_tgt, D]
        let target_queries = self
            .mask_token
            .val()
            .repeat_dim(0, batch_size)
            .repeat_dim(1, n_tgt)
            + target_pos_embed;

        // Concatenate context and target queries: [B, N_ctx + N_tgt, D]
        let mut x = Tensor::cat(vec![context_with_pos, target_queries], 1);

        // Pass through transformer blocks
        // Target queries attend to context through self-attention
        for block in &self.blocks {
            x = block.forward(x, None, None);
        }

        // Apply final layer norm
        x = self.norm.forward(x);

        // Extract predictions for target positions (last N_tgt tokens)
        x.slice([0..batch_size, n_ctx..(n_ctx + n_tgt)])
    }
}