//! The Vision Transformer (ViT) encoder module for JEPA.
//!
//! This module implements a configurable Vision Transformer encoder following
//! Burn's TransformerEncoder design patterns for flexibility and production use.

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{
    Gelu,
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// # Transformer Block Configuration
///
/// Configurable transformer block following Burn's design patterns.
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// MLP expansion ratio (hidden_dim = d_model * mlp_ratio)
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
    /// Dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Apply layer norm before sub-layers (pre-norm) instead of after (post-norm)
    #[config(default = true)]
    pub norm_first: bool,
    /// Use quiet softmax in attention
    #[config(default = false)]
    pub quiet_softmax: bool,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let mlp_hidden_dim = (self.d_model as f64 * self.mlp_ratio) as usize;

        let attention = MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
            .with_dropout(self.dropout)
            .with_quiet_softmax(self.quiet_softmax)
            .init(device);

        let norm1 = LayerNormConfig::new(self.d_model).init(device);
        let norm2 = LayerNormConfig::new(self.d_model).init(device);

        let mlp_fc1 = LinearConfig::new(self.d_model, mlp_hidden_dim)
            .with_bias(true)
            .init(device);
        let mlp_fc2 = LinearConfig::new(mlp_hidden_dim, self.d_model)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        TransformerBlock {
            attention,
            norm1,
            norm2,
            mlp_fc1,
            mlp_fc2,
            activation: Gelu::new(),
            dropout,
            norm_first: self.norm_first,
            d_model: self.d_model,
            n_heads: self.n_heads,
            mlp_ratio: self.mlp_ratio,
            dropout_rate: self.dropout,
        }
    }
}

/// # Transformer Block
///
/// A configurable transformer block with:
/// - Multi-head self-attention
/// - MLP with configurable activation
/// - Pre-norm or post-norm architecture
/// - Residual connections
/// - Optional dropout
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct TransformerBlock<B: Backend> {
    pub attention: MultiHeadAttention<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub mlp_fc1: Linear<B>,
    pub mlp_fc2: Linear<B>,
    pub activation: Gelu,
    pub dropout: Dropout,
    pub norm_first: bool,
    // Metadata for display
    pub d_model: usize,
    pub n_heads: usize,
    pub mlp_ratio: f64,
    pub dropout_rate: f64,
}

impl<B: Backend> ModuleDisplay for TransformerBlock<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("mlp_ratio", &self.mlp_ratio)
            .add("dropout", &self.dropout_rate)
            .add("norm_first", &self.norm_first)
            .optional()
    }
}

impl<B: Backend> TransformerBlock<B> {
    /// Forward pass through the transformer block
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, N, D]
    /// * `mask_pad` - Optional padding mask [B, N]
    /// * `mask_attn` - Optional attention mask [B, N, N]
    ///
    /// # Returns
    /// * Output tensor [B, N, D]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        // Multi-head attention residual path
        let mut residual_path = x.clone();

        // Pre-norm: normalize before attention
        if self.norm_first {
            residual_path = self.norm2.forward(residual_path);
        }

        // Multi-head self-attention
        let mut attn_input = MhaInput::self_attn(residual_path);
        if let Some(mask_pad) = mask_pad.clone() {
            attn_input = attn_input.mask_pad(mask_pad);
        }
        if let Some(mask_attn) = mask_attn.clone() {
            attn_input = attn_input.mask_attn(mask_attn);
        }
        let residual_path = self.attention.forward(attn_input).context;
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Post-norm: normalize after attention
        if !self.norm_first {
            x = self.norm2.forward(x);
        }

        // Feed-forward residual path
        let mut residual_path = x.clone();

        // Pre-norm: normalize before MLP
        if self.norm_first {
            residual_path = self.norm1.forward(residual_path);
        }

        // MLP
        let residual_path = self.mlp_fc1.forward(residual_path);
        let residual_path = self.activation.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let residual_path = self.mlp_fc2.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Post-norm: normalize after MLP
        if !self.norm_first {
            x = self.norm1.forward(x);
        }

        x
    }
}

/// # Vision Transformer Configuration
///
/// Configurable ViT encoder following Burn's TransformerEncoder design patterns.
#[derive(Config, Debug)]
pub struct VisionTransformerConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
    /// MLP expansion ratio
    #[config(default = 4.0)]
    pub mlp_ratio: f64,
    /// Apply layer norm before sub-layers (pre-norm) instead of after (post-norm)
    #[config(default = true)]
    pub norm_first: bool,
    /// Use quiet softmax in attention
    #[config(default = false)]
    pub quiet_softmax: bool,
}

impl VisionTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionTransformer<B> {
        let blocks: Vec<TransformerBlock<B>> = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(self.embed_dim, self.n_heads)
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_dropout(self.dropout)
                    .with_norm_first(self.norm_first)
                    .with_quiet_softmax(self.quiet_softmax)
                    .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        VisionTransformer {
            blocks,
            norm,
            embed_dim: self.embed_dim,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            dropout: self.dropout,
            mlp_ratio: self.mlp_ratio,
            norm_first: self.norm_first,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// # The Vision Transformer (ViT) Encoder.
///
/// A flexible, configurable ViT encoder following Burn's design patterns.
/// Supports various architectural choices for production use.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct VisionTransformer<B: Backend> {
    pub blocks: Vec<TransformerBlock<B>>,
    pub norm: LayerNorm<B>,
    // Metadata for display
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub dropout: f64,
    pub mlp_ratio: f64,
    pub norm_first: bool,
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for VisionTransformer<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("embed_dim", &self.embed_dim)
            .add("n_layers", &self.n_layers)
            .add("n_heads", &self.n_heads)
            .add("dropout", &self.dropout)
            .add("mlp_ratio", &self.mlp_ratio)
            .add("norm_first", &self.norm_first)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

/// Input struct for VisionTransformer forward pass
#[derive(Debug)]
pub struct VisionTransformerInput<B: Backend> {
    pub tensor: Tensor<B, 3>,
    pub mask_pad: Option<Tensor<B, 2, Bool>>,
    pub mask_attn: Option<Tensor<B, 3, Bool>>,
}

impl<B: Backend> VisionTransformerInput<B> {
    /// Create a new input
    pub fn new(tensor: Tensor<B, 3>) -> Self {
        Self {
            tensor,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Add padding mask
    pub fn mask_pad(mut self, mask_pad: Tensor<B, 2, Bool>) -> Self {
        self.mask_pad = Some(mask_pad);
        self
    }

    /// Add attention mask
    pub fn mask_attn(mut self, mask_attn: Tensor<B, 3, Bool>) -> Self {
        self.mask_attn = Some(mask_attn);
        self
    }
}

impl<B: Backend> VisionTransformer<B> {
    /// Forward pass through all patches (for teacher encoder)
    ///
    /// # Arguments
    /// * `patches` - Patch embeddings [B, N, D]
    ///
    /// # Returns
    /// * Encoded representations [B, N, D]
    pub fn forward(&self, mut patches: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pass through all transformer blocks
        for block in &self.blocks {
            patches = block.forward(patches, None, None);
        }

        // Final layer normalization
        self.norm.forward(patches)
    }

    /// Forward pass with input struct (for more complex scenarios)
    ///
    /// # Arguments
    /// * `input` - VisionTransformerInput with tensor and optional masks
    ///
    /// # Returns
    /// * Encoded representations [B, N, D]
    pub fn forward_with_input(&self, input: VisionTransformerInput<B>) -> Tensor<B, 3> {
        let mut x = input.tensor;

        // Pass through all transformer blocks with masks
        for block in &self.blocks {
            x = block.forward(x, input.mask_pad.clone(), input.mask_attn.clone());
        }

        // Final layer normalization
        self.norm.forward(x)
    }

    /// Forward pass with context masking (for student encoder)
    ///
    /// Extracts only the visible (context) patches and processes them.
    /// Uses efficient batched gather operation instead of CPU loops.
    ///
    /// # Arguments
    /// * `patches` - Full patch embeddings [B, N, D]
    /// * `context_indices` - Indices of context patches [B, N_ctx]
    ///
    /// # Returns
    /// * Encoded context representations [B, N_ctx, D]
    pub fn forward_context(
        &self,
        patches: Tensor<B, 3>,
        context_indices: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [_, _, embed_dim] = patches.dims();

        // Gather visible patches using batched gather operation
        // Expand indices from [B, N_ctx] to [B, N_ctx, D]
        let indices_expanded = context_indices.unsqueeze_dim(2).repeat_dim(2, embed_dim);

        // Gather context patches along dimension 1
        // output[i, j, k] = patches[i, context_indices[i, j, k], k]
        let mut context_patches = patches.gather(1, indices_expanded);

        // Pass through all transformer blocks
        for block in &self.blocks {
            context_patches = block.forward(context_patches, None, None);
        }

        // Final layer normalization
        self.norm.forward(context_patches)
    }
}
