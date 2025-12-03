//! Training configuration for JEPA.

use burn::config::Config;

/// # Training Configuration
///
/// Contains all hyperparameters for training JEPA.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Number of training epochs
    #[config(default = 100)]
    pub num_epochs: usize,

    /// Batch size for training
    #[config(default = 256)]
    pub batch_size: usize,

    /// Number of workers for data loading
    #[config(default = 4)]
    pub num_workers: usize,

    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,

    /// Learning rate
    #[config(default = 0.0001)]
    pub learning_rate: f64,

    /// AdamW weight decay
    #[config(default = 0.05)]
    pub weight_decay: f64,

    /// AdamW beta1
    #[config(default = 0.9)]
    pub beta1: f64,

    /// AdamW beta2
    #[config(default = 0.999)]
    pub beta2: f64,

    /// Gradient clipping max norm (0.0 = disabled)
    #[config(default = 1.0)]
    pub grad_clip_norm: f64,

    /// Warmup epochs for learning rate
    #[config(default = 10)]
    pub warmup_epochs: usize,

    /// Use cosine annealing schedule for learning rate
    #[config(default = true)]
    pub use_cosine_schedule: bool,

    /// Minimum learning rate for cosine annealing (as fraction of peak LR)
    #[config(default = 0.01)]
    pub min_lr_ratio: f64,

    /// Log interval (in steps)
    #[config(default = 100)]
    pub log_interval: usize,

    /// Checkpoint save interval (in epochs)
    #[config(default = 10)]
    pub checkpoint_interval: usize,
}

impl TrainingConfig {
    /// Get the checkpoint directory path
    pub fn checkpoint_dir(&self) -> String {
        "checkpoints".to_string()
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new()
    }
}
