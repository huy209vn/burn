//! Training engine utilities for JEPA.
//!
//! This module provides helper functions for setting up JEPA training,
//! including optimizer configuration and training step calculations.

use super::config::TrainingConfig;
use burn::optim::AdamWConfig;

/// # Create AdamW Optimizer Configuration
///
/// Creates an AdamW optimizer config with the given training configuration.
/// AdamW is the recommended optimizer for JEPA training.
///
/// # Arguments
/// * `config` - Training configuration
///
/// # Returns
/// * Configured AdamW optimizer
///
/// # Example
/// ```ignore
/// let train_config = TrainingConfig::new()
///     .with_learning_rate(0.0001)
///     .with_weight_decay(0.05);
/// let optimizer_config = create_optimizer(&train_config);
/// let optimizer = optimizer_config.init();
/// ```
pub fn create_optimizer(config: &TrainingConfig) -> AdamWConfig {
    AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32)
        .with_epsilon(1e-8)
}

/// # Calculate Total Training Steps
///
/// Computes the total number of training steps given the dataset size,
/// batch size, and number of epochs.
///
/// # Arguments
/// * `num_samples` - Total number of training samples
/// * `batch_size` - Batch size
/// * `num_epochs` - Number of training epochs
///
/// # Returns
/// * Total number of training steps
///
/// # Example
/// ```
/// use burn_jepa::train::engine::calculate_total_steps;
///
/// // 1000 samples, batch size 100, 10 epochs = 100 steps
/// let total_steps = calculate_total_steps(1000, 100, 10);
/// assert_eq!(total_steps, 100);
/// ```
pub fn calculate_total_steps(num_samples: usize, batch_size: usize, num_epochs: usize) -> usize {
    let steps_per_epoch = (num_samples + batch_size - 1) / batch_size; // Ceiling division
    steps_per_epoch * num_epochs
}

/// # Calculate Warmup Steps
///
/// Computes the number of warmup steps given warmup epochs.
///
/// # Arguments
/// * `num_samples` - Total number of training samples
/// * `batch_size` - Batch size
/// * `warmup_epochs` - Number of warmup epochs
///
/// # Returns
/// * Number of warmup steps
pub fn calculate_warmup_steps(
    num_samples: usize,
    batch_size: usize,
    warmup_epochs: usize,
) -> usize {
    calculate_total_steps(num_samples, batch_size, warmup_epochs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_optimizer() {
        let config = TrainingConfig::new()
            .with_learning_rate(0.0001)
            .with_weight_decay(0.05)
            .with_beta1(0.9)
            .with_beta2(0.999);

        let _optimizer_config = create_optimizer(&config);

        // Verify optimizer is created successfully
        // (We can't access private fields, but creation itself validates the config)
    }

    #[test]
    fn test_calculate_total_steps() {
        // Test exact division
        assert_eq!(calculate_total_steps(1000, 100, 10), 100);

        // Test ceiling division
        assert_eq!(calculate_total_steps(1050, 100, 10), 110);

        // Test single epoch
        assert_eq!(calculate_total_steps(256, 32, 1), 8);
    }

    #[test]
    fn test_calculate_warmup_steps() {
        // 1000 samples, batch 100, 2 warmup epochs = 20 steps
        assert_eq!(calculate_warmup_steps(1000, 100, 2), 20);

        // With ceiling division
        assert_eq!(calculate_warmup_steps(1050, 100, 2), 22);
    }
}
