//! Custom callbacks for JEPA training.
//!
//! This module provides callbacks for handling EMA teacher updates during training.

use crate::model::ema::CosineAnnealingMomentum;
use crate::model::jepa::Jepa;
use burn::tensor::backend::AutodiffBackend;

/// # EMA Update Callback
///
/// Handles exponential moving average updates of the teacher encoder
/// after each training step, with momentum scheduling.
///
/// The momentum typically starts at a base value (e.g., 0.996) and
/// increases to 1.0 over the course of training using cosine annealing.
pub struct EmaUpdateCallback {
    /// Momentum scheduler
    pub momentum_scheduler: CosineAnnealingMomentum,
    /// Total number of training steps
    pub total_steps: usize,
    /// Current step counter
    pub current_step: usize,
}

impl EmaUpdateCallback {
    /// Create a new EMA update callback
    ///
    /// # Arguments
    /// * `base_momentum` - Starting momentum value (e.g., 0.996)
    /// * `end_momentum` - Final momentum value (typically 1.0)
    /// * `total_steps` - Total number of training steps
    pub fn new(base_momentum: f64, end_momentum: f64, total_steps: usize) -> Self {
        Self {
            momentum_scheduler: CosineAnnealingMomentum::new(base_momentum, end_momentum),
            total_steps,
            current_step: 0,
        }
    }

    /// Update the teacher encoder with EMA
    ///
    /// Should be called after each optimizer step.
    ///
    /// # Arguments
    /// * `model` - The JEPA model to update
    ///
    /// # Returns
    /// * Updated JEPA model with teacher encoder blended via EMA
    pub fn update_teacher<B: AutodiffBackend>(&mut self, model: Jepa<B>) -> Jepa<B> {
        // Get current momentum value from scheduler
        let momentum = self
            .momentum_scheduler
            .get_momentum(self.current_step, self.total_steps);

        // Update teacher encoder via EMA
        let updated_model = model.ema_update(momentum);

        // Increment step counter
        self.current_step += 1;

        updated_model
    }

    /// Get the current momentum value
    pub fn current_momentum(&self) -> f64 {
        self.momentum_scheduler
            .get_momentum(self.current_step, self.total_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_callback_momentum_progression() {
        let mut callback = EmaUpdateCallback::new(0.996, 1.0, 1000);

        // At start, momentum should be base
        assert_eq!(callback.current_step, 0);
        assert!((callback.current_momentum() - 0.996).abs() < 1e-6);

        // Simulate steps
        callback.current_step = 500;
        let mid_momentum = callback.current_momentum();
        assert!(mid_momentum > 0.996 && mid_momentum < 1.0);

        callback.current_step = 1000;
        assert!((callback.current_momentum() - 1.0).abs() < 1e-6);
    }
}
