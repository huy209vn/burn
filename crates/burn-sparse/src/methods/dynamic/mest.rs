//! MEST: Momentum-based dynamic sparse training

use burn_core::tensor::{backend::Backend, Tensor};

use crate::core::SparseMask;

/// Configuration for MEST
#[derive(Debug, Clone)]
pub struct MestConfig {
    /// Target sparsity
    pub sparsity: f32,

    /// Update frequency
    pub update_frequency: usize,
}

impl Default for MestConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.8,
            update_frequency: 100,
        }
    }
}

/// MEST: Momentum-based Efficient Sparse Training
pub struct Mest<B: Backend> {
    config: MestConfig,
    mask: SparseMask<B>,
}

impl<B: Backend> Mest<B> {
    /// Create new MEST trainer
    pub fn new(config: MestConfig, initial_mask: SparseMask<B>) -> Self {
        Self { config, mask: initial_mask }
    }

    /// Update mask based on momentum
    pub fn update_mask(&mut self, _momentum: &Tensor<B, 2>) -> SparseMask<B> {
        // TODO: Implement
        self.mask.clone()
    }
}
