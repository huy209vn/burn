//! # JEPA
//!
//! This crate provides a `burn` implementation of the Joint Embedding Predictive Architecture (JEPA).

// Making all modules public for now during scaffolding.
// We'll refine this later.
pub mod data;
pub mod model;
pub mod train;

// Re-export key types for convenience.
pub use model::jepa::{Jepa, JepaConfig, JepaBatch, JepaStepOutput};

#[cfg(test)]
pub mod test_utils;
