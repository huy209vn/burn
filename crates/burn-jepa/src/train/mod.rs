//! # Training
//!
//! This module contains the logic for training the JEPA model. It provides a structured
//! and reusable training framework that can be easily adapted to different experiments
//! and configurations.
//!
//! ## Modules
//!
//! - **`config`**: Training configuration with hyperparameters.
//!
//! - **`engine`**: The core training engine, which implements the main training loop. It
//!   handles the forward and backward passes, optimizer steps, checkpointing, and logging.
//!
//! - **`scheduler`**: Implements learning rate and momentum schedulers. This includes
//!   standard learning rate schedulers like cosine annealing, as well as the momentum
//!   schedule for the teacher network's EMA updates.
//!
//! - **`callback`**: Custom callbacks for the training learner, e.g., for EMA updates.

pub mod config;
pub mod engine;
pub mod scheduler;
pub mod callback;

pub use config::TrainingConfig;
