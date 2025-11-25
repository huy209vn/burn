//! Experimental features for burn-sparse
//!
//! Features in this module are:
//! - **Unstable**: APIs may change
//! - **Production-ready**: Thoroughly tested, but evolving
//! - **Feature-gated**: Require `experimental` feature flag
//!
//! # Available Modules
//!
//! - [`sparseram`]: Runtime memory tiering for sparse models (GPU/RAM/Disk)

pub mod sparseram;
