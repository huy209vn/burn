//! # Data Loading and Processing
//!
//! This module handles all aspects of data loading, augmentation, and preprocessing for the
//! JEPA model. It is designed to be flexible and efficient, allowing for easy integration
//! of different datasets and data augmentation strategies.
//!
//! ## Modules
//!
//! - **`datasets`**: Provides dataset implementations, such as for ImageNet-like datasets,
//!   that handle loading images from disk and applying basic transformations. It also
//!   includes the collation logic for batching samples and generating shared masks.
//!
//! - **`augment`**: Implements a variety of data augmentation techniques, including standard
//!   augmentations like random cropping, flipping, and color jittering. It also supports
//!   more advanced strategies like multi-view augmentation, which can be useful for
//!   certain self-supervised learning methods.

pub mod augment;
pub mod datasets;