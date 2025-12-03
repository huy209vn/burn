//! Dataset definitions and data loading utilities for JEPA.

use burn::data::dataset::{Dataset, InMemDataset};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::marker::PhantomData;


/// # Represents a single image item.
/// This is the output of the `Dataset::get()` method.
#[derive(Clone, Debug)]
pub struct ImageItem<B: Backend> {
    /// The image tensor, typically `[C, H, W]`.
    pub image: Tensor<B, 3>,
}

/// # JEPA-specific Dataset Wrapper.
///
/// This struct wraps an existing dataset and applies JEPA-specific
/// preprocessing and data augmentation.
pub struct JepaDataset<B: Backend, D: Dataset<ImageItem<B>>> {
    /// The underlying dataset providing raw image items.
    pub dataset: D,
    /// The data augmentation pipeline to apply to each image.
    // pub augmentor: JepaDataAugmentation<B>,
    pub _backend: PhantomData<B>,
}

impl<B: Backend, D: Dataset<ImageItem<B>>> Dataset<ImageItem<B>> for JepaDataset<B, D> {
    fn get(&self, index: usize) -> Option<ImageItem<B>> {
        // TODO: Get item and apply augmentations.
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
