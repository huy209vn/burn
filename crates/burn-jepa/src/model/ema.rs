//! Exponential Moving Average (EMA) update logic and momentum scheduling.
//!
//! This file defines the `CosineAnnealingMomentum` schedule and a helper
//! function to apply EMA updates to module parameters.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::collections::HashMap;
use std::f64::consts::PI;

/// # Cosine Annealing Momentum Scheduler.
///
/// Implements a cosine annealing schedule for EMA momentum.
/// The momentum value typically starts lower (`base_momentum`) and
/// gradually increases to `end_momentum` over the training duration.
pub struct CosineAnnealingMomentum {
    /// The starting momentum value.
    pub base_momentum: f64,
    /// The final momentum value.
    pub end_momentum: f64,
}

impl CosineAnnealingMomentum {
    pub fn new(base_momentum: f64, end_momentum: f64) -> Self {
        Self {
            base_momentum,
            end_momentum,
        }
    }

    /// Calculates the EMA momentum for a given training step using a cosine schedule.
    ///
    /// The formula used is: `m(t) = end_momentum - (end_momentum - base_momentum) * (cos(PI * t / T) + 1) / 2`
    /// where `t` is the current step and `T` is the total number of steps.
    ///
    /// This results in momentum starting at `base_momentum` and smoothly increasing to `end_momentum`.
    pub fn get_momentum(&self, step: usize, total_steps: usize) -> f64 {
        if total_steps == 0 || step >= total_steps {
            return self.end_momentum;
        }
        let t = step as f64;
        let T = total_steps as f64;
        let cos_term = (PI * t / T).cos();
        let momentum =
            self.end_momentum - (self.end_momentum - self.base_momentum) * (cos_term + 1.0) / 2.0;
        momentum
    }
}

/// Visitor to collect student module parameters by their IDs
struct StudentParamCollector<B: Backend> {
    params: HashMap<ParamId, Box<dyn std::any::Any>>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> StudentParamCollector<B> {
    fn new() -> Self {
        Self {
            params: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for StudentParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let id = param.id;
        let tensor = param.val().clone();
        self.params.insert(id, Box::new(tensor));
    }
}

/// Mapper to apply EMA updates to teacher parameters
struct EmaUpdateMapper<B: Backend> {
    student_params: HashMap<ParamId, Box<dyn std::any::Any>>,
    momentum: f64,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> EmaUpdateMapper<B> {
    fn new(student_params: HashMap<ParamId, Box<dyn std::any::Any>>, momentum: f64) -> Self {
        Self {
            student_params,
            momentum,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleMapper<B> for EmaUpdateMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, teacher_tensor, mapper) = param.consume();

        // Try to find corresponding student parameter
        if let Some(student_any) = self.student_params.get(&id) {
            // Try to downcast to the correct tensor type
            if let Some(student_tensor) = student_any.downcast_ref::<Tensor<B, D>>() {
                // Apply EMA: teacher = momentum * teacher + (1 - momentum) * student
                let updated = teacher_tensor.mul_scalar(self.momentum as f32)
                    + student_tensor.clone().mul_scalar((1.0 - self.momentum) as f32);
                return Param::from_mapped_value(id, updated, mapper);
            }
        }

        // If no student parameter found or type mismatch, return teacher unchanged
        Param::from_mapped_value(id, teacher_tensor, mapper)
    }
}

/// # Performs an Exponential Moving Average (EMA) update on module parameters.
///
/// This function updates the parameters of the `teacher_module` towards the
/// `student_module`'s parameters using the provided `momentum`.
///
/// The update formula is: `teacher = momentum * teacher + (1 - momentum) * student`
///
/// # Implementation
///
/// This uses Burn's `ModuleVisitor` to collect student parameters by ID,
/// then uses `ModuleMapper` to update teacher parameters accordingly.
///
/// # Arguments
///
/// * `student_module`: The module whose parameters are used as the source for the EMA.
/// * `teacher_module`: The module whose parameters will be updated.
/// * `momentum`: The EMA momentum value (typically between 0.99 and 1.0).
///
/// # Returns
///
/// Updated teacher module with EMA-blended parameters.
///
/// # Example
///
/// ```ignore
/// use burn_jepa::model::ema::ema_update_params;
///
/// let momentum = 0.996;
/// teacher_encoder = ema_update_params(&student_encoder, teacher_encoder, momentum);
/// ```
pub fn ema_update_params<B: Backend, M: Module<B>>(
    student_module: &M,
    teacher_module: M,
    momentum: f64,
) -> M {
    // Step 1: Collect all student parameters by ID using a visitor
    let mut collector = StudentParamCollector::<B>::new();
    student_module.visit(&mut collector);

    // Step 2: Apply EMA update to teacher using a mapper
    let mut mapper = EmaUpdateMapper::new(collector.params, momentum);
    teacher_module.map(&mut mapper)
}
