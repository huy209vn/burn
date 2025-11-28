/// Module system integration for SparseTensor parameters
///
/// Uses newtype pattern to avoid orphan rule issues: SparseParam<B> wraps Param<SparseTensor<B>>.
/// This enables sparse tensors to be used as trainable parameters in neural networks.
///
/// The implementation decomposes SparseTensors into their component tensors
/// (values, indices) for visiting/mapping, then reassembles them.

use crate::core::{SparseTensor, SparseTensorData};
use burn_core::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param, ParamId, Parameter,
};
use burn_core::record::{PrecisionSettings, Record};
use burn_core::tensor::backend::{AutodiffBackend, Backend};
use burn_core::tensor::ops::Device;
use burn_core::tensor::{Tensor, TensorData};
use alloc::{format, string::ToString, vec::Vec};
use core::ops::{Deref, DerefMut};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Newtype wrapper for sparse tensor parameters
///
/// This wrapper enables SparseTensor to work with Burn's Module system
/// while respecting Rust's orphan rules.
#[derive(Debug, Clone)]
pub struct SparseParam<B: Backend> {
    inner: Param<SparseTensor<B>>,
}

/// Serializable representation of SparseParam for Record system
#[derive(Serialize, Deserialize)]
struct SparseParamData {
    format: crate::core::SparseFormat,
    shape: [usize; 2],
    // Serialize as raw data to avoid backend-specific issues
    values_data: Vec<f32>,
    col_indices_data: Option<Vec<i64>>,
    row_indices_data: Option<Vec<i64>>,
    row_pointers_data: Option<Vec<i64>>,
    col_pointers_data: Option<Vec<i64>>,
    mask_data: Option<Vec<bool>>,
    // For BlockCSR
    blocks_data: Option<Vec<f32>>,
    block_col_indices_data: Option<Vec<i64>>,
    block_row_pointers_data: Option<Vec<i64>>,
    block_size: Option<usize>,
    // For N:M
    metadata_data: Option<Vec<i64>>,
    n: Option<usize>,
    m: Option<usize>,
}

impl<B: Backend> Serialize for SparseParam<B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let sparse = self.val();
        let data = match sparse.data() {
            SparseTensorData::CSR { values, col_indices, row_pointers } => {
                SparseParamData {
                    format: sparse.format(),
                    shape: sparse.shape(),
                    values_data: values.to_data().to_vec().unwrap(),
                    col_indices_data: Some(col_indices.to_data().to_vec().unwrap()),
                    row_pointers_data: Some(row_pointers.to_data().to_vec().unwrap()),
                    row_indices_data: None,
                    col_pointers_data: None,
                    mask_data: None,
                    blocks_data: None,
                    block_col_indices_data: None,
                    block_row_pointers_data: None,
                    block_size: None,
                    metadata_data: None,
                    n: None,
                    m: None,
                }
            }
            _ => todo!("Serialization for other formats not yet implemented"),
        };
        data.serialize(serializer)
    }
}

impl<'de, B: Backend> Deserialize<'de> for SparseParam<B> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = SparseParamData::deserialize(deserializer)?;

        // Reconstruct on default device for now
        let device = B::Device::default();

        let sparse = match data.format {
            crate::core::SparseFormat::CSR => {
                let values_len = data.values_data.len();
                let col_indices_len = data.col_indices_data.as_ref().unwrap().len();
                let row_pointers_len = data.row_pointers_data.as_ref().unwrap().len();

                let values = Tensor::from_data(
                    TensorData::new(data.values_data, [values_len]),
                    &device
                );
                let col_indices = Tensor::from_data(
                    TensorData::new(data.col_indices_data.unwrap(), [col_indices_len]),
                    &device
                );
                let row_pointers = Tensor::from_data(
                    TensorData::new(data.row_pointers_data.unwrap(), [row_pointers_len]),
                    &device
                );
                SparseTensor::from_csr(values, col_indices, row_pointers, data.shape, device)
            }
            _ => return Err(serde::de::Error::custom("Only CSR format supported for deserialization")),
        };

        Ok(Self::from_sparse(sparse))
    }
}

impl<B: Backend> SparseParam<B> {
    /// Create a new sparse parameter from a sparse tensor.
    pub fn from_sparse(value: SparseTensor<B>) -> Self {
        let value_with_grad = Parameter::set_require_grad(value, true);
        Self {
            inner: Param::initialized(ParamId::new(), value_with_grad),
        }
    }

    /// Create from existing Param
    pub fn from_param(param: Param<SparseTensor<B>>) -> Self {
        Self { inner: param }
    }

    /// Get the underlying Param
    pub fn into_inner(self) -> Param<SparseTensor<B>> {
        self.inner
    }

    /// Get parameter ID
    pub fn id(&self) -> ParamId {
        self.inner.id
    }

    /// Get value
    pub fn val(&self) -> SparseTensor<B> {
        self.inner.val()
    }

    /// Get shape
    pub fn shape(&self) -> [usize; 2] {
        self.val().shape()
    }
}

// Deref to allow direct access to inner Param methods
impl<B: Backend> Deref for SparseParam<B> {
    type Target = Param<SparseTensor<B>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<B: Backend> DerefMut for SparseParam<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// Record trait - required for serialization
impl<B: Backend> Record<B> for SparseParam<B> {
    type Item<S: PrecisionSettings> = SparseParam<B>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

/// Module implementation - enables SparseTensor to be used in neural network modules
impl<B: Backend> Module<B> for SparseParam<B> {
    type Record = SparseParam<B>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        // Decompose sparse tensor and visit its components
        let sparse = self.val();

        match sparse.data() {
            SparseTensorData::CSR { values, .. } => {
                // Visit values tensor (the only trainable part)
                let values_param = Param::initialized(self.id, values.clone());
                visitor.visit_float(&values_param);
            }
            SparseTensorData::COO { values, .. } => {
                let values_param = Param::initialized(self.id, values.clone());
                visitor.visit_float(&values_param);
            }
            SparseTensorData::CSC { values, .. } => {
                let values_param = Param::initialized(self.id, values.clone());
                visitor.visit_float(&values_param);
            }
            SparseTensorData::Mask { values, .. } => {
                let values_param = Param::initialized(self.id, values.clone());
                visitor.visit_float(&values_param);
            }
            SparseTensorData::BlockCSR { blocks, .. } => {
                let blocks_param = Param::initialized(self.id, blocks.clone());
                visitor.visit_float(&blocks_param);
            }
            SparseTensorData::NInM { values, .. } => {
                let values_param = Param::initialized(self.id, values.clone());
                visitor.visit_float(&values_param);
            }
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let mapped_inner = self.inner.map(|sparse| {
            // Decompose, map values, reassemble
            let data = match sparse.data().clone() {
                SparseTensorData::CSR {
                    values,
                    col_indices,
                    row_pointers,
                } => {
                    let values_param = Param::initialized(ParamId::new(), values);
                    let mapped_values = mapper.map_float(values_param).val();
                    SparseTensorData::CSR {
                        values: mapped_values,
                        col_indices,
                        row_pointers,
                    }
                }
                SparseTensorData::COO {
                    values,
                    row_indices,
                    col_indices,
                } => {
                    let values_param = Param::initialized(ParamId::new(), values);
                    let mapped_values = mapper.map_float(values_param).val();
                    SparseTensorData::COO {
                        values: mapped_values,
                        row_indices,
                        col_indices,
                    }
                }
                SparseTensorData::CSC {
                    values,
                    row_indices,
                    col_pointers,
                } => {
                    let values_param = Param::initialized(ParamId::new(), values);
                    let mapped_values = mapper.map_float(values_param).val();
                    SparseTensorData::CSC {
                        values: mapped_values,
                        row_indices,
                        col_pointers,
                    }
                }
                SparseTensorData::Mask { values, mask } => {
                    let values_param = Param::initialized(ParamId::new(), values);
                    let mapped_values = mapper.map_float(values_param).val();
                    SparseTensorData::Mask {
                        values: mapped_values,
                        mask,
                    }
                }
                SparseTensorData::BlockCSR {
                    blocks,
                    block_col_indices,
                    block_row_pointers,
                    block_size,
                } => {
                    let blocks_param = Param::initialized(ParamId::new(), blocks);
                    let mapped_blocks = mapper.map_float(blocks_param).val();
                    SparseTensorData::BlockCSR {
                        blocks: mapped_blocks,
                        block_col_indices,
                        block_row_pointers,
                        block_size,
                    }
                }
                SparseTensorData::NInM {
                    values,
                    metadata,
                    n,
                    m,
                } => {
                    let values_param = Param::initialized(ParamId::new(), values);
                    let mapped_values = mapper.map_float(values_param).val();
                    SparseTensorData::NInM {
                        values: mapped_values,
                        metadata,
                        n,
                        m,
                    }
                }
            };

            SparseTensor::from_data(data, sparse.format(), sparse.shape(), sparse.device())
        });
        Self {
            inner: mapped_inner,
        }
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record
    }

    fn to_device(self, device: &Device<B>) -> Self {
        let inner = self.inner.map(|tensor| tensor.to_device(device));
        Self { inner }
    }

    fn fork(self, device: &Device<B>) -> Self {
        let inner = self.inner.map(|tensor| {
            let is_require_grad = Parameter::is_require_grad(&tensor);
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = Parameter::set_require_grad(tensor, true);
            }

            tensor
        });
        Self { inner }
    }

    fn collect_devices(&self, mut devices: alloc::vec::Vec<Device<B>>) -> alloc::vec::Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<B: Backend> ModuleDisplayDefault for SparseParam<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id())
        } else {
            "".to_string()
        };
        let sparse = self.val();
        let string = format!(
            "ParamSparseTensor {{shape: {:?}, format: {:?}, nnz: {}, sparsity: {:.2}%{id}}}",
            sparse.shape(),
            sparse.format(),
            sparse.nnz(),
            sparse.sparsity() * 100.0
        );
        content.add_formatted(&string).optional()
    }
}

impl<B: Backend> ModuleDisplay for SparseParam<B> {}

impl<B: AutodiffBackend> AutodiffModule<B> for SparseParam<B> {
    type InnerModule = SparseParam<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        // Get inner backend sparse tensor without gradients
        let sparse = self.val();
        let inner_sparse = sparse.inner();
        let inner_sparse_no_grad = Parameter::set_require_grad(inner_sparse, false);
        let inner_param = Param::initialized(self.id(), inner_sparse_no_grad);
        SparseParam {
            inner: inner_param,
        }
    }
}
