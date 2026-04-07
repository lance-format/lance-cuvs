// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::compute::filter;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt8Array, UInt32Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use cuvs::Resources;
use futures::{FutureExt, TryStreamExt, future::LocalBoxFuture};
use lance::dataset::Dataset;
use lance::index::vector::PartitionArtifactBuilder;
use lance::index::vector::utils::{infer_vector_dim, vector_column_to_fsl};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, ROW_ID, Result};
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_index::vector::utils::is_finite;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_linalg::distance::DistanceType;
use log::warn;
use ndarray::{Array2, Array3, ArrayView2};
#[cfg(feature = "python")]
use numpy::{PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyModule};
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

const PARTITION_ARTIFACT_METADATA_FILE_NAME: &str = "metadata.lance";
const PARTITION_ARTIFACT_FILE_VERSION: &str = "2.2";
const PIPELINE_SLOTS: usize = 2;

type CudaEventHandle = *mut c_void;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cuvs_sys::cudaError_t;
    fn cudaFreeHost(ptr: *mut c_void) -> cuvs_sys::cudaError_t;
    fn cudaEventCreate(event: *mut CudaEventHandle) -> cuvs_sys::cudaError_t;
    fn cudaEventDestroy(event: CudaEventHandle) -> cuvs_sys::cudaError_t;
    fn cudaEventRecord(
        event: CudaEventHandle,
        stream: cuvs_sys::cudaStream_t,
    ) -> cuvs_sys::cudaError_t;
    fn cudaEventSynchronize(event: CudaEventHandle) -> cuvs_sys::cudaError_t;
}

pub struct TrainedIvfPqIndex {
    resources: Resources,
    index: CuvsIvfPqIndex,
    num_partitions: usize,
    dimension: usize,
    num_sub_vectors: usize,
    num_bits: usize,
    metric_type: DistanceType,
    ivf_centroids: FixedSizeListArray,
    pq_codebook: FixedSizeListArray,
}

impl TrainedIvfPqIndex {
    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        &self.ivf_centroids
    }

    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        &self.pq_codebook
    }

    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    pub fn pq_code_width(&self) -> usize {
        self.num_sub_vectors
    }

    pub fn metric_type(&self) -> DistanceType {
        self.metric_type
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

#[derive(Clone)]
pub struct VectorIndexBuildParams {
    pub column: String,
    pub kind: VectorIndexKind,
    pub artifact_uri: String,
    pub batch_size: usize,
    pub filter_nan: bool,
}

#[derive(Clone)]
pub enum VectorIndexKind {
    IvfPq(IvfPqBuildParams),
}

#[derive(Clone)]
pub struct IvfPqBuildParams {
    pub num_partitions: usize,
    pub metric_type: DistanceType,
    pub num_sub_vectors: usize,
    pub sample_rate: usize,
    pub max_iters: usize,
    pub num_bits: usize,
}

pub enum VectorIndexBuildOutput {
    PartitionArtifact(PartitionArtifactBuildOutput),
}

impl VectorIndexBuildOutput {
    pub fn artifact_uri(&self) -> &str {
        match self {
            Self::PartitionArtifact(output) => &output.artifact_uri,
        }
    }

    pub fn files(&self) -> &[String] {
        match self {
            Self::PartitionArtifact(output) => &output.files,
        }
    }

    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.ivf_centroids,
        }
    }

    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.pq_codebook,
        }
    }
}

pub struct PartitionArtifactBuildOutput {
    artifact_uri: String,
    files: Vec<String>,
    ivf_centroids: FixedSizeListArray,
    pq_codebook: FixedSizeListArray,
}

pub trait VectorBuildBackend {
    fn build<'a>(
        &'a self,
        dataset: &'a Dataset,
        params: VectorIndexBuildParams,
    ) -> LocalBoxFuture<'a, Result<VectorIndexBuildOutput>>;
}

pub struct CuvsVectorBuildBackend;

impl VectorBuildBackend for CuvsVectorBuildBackend {
    fn build<'a>(
        &'a self,
        dataset: &'a Dataset,
        params: VectorIndexBuildParams,
    ) -> LocalBoxFuture<'a, Result<VectorIndexBuildOutput>> {
        async move {
            match params.kind {
                VectorIndexKind::IvfPq(build_params) => {
                    let trained = train_ivf_pq(
                        dataset,
                        &params.column,
                        build_params.num_partitions,
                        build_params.metric_type,
                        build_params.num_sub_vectors,
                        build_params.sample_rate,
                        build_params.max_iters,
                        build_params.num_bits,
                        params.filter_nan,
                    )
                    .await?;
                    let files = assign_ivf_pq_to_artifact(
                        dataset,
                        &params.column,
                        &trained,
                        &params.artifact_uri,
                        params.batch_size,
                        params.filter_nan,
                    )
                    .await?;
                    Ok(VectorIndexBuildOutput::PartitionArtifact(
                        PartitionArtifactBuildOutput {
                            artifact_uri: params.artifact_uri,
                            files,
                            ivf_centroids: trained.ivf_centroids.clone(),
                            pq_codebook: trained.pq_codebook.clone(),
                        },
                    ))
                }
            }
        }
        .boxed_local()
    }
}

struct CuvsIvfPqIndex {
    raw: cuvs_sys::cuvsIvfPqIndex_t,
}

impl CuvsIvfPqIndex {
    fn try_new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsIvfPqIndexCreate(&mut raw) },
            "create IVF_PQ index",
        )?;
        Ok(Self { raw })
    }
}

impl Drop for CuvsIvfPqIndex {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cuvs_sys::cuvsIvfPqIndexDestroy(self.raw) };
        }
    }
}

enum MatrixBuffer<'a> {
    Borrowed {
        values: &'a [f32],
        rows: usize,
        cols: usize,
    },
    Owned(Array2<f32>),
}

impl MatrixBuffer<'_> {
    fn view(&self) -> Result<ArrayView2<'_, f32>> {
        match self {
            Self::Borrowed { values, rows, cols } => ArrayView2::from_shape((*rows, *cols), values)
                .map_err(|error| {
                    Error::io(format!("failed to create borrowed matrix view: {error}"))
                }),
            Self::Owned(array) => Ok(array.view()),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Borrowed { rows, .. } => *rows,
            Self::Owned(array) => array.nrows(),
        }
    }
}

struct HostTensorView {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
}

impl HostTensorView {
    fn try_new<T: DlElement>(shape: &[usize], data: *mut std::ffi::c_void) -> Self {
        let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice {
                    device_type: cuvs_sys::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: T::dl_dtype(),
                shape: shape.as_ptr() as *mut i64,
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        Self { shape, tensor }
    }

    fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }
}

trait DlElement: Copy + Default {
    fn dl_dtype() -> cuvs_sys::DLDataType;
}

impl DlElement for f32 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLFloat as u8,
            bits: 32,
            lanes: 1,
        }
    }
}

impl DlElement for u8 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLUInt as u8,
            bits: 8,
            lanes: 1,
        }
    }
}

impl DlElement for u32 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLUInt as u8,
            bits: 32,
            lanes: 1,
        }
    }
}

struct DeviceTensor<T: DlElement> {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
    capacity_bytes: usize,
    resources: cuvs_sys::cuvsResources_t,
    _marker: PhantomData<T>,
}

impl<T: DlElement> DeviceTensor<T> {
    fn try_new(resources: &Resources, shape: &[usize]) -> Result<Self> {
        let capacity_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let mut data = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsRMMAlloc(resources.0, &mut data, capacity_bytes) },
            "allocate device tensor",
        )?;
        let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice {
                    device_type: cuvs_sys::DLDeviceType::kDLCUDA,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: T::dl_dtype(),
                shape: shape.as_ptr() as *mut i64,
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        Ok(Self {
            shape,
            tensor,
            capacity_bytes,
            resources: resources.0,
            _marker: PhantomData,
        })
    }

    fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }

    fn set_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.len() != self.shape.len() {
            return Err(Error::io(format!(
                "device tensor rank mismatch: expected {}, got {}",
                self.shape.len(),
                shape.len()
            )));
        }
        let required_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if required_bytes > self.capacity_bytes {
            return Err(Error::io(format!(
                "device tensor capacity {} bytes is smaller than requested shape {:?} ({} bytes)",
                self.capacity_bytes, shape, required_bytes
            )));
        }
        for (dst, src) in self.shape.iter_mut().zip(shape) {
            *dst = *src as i64;
        }
        Ok(())
    }

    fn current_len(&self) -> usize {
        self.shape.iter().map(|dim| *dim as usize).product()
    }

    fn current_bytes(&self) -> usize {
        self.current_len() * std::mem::size_of::<T>()
    }

    fn copy_from_host_async(&mut self, resources: &Resources, src: &[T]) -> Result<()> {
        let expected_len = self.current_len();
        if src.len() != expected_len {
            return Err(Error::io(format!(
                "device tensor copy expects {expected_len} elements, got {}",
                src.len()
            )));
        }
        check_cuda(
            unsafe {
                cuvs_sys::cudaMemcpyAsync(
                    self.tensor.dl_tensor.data,
                    src.as_ptr() as *const _,
                    self.current_bytes(),
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    resources
                        .get_cuda_stream()
                        .map_err(|e| Error::io(e.to_string()))?,
                )
            },
            "copy host tensor to device",
        )
    }

    fn copy_to_host_async(&self, resources: &Resources, dst: &mut [T]) -> Result<()> {
        let expected_len = self.current_len();
        if dst.len() != expected_len {
            return Err(Error::io(format!(
                "device tensor copy expects destination length {expected_len}, got {}",
                dst.len()
            )));
        }
        check_cuda(
            unsafe {
                cuvs_sys::cudaMemcpyAsync(
                    dst.as_mut_ptr() as *mut _,
                    self.tensor.dl_tensor.data,
                    self.current_bytes(),
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    resources
                        .get_cuda_stream()
                        .map_err(|e| Error::io(e.to_string()))?,
                )
            },
            "copy device tensor to host",
        )
    }
}

impl<T: DlElement> Drop for DeviceTensor<T> {
    fn drop(&mut self) {
        if !self.tensor.dl_tensor.data.is_null() {
            let _ = unsafe {
                cuvs_sys::cuvsRMMFree(
                    self.resources,
                    self.tensor.dl_tensor.data,
                    self.capacity_bytes,
                )
            };
        }
    }
}

struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> PinnedHostBuffer<T> {
    fn try_new(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::io("pinned host allocation size overflow"))?;
        let mut raw = ptr::null_mut();
        check_cuda(
            unsafe { cudaMallocHost(&mut raw, bytes) },
            "allocate pinned host buffer",
        )?;
        Ok(Self {
            ptr: raw.cast::<T>(),
            len,
            _marker: PhantomData,
        })
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn prefix(&self, len: usize) -> Result<&[T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&self.as_slice()[..len])
    }

    fn prefix_mut(&mut self, len: usize) -> Result<&mut [T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&mut self.as_mut_slice()[..len])
    }

    fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than source length {}",
                self.len,
                src.len()
            )));
        }
        self.prefix_mut(src.len())?.copy_from_slice(src);
        Ok(())
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFreeHost(self.ptr.cast::<c_void>()) };
        }
    }
}

struct CudaEvent {
    raw: CudaEventHandle,
}

impl CudaEvent {
    fn try_new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check_cuda(unsafe { cudaEventCreate(&mut raw) }, "create CUDA event")?;
        Ok(Self { raw })
    }

    fn record(&self, stream: cuvs_sys::cudaStream_t) -> Result<()> {
        check_cuda(
            unsafe { cudaEventRecord(self.raw, stream) },
            "record CUDA event",
        )
    }

    fn synchronize(&self) -> Result<()> {
        check_cuda(
            unsafe { cudaEventSynchronize(self.raw) },
            "synchronize CUDA event",
        )
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudaEventDestroy(self.raw) };
        }
    }
}

fn check_cuvs(status: cuvs_sys::cuvsError_t, context: &str) -> Result<()> {
    if status == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }

    let message = unsafe {
        let text = cuvs_sys::cuvsGetLastErrorText();
        if text.is_null() {
            format!("{status:?}")
        } else {
            format!(
                "{status:?}: {}",
                CStr::from_ptr(text).to_string_lossy().into_owned()
            )
        }
    };
    Err(Error::io(format!("cuVS failed to {context}: {message}")))
}

fn check_cuda(status: cuvs_sys::cudaError_t, context: &str) -> Result<()> {
    if status == cuvs_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error::io(format!("CUDA failed to {context}: {status:?}")))
    }
}

fn cuvs_distance_type(metric_type: DistanceType) -> Result<cuvs_sys::cuvsDistanceType> {
    match metric_type {
        DistanceType::L2 => Ok(cuvs_sys::cuvsDistanceType::L2Expanded),
        DistanceType::Cosine => Ok(cuvs_sys::cuvsDistanceType::CosineExpanded),
        DistanceType::Dot => Ok(cuvs_sys::cuvsDistanceType::InnerProduct),
        other => Err(Error::not_supported(format!(
            "cuVS IVF_PQ does not support metric {other:?}"
        ))),
    }
}

fn create_index_params(
    metric_type: DistanceType,
    num_partitions: usize,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
) -> Result<cuvs_sys::cuvsIvfPqIndexParams_t> {
    let mut params = ptr::null_mut();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexParamsCreate(&mut params) },
        "allocate IVF_PQ index params",
    )?;
    let metric = cuvs_distance_type(metric_type)?;
    unsafe {
        (*params).metric = metric;
        (*params).metric_arg = 0.0;
        (*params).add_data_on_build = false;
        (*params).n_lists = num_partitions as u32;
        (*params).kmeans_n_iters = max_iters as u32;
        (*params).kmeans_trainset_fraction = 1.0;
        (*params).pq_bits = num_bits as u32;
        (*params).pq_dim = num_sub_vectors as u32;
        (*params).codebook_kind =
            cuvs_sys::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE;
        (*params).force_random_rotation = false;
        (*params).conservative_memory_allocation = false;
        (*params).max_train_points_per_pq_code = sample_rate as u32;
        (*params).codes_layout = cuvs_sys::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_FLAT;
    }
    Ok(params)
}

fn destroy_index_params(params: cuvs_sys::cuvsIvfPqIndexParams_t) {
    if !params.is_null() {
        let _ = unsafe { cuvs_sys::cuvsIvfPqIndexParamsDestroy(params) };
    }
}

fn make_tensor_view() -> HostTensorView {
    let shape = Vec::new();
    let tensor = cuvs_sys::DLManagedTensor {
        dl_tensor: cuvs_sys::DLTensor {
            data: ptr::null_mut(),
            device: cuvs_sys::DLDevice {
                device_type: cuvs_sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 0,
            dtype: <f32 as DlElement>::dl_dtype(),
            shape: shape.as_ptr() as *mut i64,
            strides: ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ptr::null_mut(),
        deleter: None,
    };
    HostTensorView { shape, tensor }
}

fn tensor_shape(tensor: &cuvs_sys::DLManagedTensor) -> Vec<usize> {
    let dl_tensor = &tensor.dl_tensor;
    (0..dl_tensor.ndim)
        .map(|idx| unsafe { *dl_tensor.shape.add(idx as usize) as usize })
        .collect()
}

fn tensor_num_bytes(tensor: &cuvs_sys::DLManagedTensor) -> usize {
    let shape = tensor_shape(tensor);
    let numel = shape.into_iter().product::<usize>();
    numel * ((tensor.dl_tensor.dtype.bits as usize) / 8)
}

fn copy_tensor_to_host_f32_2d(
    resources: &Resources,
    tensor: &cuvs_sys::DLManagedTensor,
) -> Result<Array2<f32>> {
    let shape = tensor_shape(tensor);
    if shape.len() != 2 {
        return Err(Error::io(format!(
            "expected 2D tensor, got shape {shape:?}"
        )));
    }
    let mut array = Array2::<f32>::zeros((shape[0], shape[1]));
    check_cuda(
        unsafe {
            cuvs_sys::cudaMemcpyAsync(
                array.as_mut_ptr() as *mut _,
                tensor.dl_tensor.data,
                tensor_num_bytes(tensor),
                cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                resources
                    .get_cuda_stream()
                    .map_err(|e| Error::io(e.to_string()))?,
            )
        },
        "copy tensor to host",
    )?;
    resources
        .sync_stream()
        .map_err(|e| Error::io(e.to_string()))?;
    Ok(array)
}

fn copy_tensor_to_host_f32_3d(
    resources: &Resources,
    tensor: &cuvs_sys::DLManagedTensor,
) -> Result<(Vec<f32>, [usize; 3])> {
    let shape = tensor_shape(tensor);
    if shape.len() != 3 {
        return Err(Error::io(format!(
            "expected 3D tensor, got shape {shape:?}"
        )));
    }
    let mut values = vec![0.0f32; shape.iter().product()];
    check_cuda(
        unsafe {
            cuvs_sys::cudaMemcpyAsync(
                values.as_mut_ptr() as *mut _,
                tensor.dl_tensor.data,
                tensor_num_bytes(tensor),
                cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                resources
                    .get_cuda_stream()
                    .map_err(|e| Error::io(e.to_string()))?,
            )
        },
        "copy tensor to host",
    )?;
    resources
        .sync_stream()
        .map_err(|e| Error::io(e.to_string()))?;
    Ok((values, [shape[0], shape[1], shape[2]]))
}

fn infer_dimension(dataset: &Dataset, column: &str) -> Result<usize> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{column}' does not exist in dataset schema"
        ))
    })?;
    infer_vector_dim(&field.data_type())
}

fn matrix_from_vectors<'a>(vectors: &'a FixedSizeListArray) -> Result<MatrixBuffer<'a>> {
    let dim = vectors.value_length() as usize;
    match vectors.value_type() {
        DataType::Float32 => {
            let values = vectors.values().as_primitive::<Float32Type>();
            let values: &[f32] = values.values().as_ref();
            Ok(MatrixBuffer::Borrowed {
                values,
                rows: vectors.len(),
                cols: dim,
            })
        }
        DataType::Float16 => {
            let values = vectors.values().as_primitive::<Float16Type>();
            let data = values
                .values()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            Ok(MatrixBuffer::Owned(
                Array2::from_shape_vec((vectors.len(), dim), data).map_err(|error| {
                    Error::io(format!("failed to create float16 matrix copy: {error}"))
                })?,
            ))
        }
        DataType::Float64 => {
            let values = vectors.values().as_primitive::<Float64Type>();
            let data = values
                .values()
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            Ok(MatrixBuffer::Owned(
                Array2::from_shape_vec((vectors.len(), dim), data).map_err(|error| {
                    Error::io(format!("failed to create float64 matrix copy: {error}"))
                })?,
            ))
        }
        other => Err(Error::not_supported(format!(
            "cuVS IVF_PQ currently supports float16/float32/float64 vectors, got {other}"
        ))),
    }
}

fn ivf_centroids_from_host(array: Array2<f32>) -> Result<FixedSizeListArray> {
    let dim = array.ncols() as i32;
    let values = Float32Array::from_iter_values(array.into_iter());
    Ok(FixedSizeListArray::try_new_from_values(values, dim)?)
}

fn pq_codebook_from_host(
    values: Vec<f32>,
    shape: [usize; 3],
    num_sub_vectors: usize,
    dimension: usize,
    num_bits: usize,
) -> Result<FixedSizeListArray> {
    let pq_book_size = 1usize << num_bits;
    let subvector_dim = dimension / num_sub_vectors;
    let expected = [num_sub_vectors, subvector_dim, pq_book_size];
    if shape != expected {
        return Err(Error::io(format!(
            "cuVS returned incompatible PQ codebook shape: expected {expected:?}, got {shape:?}"
        )));
    }

    let mut flattened = Vec::with_capacity(values.len());
    for subspace in 0..num_sub_vectors {
        for centroid in 0..pq_book_size {
            for component in 0..subvector_dim {
                let source_idx = ((subspace * subvector_dim + component) * pq_book_size) + centroid;
                flattened.push(values[source_idx]);
            }
        }
    }

    Ok(FixedSizeListArray::try_new_from_values(
        Float32Array::from(flattened),
        subvector_dim as i32,
    )?)
}

fn build_metadata_batch(
    ivf_centroids: &FixedSizeListArray,
    pq_codebook: &FixedSizeListArray,
) -> Result<RecordBatch> {
    let ivf_offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, ivf_centroids.len() as i32]));
    let pq_offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, pq_codebook.len() as i32]));
    let ivf_list = ListArray::new(
        Arc::new(Field::new(
            "_ivf_centroids_item",
            ivf_centroids.data_type().clone(),
            false,
        )),
        ivf_offsets,
        Arc::new(ivf_centroids.clone()),
        None,
    );
    let pq_list = ListArray::new(
        Arc::new(Field::new(
            "_pq_codebook_item",
            pq_codebook.data_type().clone(),
            false,
        )),
        pq_offsets,
        Arc::new(pq_codebook.clone()),
        None,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("_ivf_centroids", ivf_list.data_type().clone(), false),
        Field::new("_pq_codebook", pq_list.data_type().clone(), false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(ivf_list), Arc::new(pq_list)],
    )?)
}

fn metadata_writer_options() -> Result<FileWriterOptions> {
    Ok(FileWriterOptions {
        format_version: Some(
            PARTITION_ARTIFACT_FILE_VERSION
                .parse::<LanceFileVersion>()
                .map_err(|error| {
                    Error::invalid_input(format!(
                        "invalid partition artifact file version '{}': {}",
                        PARTITION_ARTIFACT_FILE_VERSION, error
                    ))
                })?,
        ),
        ..Default::default()
    })
}

async fn write_partition_artifact_metadata(
    artifact_uri: &str,
    trained: &TrainedIvfPqIndex,
) -> Result<()> {
    let (object_store, root_dir) = lance::io::ObjectStore::from_uri(artifact_uri)
        .await
        .map_err(|error| Error::io(error.to_string()))?;
    let path = root_dir.child(PARTITION_ARTIFACT_METADATA_FILE_NAME);
    let batch = build_metadata_batch(&trained.ivf_centroids, &trained.pq_codebook)?;
    let mut writer = FileWriter::try_new(
        object_store.create(&path).await?,
        lance_core::datatypes::Schema::try_from(batch.schema().as_ref())?,
        metadata_writer_options()?,
    )?;
    writer.add_schema_metadata(
        "lance:index_build:artifact_version".to_string(),
        "1".to_string(),
    );
    writer.add_schema_metadata(
        "lance:index_build:distance_type".to_string(),
        trained.metric_type.to_string(),
    );
    writer.add_schema_metadata(
        "lance:index_build:num_partitions".to_string(),
        trained.num_partitions.to_string(),
    );
    writer.add_schema_metadata(
        "lance:index_build:num_sub_vectors".to_string(),
        trained.num_sub_vectors.to_string(),
    );
    writer.add_schema_metadata(
        "lance:index_build:num_bits".to_string(),
        trained.num_bits.to_string(),
    );
    writer.add_schema_metadata(
        "lance:index_build:dimension".to_string(),
        trained.dimension.to_string(),
    );
    writer.write_batch(&batch).await?;
    writer.finish().await?;
    Ok(())
}

fn build_partition_batch(
    row_ids: Arc<dyn Array>,
    partitions: &[u32],
    pq_codes: &[u8],
    code_width: usize,
) -> Result<RecordBatch> {
    if pq_codes.len() != partitions.len() * code_width {
        return Err(Error::io(format!(
            "partition artifact batch expects {} PQ codes for {} rows and code width {}, got {}",
            partitions.len() * code_width,
            partitions.len(),
            code_width,
            pq_codes.len()
        )));
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(PART_ID_COLUMN, DataType::UInt32, false),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                code_width as i32,
            ),
            true,
        ),
    ]));
    let pq_codes = FixedSizeListArray::try_new_from_values(
        UInt8Array::from_iter_values(pq_codes.iter().copied()),
        code_width as i32,
    )?;
    Ok(RecordBatch::try_new(
        schema,
        vec![
            row_ids,
            Arc::new(UInt32Array::from_iter_values(partitions.iter().copied())),
            Arc::new(pq_codes),
        ],
    )?)
}

struct TransformSlot {
    input_host: PinnedHostBuffer<f32>,
    input_device: DeviceTensor<f32>,
    labels_host: PinnedHostBuffer<u32>,
    labels_device: DeviceTensor<u32>,
    codes_host: PinnedHostBuffer<u8>,
    codes_device: DeviceTensor<u8>,
    h2d_start: CudaEvent,
    h2d_done: CudaEvent,
    transform_done: CudaEvent,
    output_ready: CudaEvent,
    row_ids: Option<Arc<dyn Array>>,
    rows: usize,
}

impl TransformSlot {
    fn try_new(
        resources: &Resources,
        max_rows: usize,
        dimension: usize,
        code_width: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_host: PinnedHostBuffer::try_new(max_rows * dimension)?,
            input_device: DeviceTensor::try_new(resources, &[max_rows, dimension])?,
            labels_host: PinnedHostBuffer::try_new(max_rows)?,
            labels_device: DeviceTensor::try_new(resources, &[max_rows])?,
            codes_host: PinnedHostBuffer::try_new(max_rows * code_width)?,
            codes_device: DeviceTensor::try_new(resources, &[max_rows, code_width])?,
            h2d_start: CudaEvent::try_new()?,
            h2d_done: CudaEvent::try_new()?,
            transform_done: CudaEvent::try_new()?,
            output_ready: CudaEvent::try_new()?,
            row_ids: None,
            rows: 0,
        })
    }

    fn has_pending_output(&self) -> bool {
        self.row_ids.is_some()
    }

    fn launch(
        &mut self,
        trained: &TrainedIvfPqIndex,
        stream: cuvs_sys::cudaStream_t,
        row_ids: Arc<dyn Array>,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
    ) -> Result<()> {
        let code_width = trained.pq_code_width();
        self.input_host.copy_from_slice(matrix)?;
        self.input_device.set_shape(&[rows, dimension])?;
        self.labels_device.set_shape(&[rows])?;
        self.codes_device.set_shape(&[rows, code_width])?;
        self.rows = rows;
        self.row_ids = Some(row_ids);

        self.h2d_start.record(stream)?;
        self.input_device.copy_from_host_async(
            &trained.resources,
            self.input_host.prefix(rows * dimension)?,
        )?;
        self.h2d_done.record(stream)?;
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsIvfPqTransform(
                    trained.resources.0,
                    trained.index.raw,
                    self.input_device.as_mut_ptr(),
                    self.labels_device.as_mut_ptr(),
                    self.codes_device.as_mut_ptr(),
                )
            },
            "transform vectors with IVF_PQ",
        )?;
        self.transform_done.record(stream)?;
        self.labels_device
            .copy_to_host_async(&trained.resources, self.labels_host.prefix_mut(rows)?)?;
        self.codes_device.copy_to_host_async(
            &trained.resources,
            self.codes_host.prefix_mut(rows * code_width)?,
        )?;
        self.output_ready.record(stream)?;
        Ok(())
    }

    fn drain_to_batch(&mut self, code_width: usize) -> Result<Option<RecordBatch>> {
        if !self.has_pending_output() {
            return Ok(None);
        }

        self.output_ready.synchronize()?;
        let row_ids = self
            .row_ids
            .take()
            .ok_or_else(|| Error::io("transform slot is missing row ids"))?;
        let batch = build_partition_batch(
            row_ids,
            self.labels_host.prefix(self.rows)?,
            self.codes_host.prefix(self.rows * code_width)?,
            code_width,
        )?;
        self.rows = 0;
        Ok(Some(batch))
    }
}

async fn for_each_transformed_batch<F, Fut>(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    batch_size: usize,
    filter_nan: bool,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(RecordBatch) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let code_width = trained.pq_code_width();
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    if dataset
        .schema()
        .field(column)
        .is_some_and(|field| field.nullable && filter_nan)
    {
        scanner.filter(&format!("{column} is not null"))?;
    }
    scanner.with_row_id();
    scanner.batch_size(batch_size);
    let mut stream = scanner.try_into_stream().await?;
    let cuda_stream = trained
        .resources
        .get_cuda_stream()
        .map_err(|error| Error::io(error.to_string()))?;
    let mut slots = (0..PIPELINE_SLOTS)
        .map(|_| {
            TransformSlot::try_new(
                &trained.resources,
                batch_size,
                trained.dimension,
                code_width,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut next_slot = 0usize;

    loop {
        let Some(batch) = stream.try_next().await? else {
            break;
        };
        let slot = &mut slots[next_slot];
        if let Some(transformed) = slot.drain_to_batch(code_width)? {
            on_batch(transformed).await?;
        }

        let vectors = vector_column_to_fsl(&batch, column)?;
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::invalid_input(format!("transform batch is missing {ROW_ID}")))?;
        let finite_mask = is_finite(&vectors);
        let valid_rows = finite_mask.true_count();
        if valid_rows == 0 {
            continue;
        }
        if valid_rows != vectors.len() {
            warn!(
                "{} vectors are ignored during partition assignment because they are null or non-finite",
                vectors.len() - valid_rows
            );
        }

        let filtered_row_ids = if valid_rows == row_ids.len() {
            row_ids.clone()
        } else {
            filter(row_ids.as_ref(), &finite_mask)?
        };
        let filtered_vectors = if valid_rows == vectors.len() {
            vectors
        } else {
            let vector_column = batch.column_by_name(column).ok_or_else(|| {
                Error::invalid_input(format!(
                    "transform batch is missing vector column '{column}'"
                ))
            })?;
            let field = batch
                .schema()
                .field_with_name(column)
                .map_err(|_| {
                    Error::invalid_input(format!(
                        "transform batch schema is missing field '{column}'"
                    ))
                })?
                .clone();
            let filtered_vectors = filter(vector_column.as_ref(), &finite_mask)?;
            vector_column_to_fsl(
                &RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![field])),
                    vec![filtered_vectors],
                )?,
                column,
            )?
        };

        let matrix = matrix_from_vectors(&filtered_vectors)?;
        let matrix_view = matrix.view()?;
        let input_slice = matrix_view
            .as_slice_memory_order()
            .ok_or_else(|| Error::io("transform matrix is not contiguous"))?;

        slot.launch(
            trained,
            cuda_stream,
            filtered_row_ids,
            input_slice,
            matrix.rows(),
            matrix_view.ncols(),
        )?;
        next_slot = (next_slot + 1) % PIPELINE_SLOTS;
    }

    for slot in &mut slots {
        if let Some(transformed) = slot.drain_to_batch(code_width)? {
            on_batch(transformed).await?;
        }
    }
    Ok(())
}

pub async fn train_ivf_pq(
    dataset: &Dataset,
    column: &str,
    num_partitions: usize,
    metric_type: DistanceType,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
    filter_nan: bool,
) -> Result<TrainedIvfPqIndex> {
    if num_bits != 8 {
        return Err(Error::not_supported(
            "cuVS IVF_PQ currently supports only num_bits=8",
        ));
    }

    let dimension = infer_dimension(dataset, column)?;
    if dimension % num_sub_vectors != 0 {
        return Err(Error::invalid_input(format!(
            "cuVS IVF_PQ requires vector dimension {} to be divisible by num_sub_vectors {}",
            dimension, num_sub_vectors
        )));
    }

    let num_rows = dataset.count_rows(None).await?;
    if num_rows == 0 {
        return Err(Error::invalid_input(
            "cuVS training requires at least one training vector",
        ));
    }
    let train_rows = num_rows
        .min((num_partitions * sample_rate).max(256 * 256))
        .max(1);
    let train_vectors = if filter_nan {
        let batch = dataset.scan().project(&[column])?.try_into_batch().await?;
        let vectors = vector_column_to_fsl(&batch, column)?;
        let mask = is_finite(&vectors);
        let filtered = filter(&vectors, &mask)?.as_fixed_size_list().clone();
        filtered.slice(0, train_rows.min(filtered.len()))
    } else {
        let projection = dataset.schema().project(&[column])?;
        let batch = dataset.sample(train_rows, &projection, None).await?;
        vector_column_to_fsl(&batch, column)?
    };
    if train_vectors.is_empty() {
        return Err(Error::invalid_input(
            "cuVS training requires at least one non-null training vector",
        ));
    }

    let matrix = matrix_from_vectors(&train_vectors)?;
    let resources = Resources::new().map_err(|error| Error::io(error.to_string()))?;
    let index = CuvsIvfPqIndex::try_new()?;
    let params = create_index_params(
        metric_type,
        num_partitions,
        num_sub_vectors,
        sample_rate,
        max_iters,
        num_bits,
    )?;
    let matrix_view = matrix.view()?;
    let mut dataset_tensor = HostTensorView::try_new::<f32>(
        &[matrix_view.nrows(), matrix_view.ncols()],
        matrix_view.as_ptr() as *mut std::ffi::c_void,
    );

    let build_result = check_cuvs(
        unsafe {
            cuvs_sys::cuvsIvfPqBuild(resources.0, params, dataset_tensor.as_mut_ptr(), index.raw)
        },
        "build IVF_PQ index",
    );
    destroy_index_params(params);
    build_result?;

    let mut centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetCenters(index.raw, centers.as_mut_ptr()) },
        "get IVF centroids",
    )?;
    let ivf_centroids =
        ivf_centroids_from_host(copy_tensor_to_host_f32_2d(&resources, &centers.tensor)?)?;

    let mut pq_centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetPqCenters(index.raw, pq_centers.as_mut_ptr()) },
        "get PQ codebook",
    )?;
    let (pq_codebook_values, pq_codebook_shape) =
        copy_tensor_to_host_f32_3d(&resources, &pq_centers.tensor)?;
    let pq_codebook = pq_codebook_from_host(
        pq_codebook_values,
        pq_codebook_shape,
        num_sub_vectors,
        dimension,
        num_bits,
    )?;

    Ok(TrainedIvfPqIndex {
        resources,
        index,
        num_partitions,
        dimension,
        num_sub_vectors,
        num_bits,
        metric_type,
        ivf_centroids,
        pq_codebook,
    })
}

pub async fn assign_ivf_pq_to_artifact(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    artifact_uri: &str,
    batch_size: usize,
    filter_nan: bool,
) -> Result<Vec<String>> {
    let code_width = trained.pq_code_width();
    let builder = Arc::new(tokio::sync::Mutex::new(
        PartitionArtifactBuilder::try_new(artifact_uri, trained.num_partitions, code_width, None)
            .await?,
    ));
    for_each_transformed_batch(dataset, column, trained, batch_size, filter_nan, |batch| {
        let builder = builder.clone();
        async move {
            builder.lock().await.append_batch(&batch).await?;
            Ok(())
        }
    })
    .await?;
    let mut builder = Arc::try_unwrap(builder)
        .map_err(|_| Error::io("partition artifact builder still has outstanding references"))?
        .into_inner();

    write_partition_artifact_metadata(artifact_uri, trained).await?;
    let mut files = builder
        .finish(PARTITION_ARTIFACT_METADATA_FILE_NAME, None)
        .await?;
    if files.len() > 1 {
        files.insert(1, PARTITION_ARTIFACT_METADATA_FILE_NAME.to_string());
    }
    Ok(files)
}

pub async fn build_vector_index(
    dataset: &Dataset,
    params: VectorIndexBuildParams,
) -> Result<VectorIndexBuildOutput> {
    CuvsVectorBuildBackend.build(dataset, params).await
}

#[cfg(feature = "python")]
fn parse_distance_type(metric: &str) -> PyResult<DistanceType> {
    match metric.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" => Ok(DistanceType::Dot),
        other => Err(PyValueError::new_err(format!(
            "unsupported metric_type for cuVS IVF_PQ: {other}"
        ))),
    }
}

#[cfg(feature = "python")]
fn fixed_size_list_to_array2(array: &FixedSizeListArray) -> PyResult<Array2<f32>> {
    let dim = array.value_length() as usize;
    let values = array.values().as_primitive::<Float32Type>();
    Array2::from_shape_vec((array.len(), dim), values.values().to_vec()).map_err(|error| {
        PyRuntimeError::new_err(format!(
            "failed to reshape IVF centroids into ndarray: {error}"
        ))
    })
}

#[cfg(feature = "python")]
fn fixed_size_list_to_array3(
    array: &FixedSizeListArray,
    num_sub_vectors: usize,
) -> PyResult<Array3<f32>> {
    if num_sub_vectors == 0 {
        return Err(PyValueError::new_err(
            "num_sub_vectors must be greater than 0",
        ));
    }

    if array.len() % num_sub_vectors != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "PQ codebook rows {} are not divisible by num_sub_vectors {}",
            array.len(),
            num_sub_vectors
        )));
    }

    let pq_book_size = array.len() / num_sub_vectors;
    let subvector_dim = array.value_length() as usize;
    let values = array.values().as_primitive::<Float32Type>();
    Array3::from_shape_vec(
        (num_sub_vectors, pq_book_size, subvector_dim),
        values.values().to_vec(),
    )
    .map_err(|error| {
        PyRuntimeError::new_err(format!(
            "failed to reshape PQ codebook into ndarray: {error}"
        ))
    })
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(
    signature = (
        dataset_uri,
        column,
        *,
        artifact_uri,
        metric_type = "L2",
        num_partitions,
        num_sub_vectors,
        sample_rate = 256,
        max_iters = 50,
        num_bits = 8,
        batch_size = 1024 * 128,
        filter_nan = true,
    )
)]
fn build_ivf_pq_artifact<'py>(
    py: Python<'py>,
    dataset_uri: &str,
    column: &str,
    artifact_uri: &str,
    metric_type: &str,
    num_partitions: usize,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
    batch_size: usize,
    filter_nan: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let metric_type = parse_distance_type(metric_type)?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let dataset = runtime
        .block_on(Dataset::open(dataset_uri))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let output = runtime
        .block_on(build_vector_index(
            &dataset,
            VectorIndexBuildParams {
                column: column.to_string(),
                kind: VectorIndexKind::IvfPq(IvfPqBuildParams {
                    num_partitions,
                    metric_type,
                    num_sub_vectors,
                    sample_rate,
                    max_iters,
                    num_bits,
                }),
                artifact_uri: artifact_uri.to_string(),
                batch_size,
                filter_nan,
            },
        ))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("artifact_uri", output.artifact_uri())?;
    dict.set_item("files", output.files().to_vec())?;
    dict.set_item(
        "ivf_centroids",
        PyArray2::from_owned_array(py, fixed_size_list_to_array2(output.ivf_centroids())?),
    )?;
    dict.set_item(
        "pq_codebook",
        PyArray3::from_owned_array(
            py,
            fixed_size_list_to_array3(output.pq_codebook(), num_sub_vectors)?,
        ),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_ivf_pq_artifact, m)?)?;
    Ok(())
}
