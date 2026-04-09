// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray};
use arrow_schema::DataType;
use cuvs::Resources;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use ndarray::{Array2, ArrayView2};
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::ptr;

pub(crate) type CudaEventHandle = *mut c_void;

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

pub(crate) struct CuvsIvfPqIndex {
    pub(crate) raw: cuvs_sys::cuvsIvfPqIndex_t,
}

impl CuvsIvfPqIndex {
    pub(crate) fn try_new() -> Result<Self> {
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

pub(crate) enum MatrixBuffer<'a> {
    Borrowed {
        values: &'a [f32],
        rows: usize,
        cols: usize,
    },
    Owned(Array2<f32>),
}

impl MatrixBuffer<'_> {
    pub(crate) fn view(&self) -> Result<ArrayView2<'_, f32>> {
        match self {
            Self::Borrowed { values, rows, cols } => ArrayView2::from_shape((*rows, *cols), values)
                .map_err(|error| {
                    Error::io(format!("failed to create borrowed matrix view: {error}"))
                }),
            Self::Owned(array) => Ok(array.view()),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Borrowed { rows, .. } => *rows,
            Self::Owned(array) => array.nrows(),
        }
    }
}

pub(crate) struct HostTensorView {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
}

impl HostTensorView {
    pub(crate) fn try_new<T: DlElement>(shape: &[usize], data: *mut c_void) -> Self {
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

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }

    pub(crate) fn tensor(&self) -> &cuvs_sys::DLManagedTensor {
        &self.tensor
    }
}

pub(crate) trait DlElement: Copy + Default {
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

pub(crate) struct DeviceTensor<T: DlElement> {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
    capacity_bytes: usize,
    resources: cuvs_sys::cuvsResources_t,
    _marker: PhantomData<T>,
}

impl<T: DlElement> DeviceTensor<T> {
    pub(crate) fn try_new(resources: &Resources, shape: &[usize]) -> Result<Self> {
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

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }

    pub(crate) fn set_shape(&mut self, shape: &[usize]) -> Result<()> {
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

    pub(crate) fn copy_from_host_async(&mut self, resources: &Resources, src: &[T]) -> Result<()> {
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

    pub(crate) fn copy_to_host_async(&self, resources: &Resources, dst: &mut [T]) -> Result<()> {
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

pub(crate) struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> PinnedHostBuffer<T> {
    pub(crate) fn try_new(len: usize) -> Result<Self> {
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

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) fn prefix(&self, len: usize) -> Result<&[T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&self.as_slice()[..len])
    }

    pub(crate) fn prefix_mut(&mut self, len: usize) -> Result<&mut [T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&mut self.as_mut_slice()[..len])
    }

    pub(crate) fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
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

pub(crate) struct CudaEvent {
    raw: CudaEventHandle,
}

impl CudaEvent {
    pub(crate) fn try_new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check_cuda(unsafe { cudaEventCreate(&mut raw) }, "create CUDA event")?;
        Ok(Self { raw })
    }

    pub(crate) fn record(&self, stream: cuvs_sys::cudaStream_t) -> Result<()> {
        check_cuda(
            unsafe { cudaEventRecord(self.raw, stream) },
            "record CUDA event",
        )
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
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

pub(crate) fn check_cuvs(status: cuvs_sys::cuvsError_t, context: &str) -> Result<()> {
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

pub(crate) fn check_cuda(status: cuvs_sys::cudaError_t, context: &str) -> Result<()> {
    if status == cuvs_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error::io(format!("CUDA failed to {context}: {status:?}")))
    }
}

pub(crate) fn cuvs_distance_type(metric_type: DistanceType) -> Result<cuvs_sys::cuvsDistanceType> {
    match metric_type {
        DistanceType::L2 => Ok(cuvs_sys::cuvsDistanceType::L2Expanded),
        DistanceType::Cosine => Ok(cuvs_sys::cuvsDistanceType::CosineExpanded),
        DistanceType::Dot => Ok(cuvs_sys::cuvsDistanceType::InnerProduct),
        other => Err(Error::not_supported(format!(
            "cuVS IVF_PQ does not support metric {other:?}"
        ))),
    }
}

pub(crate) fn create_index_params(
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

pub(crate) fn destroy_index_params(params: cuvs_sys::cuvsIvfPqIndexParams_t) {
    if !params.is_null() {
        let _ = unsafe { cuvs_sys::cuvsIvfPqIndexParamsDestroy(params) };
    }
}

pub(crate) fn make_tensor_view() -> HostTensorView {
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

pub(crate) fn tensor_shape(tensor: &cuvs_sys::DLManagedTensor) -> Vec<usize> {
    let dl_tensor = &tensor.dl_tensor;
    (0..dl_tensor.ndim)
        .map(|idx| unsafe { *dl_tensor.shape.add(idx as usize) as usize })
        .collect()
}

pub(crate) fn tensor_num_bytes(tensor: &cuvs_sys::DLManagedTensor) -> usize {
    let shape = tensor_shape(tensor);
    let numel = shape.into_iter().product::<usize>();
    numel * ((tensor.dl_tensor.dtype.bits as usize) / 8)
}

pub(crate) fn copy_tensor_to_host_f32_2d(
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

pub(crate) fn copy_tensor_to_host_f32_3d(
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

pub(crate) fn matrix_from_vectors<'a>(vectors: &'a FixedSizeListArray) -> Result<MatrixBuffer<'a>> {
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

pub(crate) fn ivf_centroids_from_host(array: Array2<f32>) -> Result<FixedSizeListArray> {
    let dim = array.ncols() as i32;
    let values = arrow_array::Float32Array::from_iter_values(array.into_iter());
    Ok(FixedSizeListArray::try_new_from_values(values, dim)?)
}

pub(crate) fn pq_codebook_from_host(
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
        arrow_array::Float32Array::from(flattened),
        subvector_dim as i32,
    )?)
}
