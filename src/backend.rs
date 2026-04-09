// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cuda::{
    CudaEvent, CuvsIvfPqIndex, DeviceTensor, HostTensorView, PinnedHostBuffer, check_cuvs,
    copy_tensor_to_host_f32_2d, copy_tensor_to_host_f32_3d, create_index_params,
    destroy_index_params, ivf_centroids_from_host, make_tensor_view, matrix_from_vectors,
    pq_codebook_from_host,
};
use arrow::compute::filter;
use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, ListArray, RecordBatch, UInt8Array, UInt32Array};
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
use std::collections::HashMap;
use std::sync::Arc;

const PARTITION_ARTIFACT_METADATA_FILE_NAME: &str = "metadata.lance";
const PARTITION_ARTIFACT_FILE_VERSION: &str = "2.2";
const PIPELINE_SLOTS: usize = 2;

/// A trained cuVS IVF_PQ model that can be reused for artifact builds.
///
/// The training outputs are exposed as Arrow arrays so callers can feed them
/// directly back into Lance's index finalization APIs.
pub struct TrainedIvfPqIndex {
    pub(crate) resources: Resources,
    pub(crate) index: CuvsIvfPqIndex,
    pub(crate) num_partitions: usize,
    pub(crate) dimension: usize,
    pub(crate) num_sub_vectors: usize,
    pub(crate) num_bits: usize,
    pub(crate) metric_type: DistanceType,
    pub(crate) ivf_centroids: FixedSizeListArray,
    pub(crate) pq_codebook: FixedSizeListArray,
}

impl TrainedIvfPqIndex {
    /// Return IVF centroids as a fixed-size list Arrow array.
    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        &self.ivf_centroids
    }

    /// Return the PQ codebook as a fixed-size list Arrow array.
    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        &self.pq_codebook
    }

    /// Return the number of trained IVF partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Return the encoded PQ byte width, which equals the number of subvectors.
    pub fn pq_code_width(&self) -> usize {
        self.num_sub_vectors
    }

    /// Return the distance metric used during training.
    pub fn metric_type(&self) -> DistanceType {
        self.metric_type
    }

    /// Return the number of bits used per PQ code.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

/// Parameters for a vector index build request handled by this crate.
///
/// The request describes only the backend-owned steps: training and artifact
/// generation. Lance finalization happens outside this crate.
#[derive(Clone)]
pub struct VectorIndexBuildParams {
    pub column: String,
    pub kind: VectorIndexKind,
    pub artifact_uri: String,
    pub batch_size: usize,
    pub filter_nan: bool,
}

/// Supported vector index kinds for the current backend surface.
#[derive(Clone)]
pub enum VectorIndexKind {
    /// Build an IVF_PQ artifact with cuVS.
    IvfPq(IvfPqBuildParams),
}

/// Build parameters for a cuVS IVF_PQ job.
#[derive(Clone)]
pub struct IvfPqBuildParams {
    pub num_partitions: usize,
    pub metric_type: DistanceType,
    pub num_sub_vectors: usize,
    pub sample_rate: usize,
    pub max_iters: usize,
    pub num_bits: usize,
}

/// Backend output that callers can pass to Lance finalization.
pub enum VectorIndexBuildOutput {
    /// A partition-local artifact plus the Arrow-native training outputs used
    /// to build it.
    PartitionArtifact(PartitionArtifactBuildOutput),
}

impl VectorIndexBuildOutput {
    /// Return the output artifact URI.
    pub fn artifact_uri(&self) -> &str {
        match self {
            Self::PartitionArtifact(output) => &output.artifact_uri,
        }
    }

    /// Return the artifact file list relative to the artifact root.
    pub fn files(&self) -> &[String] {
        match self {
            Self::PartitionArtifact(output) => &output.files,
        }
    }

    /// Return trained IVF centroids.
    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.ivf_centroids,
        }
    }

    /// Return the trained PQ codebook.
    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.pq_codebook,
        }
    }
}

/// Result of building a partition-local artifact.
pub struct PartitionArtifactBuildOutput {
    pub(crate) artifact_uri: String,
    pub(crate) files: Vec<String>,
    pub(crate) ivf_centroids: FixedSizeListArray,
    pub(crate) pq_codebook: FixedSizeListArray,
}

/// Minimal backend interface for vector build providers.
pub trait VectorBuildBackend {
    /// Execute a backend build request and return a Lance-consumable output.
    fn build<'a>(
        &'a self,
        dataset: &'a Dataset,
        params: VectorIndexBuildParams,
    ) -> LocalBoxFuture<'a, Result<VectorIndexBuildOutput>>;
}

/// cuVS implementation of [`VectorBuildBackend`].
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
                        None,
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

fn infer_dimension(dataset: &Dataset, column: &str) -> Result<usize> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{column}' does not exist in dataset schema"
        ))
    })?;
    infer_vector_dim(&field.data_type())
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
    storage_options: Option<&HashMap<String, String>>,
) -> Result<()> {
    let registry = Arc::new(lance_io::object_store::ObjectStoreRegistry::default());
    let params = if let Some(storage_options) = storage_options {
        lance_io::object_store::ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance_io::object_store::StorageOptionsAccessor::with_static_options(
                    storage_options.clone(),
                ),
            )),
            ..Default::default()
        }
    } else {
        lance_io::object_store::ObjectStoreParams::default()
    };
    let (object_store, root_dir) =
        lance::io::ObjectStore::from_uri_and_params(registry, artifact_uri, &params)
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

/// Train an IVF_PQ model with cuVS and return Arrow-native training outputs.
///
/// This function performs only the backend-owned training step. The returned
/// value can be reused across multiple artifact builds.
///
/// # Errors
///
/// Returns an error when the input column is missing, empty, incompatible with
/// cuVS, or when CUDA/cuVS reports a build failure.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::train_ivf_pq;
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let training = train_ivf_pq(
///     dataset,
///     "vector",
///     256,
///     DistanceType::L2,
///     16,
///     256,
///     50,
///     8,
///     true,
/// )
/// .await?;
/// assert_eq!(training.num_partitions(), 256);
/// # Ok(())
/// # }
/// ```
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
        ivf_centroids_from_host(copy_tensor_to_host_f32_2d(&resources, centers.tensor())?)?;

    let mut pq_centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetPqCenters(index.raw, pq_centers.as_mut_ptr()) },
        "get PQ codebook",
    )?;
    let (pq_codebook_values, pq_codebook_shape) =
        copy_tensor_to_host_f32_3d(&resources, pq_centers.tensor())?;
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

/// Build a partition-local IVF_PQ artifact from a trained model.
///
/// The output artifact is intended to be consumed by Lance's
/// `precomputed_partition_artifact_uri` finalization path.
///
/// # Errors
///
/// Returns an error when scanning, encoding, or writing the artifact fails.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::{assign_ivf_pq_to_artifact, train_ivf_pq};
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let training = train_ivf_pq(
///     dataset,
///     "vector",
///     256,
///     DistanceType::L2,
///     16,
///     256,
///     50,
///     8,
///     true,
/// )
/// .await?;
/// let files = assign_ivf_pq_to_artifact(
///     dataset,
///     "vector",
///     &training,
///     "/tmp/lance-cuvs-artifact",
///     1024 * 128,
///     true,
/// )
/// .await?;
/// assert!(!files.is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn assign_ivf_pq_to_artifact(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    artifact_uri: &str,
    batch_size: usize,
    filter_nan: bool,
    storage_options: Option<&HashMap<String, String>>,
) -> Result<Vec<String>> {
    let code_width = trained.pq_code_width();
    let builder = Arc::new(tokio::sync::Mutex::new(
        PartitionArtifactBuilder::try_new(
            artifact_uri,
            trained.num_partitions,
            code_width,
            storage_options,
        )
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

    write_partition_artifact_metadata(artifact_uri, trained, storage_options).await?;
    let mut files = builder
        .finish(PARTITION_ARTIFACT_METADATA_FILE_NAME, None)
        .await?;
    if files.len() > 1 {
        files.insert(1, PARTITION_ARTIFACT_METADATA_FILE_NAME.to_string());
    }
    Ok(files)
}

/// Execute a full backend build request.
///
/// This convenience entrypoint wraps training and artifact construction behind
/// [`VectorBuildBackend`]. It still stops before Lance finalization.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::{
/// #     build_vector_index, IvfPqBuildParams, VectorIndexBuildParams, VectorIndexKind,
/// # };
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let output = build_vector_index(
///     dataset,
///     VectorIndexBuildParams {
///         column: "vector".to_string(),
///         kind: VectorIndexKind::IvfPq(IvfPqBuildParams {
///             num_partitions: 256,
///             metric_type: DistanceType::L2,
///             num_sub_vectors: 16,
///             sample_rate: 256,
///             max_iters: 50,
///             num_bits: 8,
///         }),
///         artifact_uri: "/tmp/lance-cuvs-artifact".to_string(),
///         batch_size: 1024 * 128,
///         filter_nan: true,
///     },
/// )
/// .await?;
/// assert!(!output.files().is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn build_vector_index(
    dataset: &Dataset,
    params: VectorIndexBuildParams,
) -> Result<VectorIndexBuildOutput> {
    CuvsVectorBuildBackend.build(dataset, params).await
}
