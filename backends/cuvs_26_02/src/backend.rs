// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cuda::{
    CudaEvent, CuvsIvfPqIndex, DeviceTensor, HostTensorView, MatrixBuffer, PinnedHostBuffer,
    RegisteredHostBuffer, check_cuvs, copy_tensor_to_host_f32_2d, copy_tensor_to_host_f32_3d,
    create_index_params, destroy_index_params, ivf_centroids_from_host, make_tensor_view,
    matrix_from_vectors, pq_codebook_from_host,
};
use arrow::compute::{concat_batches, filter};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt8Array, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use cuvs::Resources;
use futures::{
    FutureExt, SinkExt, StreamExt, TryStreamExt, channel::mpsc, future::LocalBoxFuture, stream,
};
use lance::dataset::Dataset;
use lance::index::vector::PartitionArtifactBuilder;
use lance::index::vector::utils::infer_vector_dim;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::utils::is_finite;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_linalg::distance::DistanceType;
use log::warn;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PARTITION_ARTIFACT_METADATA_FILE_NAME: &str = "metadata.lance";
const PIPELINE_SLOTS: usize = 2;
const DEFAULT_SCAN_FRAGMENT_READAHEAD: usize = 64;
const DEFAULT_SCAN_IO_BUFFER_SIZE: u64 = 16 * 1024 * 1024 * 1024;
const TRAINING_SAMPLE_CHUNK_ROWS: usize = 8 * 1024;
const TRAINING_SAMPLE_BATCH_READAHEAD: usize = 64;

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
                    let train_start = Instant::now();
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
                    eprintln!(
                        "cuVS train_ivf_pq time: {:.3}s",
                        train_start.elapsed().as_secs_f64()
                    );
                    let artifact_start = Instant::now();
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
                    eprintln!(
                        "cuVS assign_ivf_pq_to_artifact time: {:.3}s files={}",
                        artifact_start.elapsed().as_secs_f64(),
                        files.len()
                    );
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

fn get_column_from_batch(batch: &RecordBatch, column: &str) -> Result<ArrayRef> {
    if let Some(col) = batch.column_by_name(column) {
        return Ok(col.clone());
    }

    let parts = lance_core::datatypes::parse_field_path(column)
        .map_err(|error| Error::index(format!("failed to parse field path '{column}': {error}")))?;
    if parts.is_empty() {
        return Err(Error::index(format!("invalid empty field path: {column}")));
    }

    let mut current_array = batch
        .column_by_name(&parts[0])
        .ok_or_else(|| {
            Error::index(format!(
                "column '{column}' does not exist in batch (missing root field '{}')",
                parts[0]
            ))
        })?
        .clone();

    for part in &parts[1..] {
        let struct_array = current_array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| {
                Error::index(format!(
                    "cannot access nested field '{part}' in column '{column}': parent is not a struct"
                ))
            })?;
        current_array = struct_array
            .column_by_name(part)
            .ok_or_else(|| {
                Error::index(format!(
                    "nested field '{part}' does not exist in column '{column}'"
                ))
            })?
            .clone();
    }

    Ok(current_array)
}

fn vector_column_to_fsl(batch: &RecordBatch, column: &str) -> Result<FixedSizeListArray> {
    let array = get_column_from_batch(batch, column)?;
    match array.data_type() {
        DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().clone()),
        DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            Ok(list_array.values().as_fixed_size_list().clone())
        }
        _ => Err(Error::index(format!(
            "column '{column}' is not a vector column"
        ))),
    }
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
    input_device: DeviceTensor<f32>,
    labels_host: PinnedHostBuffer<u32>,
    labels_device: DeviceTensor<u32>,
    codes_host: PinnedHostBuffer<u8>,
    codes_device: DeviceTensor<u8>,
    h2d_start: CudaEvent,
    h2d_done: CudaEvent,
    transform_done: CudaEvent,
    output_ready: CudaEvent,
    input_vectors: Option<FixedSizeListArray>,
    input_registration: Option<RegisteredHostBuffer>,
    row_ids: Option<Arc<dyn Array>>,
    rows: usize,
}

struct PreparedTransformBatch {
    row_ids: Arc<dyn Array>,
    vectors: FixedSizeListArray,
    input_registration: Option<RegisteredHostBuffer>,
}

#[derive(Default)]
struct ArtifactPrepareStats {
    input_batches: usize,
    input_rows: usize,
    scan_wait: Duration,
    send: Duration,
    vector: Duration,
    filter: Duration,
    matrix: Duration,
    register: Duration,
    registered_bytes: usize,
}

impl TransformSlot {
    fn try_new(
        resources: &Resources,
        max_rows: usize,
        dimension: usize,
        code_width: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_device: DeviceTensor::try_new(resources, &[max_rows, dimension])?,
            labels_host: PinnedHostBuffer::try_new(max_rows)?,
            labels_device: DeviceTensor::try_new(resources, &[max_rows])?,
            codes_host: PinnedHostBuffer::try_new(max_rows * code_width)?,
            codes_device: DeviceTensor::try_new(resources, &[max_rows, code_width])?,
            h2d_start: CudaEvent::try_new()?,
            h2d_done: CudaEvent::try_new()?,
            transform_done: CudaEvent::try_new()?,
            output_ready: CudaEvent::try_new()?,
            input_vectors: None,
            input_registration: None,
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
        prepared: PreparedTransformBatch,
    ) -> Result<()> {
        let code_width = trained.pq_code_width();
        let row_ids = prepared.row_ids;
        let vectors = prepared.vectors;
        let matrix = matrix_from_vectors(&vectors)?;
        let (input_slice, rows, dimension, keep_input_vectors) = match &matrix {
            MatrixBuffer::Borrowed { values, rows, cols } => (*values, *rows, *cols, true),
            MatrixBuffer::Owned(array) => (
                array
                    .as_slice_memory_order()
                    .ok_or_else(|| Error::io("transform matrix is not contiguous"))?,
                array.nrows(),
                array.ncols(),
                false,
            ),
        };

        self.input_device.set_shape(&[rows, dimension])?;
        self.labels_device.set_shape(&[rows])?;
        self.codes_device.set_shape(&[rows, code_width])?;
        self.rows = rows;
        self.row_ids = Some(row_ids);
        self.input_registration = prepared.input_registration;

        self.h2d_start.record(stream)?;
        self.input_device
            .copy_from_host_async(&trained.resources, input_slice)?;
        self.h2d_done.record(stream)?;
        if keep_input_vectors {
            self.input_vectors = Some(vectors);
        } else {
            self.h2d_done.synchronize()?;
            self.input_vectors = None;
        }
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
        self.input_registration = None;
        self.input_vectors = None;
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

#[derive(Default)]
struct ArtifactBuildStats {
    input_batches: usize,
    input_rows: usize,
    output_batches: usize,
    output_rows: usize,
    scan_wait: Duration,
    drain: Duration,
    send: Duration,
    prepare_send: Duration,
    vector: Duration,
    filter: Duration,
    matrix: Duration,
    launch: Duration,
    register: Duration,
    registered_bytes: usize,
}

impl ArtifactBuildStats {
    fn merge_prepare(&mut self, prepare: ArtifactPrepareStats) {
        self.input_batches += prepare.input_batches;
        self.input_rows += prepare.input_rows;
        self.scan_wait += prepare.scan_wait;
        self.prepare_send += prepare.send;
        self.vector += prepare.vector;
        self.filter += prepare.filter;
        self.matrix += prepare.matrix;
        self.register += prepare.register;
        self.registered_bytes += prepare.registered_bytes;
    }

    fn record_output(&mut self, batch: &RecordBatch) {
        self.output_batches += 1;
        self.output_rows += batch.num_rows();
    }

    fn log(&self) {
        eprintln!(
            "cuVS artifact stages: input_batches={} input_rows={} output_batches={} output_rows={} scan_wait_s={:.3} drain_s={:.3} send_s={:.3} prepare_send_s={:.3} vector_s={:.3} filter_s={:.3} matrix_s={:.3} launch_s={:.3}",
            self.input_batches,
            self.input_rows,
            self.output_batches,
            self.output_rows,
            secs(self.scan_wait),
            secs(self.drain),
            secs(self.send),
            secs(self.prepare_send),
            secs(self.vector),
            secs(self.filter),
            secs(self.matrix),
            secs(self.launch),
        );
        eprintln!(
            "cuVS artifact h2d registration: register_s={:.3} registered_gib={:.3}",
            secs(self.register),
            self.registered_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn training_sample_ranges(num_rows: usize, sample_rows: usize) -> Vec<Range<u64>> {
    let sample_rows = sample_rows.min(num_rows);
    if sample_rows == 0 {
        return Vec::new();
    }
    if sample_rows == num_rows {
        return vec![0..num_rows as u64];
    }

    let chunk_rows = TRAINING_SAMPLE_CHUNK_ROWS.min(sample_rows);
    let num_chunks = sample_rows.div_ceil(chunk_rows);
    let mut remaining = sample_rows;
    let mut ranges = Vec::with_capacity(num_chunks);
    for chunk_idx in 0..num_chunks {
        let rows = chunk_rows.min(remaining);
        remaining -= rows;
        let max_start = num_rows - rows;
        let start = if num_chunks == 1 {
            max_start / 2
        } else {
            ((chunk_idx as u128 * max_start as u128) / (num_chunks - 1) as u128) as usize
        };
        ranges.push(start as u64..(start + rows) as u64);
    }
    ranges
}

async fn sample_training_vectors(
    dataset: &Dataset,
    column: &str,
    sample_rows: usize,
) -> Result<FixedSizeListArray> {
    let num_rows = dataset.count_rows(None).await?;
    if num_rows == 0 {
        return Err(Error::invalid_input(
            "cuVS training requires at least one training vector",
        ));
    }

    let ranges = training_sample_ranges(num_rows, sample_rows);
    let projection = Arc::new(dataset.schema().project(&[column])?);
    let stream = dataset.take_scan(
        Box::pin(stream::iter(ranges.into_iter().map(Ok))),
        projection,
        TRAINING_SAMPLE_BATCH_READAHEAD,
    );
    let batches = stream.try_collect::<Vec<_>>().await?;
    let Some(schema) = batches.first().map(RecordBatch::schema) else {
        return Err(Error::invalid_input(
            "cuVS training sample did not return any vectors",
        ));
    };
    let batch = concat_batches(&schema, &batches)?;
    Ok(vector_column_to_fsl(&batch, column)?)
}

async fn prepare_transform_batches(
    dataset: Dataset,
    column: String,
    batch_size: usize,
    filter_nan: bool,
    mut prepared_tx: mpsc::Sender<PreparedTransformBatch>,
) -> Result<ArtifactPrepareStats> {
    let mut scanner = dataset.scan();
    scanner.project(&[&column])?;
    if dataset
        .schema()
        .field(&column)
        .is_some_and(|field| field.nullable && filter_nan)
    {
        scanner.filter(&format!("{column} is not null"))?;
    }
    scanner.with_row_id();
    scanner.batch_size(batch_size);
    scanner.scan_in_order(false);
    scanner.fragment_readahead(DEFAULT_SCAN_FRAGMENT_READAHEAD);
    scanner.io_buffer_size(DEFAULT_SCAN_IO_BUFFER_SIZE);
    let mut stream = scanner.try_into_stream().await?;
    let mut stats = ArtifactPrepareStats::default();

    loop {
        let scan_start = Instant::now();
        let Some(batch) = stream.try_next().await? else {
            stats.scan_wait += scan_start.elapsed();
            break;
        };
        stats.scan_wait += scan_start.elapsed();
        stats.input_batches += 1;
        stats.input_rows += batch.num_rows();

        let vector_start = Instant::now();
        let vectors = vector_column_to_fsl(&batch, &column)?;
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::invalid_input(format!("transform batch is missing {ROW_ID}")))?;
        stats.vector += vector_start.elapsed();

        let filter_start = Instant::now();
        let (filtered_row_ids, filtered_vectors) = if filter_nan {
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
                let vector_column = batch.column_by_name(&column).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "transform batch is missing vector column '{column}'"
                    ))
                })?;
                let field = batch
                    .schema()
                    .field_with_name(&column)
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
                    &column,
                )?
            };
            (filtered_row_ids, filtered_vectors)
        } else {
            (row_ids.clone(), vectors)
        };
        stats.filter += filter_start.elapsed();

        let matrix_start = Instant::now();
        let matrix = matrix_from_vectors(&filtered_vectors)?;
        let input_registration = match &matrix {
            MatrixBuffer::Borrowed { values, .. } => {
                let register_start = Instant::now();
                let registration = match RegisteredHostBuffer::try_new(values) {
                    Ok(registration) => Some(registration),
                    Err(error) => {
                        warn!(
                            "failed to register host vector buffer for CUDA H2D; falling back to pageable memory: {error}"
                        );
                        None
                    }
                };
                stats.register += register_start.elapsed();
                stats.registered_bytes += registration
                    .as_ref()
                    .map(RegisteredHostBuffer::original_bytes)
                    .unwrap_or_default();
                registration
            }
            MatrixBuffer::Owned(_) => None,
        };
        stats.matrix += matrix_start.elapsed();

        let prepared = PreparedTransformBatch {
            row_ids: filtered_row_ids,
            vectors: filtered_vectors,
            input_registration,
        };
        let send_start = Instant::now();
        prepared_tx
            .send(prepared)
            .await
            .map_err(|error| Error::io(format!("failed to forward prepared batch: {error}")))?;
        stats.send += send_start.elapsed();
    }

    Ok(stats)
}

async fn append_transformed_batches_to_artifact(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    batch_size: usize,
    filter_nan: bool,
    append_tx: &mut mpsc::Sender<Result<RecordBatch>>,
) -> Result<()> {
    let code_width = trained.pq_code_width();
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
    let mut stats = ArtifactBuildStats::default();
    let (prepared_tx, mut prepared_rx) = mpsc::channel::<PreparedTransformBatch>(PIPELINE_SLOTS);
    let prepare_task = tokio::spawn(prepare_transform_batches(
        dataset.clone(),
        column.to_string(),
        batch_size,
        filter_nan,
        prepared_tx,
    ));

    while let Some(prepared) = prepared_rx.next().await {
        let slot = &mut slots[next_slot];
        let drain_start = Instant::now();
        let transformed = if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_output(&transformed);
            Some(transformed)
        } else {
            stats.drain += drain_start.elapsed();
            None
        };

        let launch_start = Instant::now();
        slot.launch(trained, cuda_stream, prepared)?;
        stats.launch += launch_start.elapsed();

        if let Some(transformed) = transformed {
            let send_start = Instant::now();
            append_tx.send(Ok(transformed)).await.map_err(|error| {
                Error::io(format!("failed to forward transformed batch: {error}"))
            })?;
            stats.send += send_start.elapsed();
        }
        next_slot = (next_slot + 1) % PIPELINE_SLOTS;
    }
    let prepare_stats = prepare_task
        .await
        .map_err(|error| Error::io(format!("prepare transform task failed: {error}")))??;
    stats.merge_prepare(prepare_stats);

    for slot in &mut slots {
        let drain_start = Instant::now();
        if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_output(&transformed);
            let send_start = Instant::now();
            append_tx.send(Ok(transformed)).await.map_err(|error| {
                Error::io(format!("failed to forward transformed batch: {error}"))
            })?;
            stats.send += send_start.elapsed();
        } else {
            stats.drain += drain_start.elapsed();
        }
    }
    stats.log();
    Ok(())
}

async fn append_artifact_batches(
    mut artifact: PartitionArtifactBuilder,
    mut rx: mpsc::Receiver<Result<RecordBatch>>,
) -> Result<Vec<String>> {
    let mut batches = 0usize;
    let mut rows = 0usize;
    let mut append_time = Duration::default();
    while let Some(batch) = rx.next().await {
        let batch = batch?;
        batches += 1;
        rows += batch.num_rows();
        let append_start = Instant::now();
        artifact.append_batch(&batch).await?;
        append_time += append_start.elapsed();
    }

    let finish_start = Instant::now();
    let files = artifact
        .finish(PARTITION_ARTIFACT_METADATA_FILE_NAME, None)
        .await?;
    let finish_time = finish_start.elapsed();
    eprintln!(
        "cuVS artifact append task: batches={} rows={} append_s={:.3} finish_s={:.3} files={}",
        batches,
        rows,
        secs(append_time),
        secs(finish_time),
        files.len()
    );
    Ok(files)
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
#[allow(clippy::too_many_arguments)]
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

    let train_rows = (num_partitions * sample_rate).max(256 * 256).max(1);
    let sample_start = Instant::now();
    let train_vectors = sample_training_vectors(dataset, column, train_rows).await?;
    eprintln!(
        "cuVS train sample time: {:.3}s rows={}",
        sample_start.elapsed().as_secs_f64(),
        train_vectors.len()
    );
    let train_vectors = if filter_nan {
        let mask = is_finite(&train_vectors);
        let filtered = filter(&train_vectors, &mask)?.as_fixed_size_list().clone();
        filtered.slice(0, train_rows.min(filtered.len()))
    } else {
        train_vectors
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
    let artifact = PartitionArtifactBuilder::try_new(
        artifact_uri,
        trained.num_partitions,
        trained.pq_code_width(),
        storage_options,
    )
    .await?;

    let (mut append_tx, append_rx) = mpsc::channel::<Result<RecordBatch>>(PIPELINE_SLOTS);
    let append_task = tokio::spawn(append_artifact_batches(artifact, append_rx));

    let append_start = Instant::now();
    let append_result = append_transformed_batches_to_artifact(
        dataset,
        column,
        trained,
        batch_size,
        filter_nan,
        &mut append_tx,
    )
    .await;
    drop(append_tx);
    if let Err(error) = append_result {
        append_task.abort();
        return Err(error);
    }
    eprintln!(
        "cuVS artifact append_transformed_batches time: {:.3}s",
        append_start.elapsed().as_secs_f64()
    );
    let files = append_task
        .await
        .map_err(|error| Error::io(format!("partition artifact append task failed: {error}")))??;
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
