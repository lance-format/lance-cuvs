// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PyO3 bindings for the public Python package surface.

use crate::backend::{TrainedIvfPqIndex, assign_ivf_pq_to_artifact, train_ivf_pq};
use arrow_array::Array;
use arrow_pyarrow::ToPyArrow;
use lance::dataset::builder::DatasetBuilder;
use lance_linalg::distance::DistanceType;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;

#[pyclass(
    name = "IvfPqTrainingOutput",
    module = "lance_cuvs_backend_cu12._native",
    unsendable
)]
/// Python wrapper around a trained cuVS IVF_PQ model.
struct PyTrainedIvfPqIndex {
    inner: TrainedIvfPqIndex,
}

#[pymethods]
impl PyTrainedIvfPqIndex {
    #[getter]
    /// Number of trained IVF partitions.
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    #[getter]
    /// Number of PQ subvectors.
    fn num_sub_vectors(&self) -> usize {
        self.inner.num_sub_vectors
    }

    #[getter]
    /// Number of bits per PQ code.
    fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    #[getter]
    /// Distance metric used during training.
    fn metric_type(&self) -> &'static str {
        match self.inner.metric_type() {
            DistanceType::L2 => "L2",
            DistanceType::Cosine => "Cosine",
            DistanceType::Dot => "Dot",
            _ => "Unknown",
        }
    }

    /// Return IVF centroids as a `pyarrow.FixedSizeListArray`.
    fn ivf_centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.ivf_centroids().to_data().to_pyarrow(py)
    }

    /// Return the PQ codebook as a `pyarrow.FixedSizeListArray`.
    fn pq_codebook<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.pq_codebook().to_data().to_pyarrow(py)
    }
}

#[pyclass(
    name = "IvfPqArtifactOutput",
    module = "lance_cuvs_backend_cu12._native",
    unsendable
)]
/// Python wrapper around a partition-local artifact build result.
struct PyPartitionArtifactBuildOutput {
    artifact_uri: String,
    files: Vec<String>,
}

#[pymethods]
impl PyPartitionArtifactBuildOutput {
    #[getter]
    /// Root URI of the generated artifact.
    fn artifact_uri(&self) -> &str {
        &self.artifact_uri
    }

    #[getter]
    /// Relative file names produced under the artifact root.
    fn files(&self) -> Vec<String> {
        self.files.clone()
    }
}

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

#[pyfunction]
#[pyo3(
    signature = (
        dataset_uri,
        column,
        *,
        metric_type = "L2",
        num_partitions,
        num_sub_vectors,
        sample_rate = 256,
        max_iters = 50,
        num_bits = 8,
        filter_nan = true,
        storage_options = None,
    )
)]
#[pyo3(name = "train_ivf_pq")]
/// Train an IVF_PQ model and return Arrow-native training outputs.
#[allow(clippy::too_many_arguments)]
fn train_ivf_pq_py(
    py: Python<'_>,
    dataset_uri: &str,
    column: &str,
    metric_type: &str,
    num_partitions: usize,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
    filter_nan: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyTrainedIvfPqIndex>> {
    let metric_type = parse_distance_type(metric_type)?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let mut builder = DatasetBuilder::from_uri(dataset_uri);
    if let Some(storage_options) = storage_options {
        builder = builder.with_storage_options(storage_options);
    }
    let dataset = runtime
        .block_on(builder.load())
        .map_err(|error: lance::Error| PyRuntimeError::new_err(error.to_string()))?;
    let trained = runtime
        .block_on(train_ivf_pq(
            &dataset,
            column,
            num_partitions,
            metric_type,
            num_sub_vectors,
            sample_rate,
            max_iters,
            num_bits,
            filter_nan,
        ))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    Py::new(py, PyTrainedIvfPqIndex { inner: trained })
}

#[pyfunction]
#[pyo3(
    signature = (
        dataset_uri,
        column,
        *,
        artifact_uri,
        training,
        batch_size = 1024 * 128,
        filter_nan = true,
        storage_options = None,
    )
)]
/// Encode a dataset into a partition-local IVF_PQ artifact.
#[allow(clippy::too_many_arguments)]
fn build_ivf_pq_artifact<'py>(
    py: Python<'py>,
    dataset_uri: &str,
    column: &str,
    artifact_uri: &str,
    training: PyRef<'py, PyTrainedIvfPqIndex>,
    batch_size: usize,
    filter_nan: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyPartitionArtifactBuildOutput>> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let mut builder = DatasetBuilder::from_uri(dataset_uri);
    if let Some(storage_options) = storage_options.clone() {
        builder = builder.with_storage_options(storage_options);
    }
    let dataset = runtime
        .block_on(builder.load())
        .map_err(|error: lance::Error| PyRuntimeError::new_err(error.to_string()))?;
    let files = runtime
        .block_on(assign_ivf_pq_to_artifact(
            &dataset,
            column,
            &training.inner,
            artifact_uri,
            batch_size,
            filter_nan,
            storage_options.as_ref(),
        ))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    Py::new(
        py,
        PyPartitionArtifactBuildOutput {
            artifact_uri: artifact_uri.to_string(),
            files,
        },
    )
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainedIvfPqIndex>()?;
    m.add_class::<PyPartitionArtifactBuildOutput>()?;
    m.add_function(wrap_pyfunction!(train_ivf_pq_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_ivf_pq_artifact, m)?)?;
    Ok(())
}
