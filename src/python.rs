// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::backend::{TrainedIvfPqIndex, assign_ivf_pq_to_artifact, train_ivf_pq};
use arrow_array::Array;
use arrow_pyarrow::ToPyArrow;
use lance::dataset::Dataset;
use lance_linalg::distance::DistanceType;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(
    name = "IvfPqTrainingOutput",
    module = "lance_cuvs._native",
    unsendable
)]
struct PyTrainedIvfPqIndex {
    inner: TrainedIvfPqIndex,
}

#[pymethods]
impl PyTrainedIvfPqIndex {
    #[getter]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    #[getter]
    fn num_sub_vectors(&self) -> usize {
        self.inner.num_sub_vectors
    }

    #[getter]
    fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    #[getter]
    fn metric_type(&self) -> &'static str {
        match self.inner.metric_type() {
            DistanceType::L2 => "L2",
            DistanceType::Cosine => "Cosine",
            DistanceType::Dot => "Dot",
            _ => "Unknown",
        }
    }

    fn ivf_centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.ivf_centroids().to_data().to_pyarrow(py)
    }

    fn pq_codebook<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.pq_codebook().to_data().to_pyarrow(py)
    }
}

#[pyclass(
    name = "IvfPqArtifactOutput",
    module = "lance_cuvs._native",
    unsendable
)]
struct PyPartitionArtifactBuildOutput {
    artifact_uri: String,
    files: Vec<String>,
}

#[pymethods]
impl PyPartitionArtifactBuildOutput {
    #[getter]
    fn artifact_uri(&self) -> &str {
        &self.artifact_uri
    }

    #[getter]
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
    )
)]
#[pyo3(name = "train_ivf_pq")]
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
) -> PyResult<Py<PyTrainedIvfPqIndex>> {
    let metric_type = parse_distance_type(metric_type)?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let dataset = runtime
        .block_on(Dataset::open(dataset_uri))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
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
    )
)]
fn build_ivf_pq_artifact<'py>(
    py: Python<'py>,
    dataset_uri: &str,
    column: &str,
    artifact_uri: &str,
    training: PyRef<'py, PyTrainedIvfPqIndex>,
    batch_size: usize,
    filter_nan: bool,
) -> PyResult<Py<PyPartitionArtifactBuildOutput>> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let dataset = runtime
        .block_on(Dataset::open(dataset_uri))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let files = runtime
        .block_on(assign_ivf_pq_to_artifact(
            &dataset,
            column,
            &training.inner,
            artifact_uri,
            batch_size,
            filter_nan,
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
