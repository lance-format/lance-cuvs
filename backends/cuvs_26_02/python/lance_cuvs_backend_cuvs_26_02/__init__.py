"""Python bindings for the cuVS 26.02 `lance-cuvs` backend package."""

from __future__ import annotations

import os
import tempfile

from ._native import (
    IvfPqArtifactOutput,
    IvfPqTrainingOutput,
    build_ivf_pq_artifact as _build_ivf_pq_artifact,
    train_ivf_pq as _train_ivf_pq,
)


def _dataset_uri(dataset_uri: str | os.PathLike[str]) -> str:
    if isinstance(dataset_uri, (str, os.PathLike)):
        return os.fspath(dataset_uri)

    raise TypeError("dataset_uri must be a dataset URI/path")


def train_ivf_pq(
    dataset_uri: str | os.PathLike[str],
    column: str,
    *,
    metric_type: str = "L2",
    num_partitions: int,
    num_sub_vectors: int,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    filter_nan: bool = True,
) -> IvfPqTrainingOutput:
    """Train an IVF_PQ model with cuVS.

    Parameters
    ----------
    dataset_uri:
        URI or local path of the Lance dataset to read.
    column:
        Name of the vector column.
    metric_type:
        Distance metric understood by cuVS. Supported values are ``"L2"``,
        ``"Cosine"``, and ``"Dot"``.
    num_partitions:
        Number of IVF partitions to train.
    num_sub_vectors:
        Number of PQ subvectors.
    sample_rate:
        Training sample rate used to size the sampled training set.
    max_iters:
        Maximum number of cuVS k-means iterations.
    num_bits:
        Number of bits per PQ code. cuVS currently supports only ``8`` here.
    filter_nan:
        Whether to drop null or non-finite vectors before training.

    Returns
    -------
    IvfPqTrainingOutput
        A reusable training object whose centroids and PQ codebook are exposed
        as Arrow arrays.
    """
    return _train_ivf_pq(
        dataset_uri=_dataset_uri(dataset_uri),
        column=column,
        metric_type=metric_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
    )


def build_ivf_pq_artifact(
    dataset_uri: str | os.PathLike[str],
    column: str,
    *,
    training: IvfPqTrainingOutput,
    artifact_uri: str | os.PathLike[str] | None = None,
    batch_size: int = 1024 * 128,
    filter_nan: bool = True,
) -> IvfPqArtifactOutput:
    """Encode a dataset into a partition-local IVF_PQ artifact.

    Parameters
    ----------
    dataset_uri:
        URI or local path of the Lance dataset to encode.
    column:
        Name of the vector column.
    training:
        A training result previously returned by :func:`train_ivf_pq`.
    artifact_uri:
        Destination directory for the partition-local artifact. This must
        resolve to the local filesystem. A temporary directory is created when
        omitted.
    batch_size:
        Number of rows per transform batch.
    filter_nan:
        Whether to drop null or non-finite vectors during artifact build.

    Returns
    -------
    IvfPqArtifactOutput
        The partition-local artifact metadata. Callers can feed its
        ``artifact_uri`` back into Lance finalization.
    """
    if artifact_uri is None:
        artifact_uri = tempfile.mkdtemp(prefix="lance-cuvs-artifact-")

    return _build_ivf_pq_artifact(
        dataset_uri=_dataset_uri(dataset_uri),
        column=column,
        artifact_uri=os.fspath(artifact_uri),
        training=training,
        batch_size=batch_size,
        filter_nan=filter_nan,
    )


__all__ = [
    "IvfPqArtifactOutput",
    "IvfPqTrainingOutput",
    "train_ivf_pq",
    "build_ivf_pq_artifact",
]
