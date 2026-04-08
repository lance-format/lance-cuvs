"""Python bindings for the `lance-cuvs` backend package.

This package provides the backend side of Lance's cuVS integration:

- train an IVF_PQ model with cuVS
- encode a dataset into a partition-local artifact

It deliberately stops before Lance's canonical index finalize step.
"""

from __future__ import annotations

import ctypes
import os
import site
import tempfile
from pathlib import Path

_PRELOAD_DONE = False


def _shared_library_roots() -> list[Path]:
    roots: list[Path] = []

    for site_packages in site.getsitepackages():
        site_root = Path(site_packages)
        for package_name in ("rapids_logger", "librmm", "libraft", "libcuvs"):
            for suffix in ("lib64", "lib"):
                candidate = site_root / package_name / suffix
                if candidate.is_dir():
                    roots.append(candidate)
        for candidate in site_root.iterdir():
            if (
                candidate.is_dir()
                and candidate.name.endswith(".libs")
                and any(
                    token in candidate.name.lower()
                    for token in ("cuvs", "raft", "rmm", "rapids")
                )
            ):
                roots.append(candidate)

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home:
            cuda_root = Path(cuda_home)
            for candidate in (
                cuda_root / "targets" / "x86_64-linux" / "lib",
                cuda_root / "lib64",
                cuda_root / "lib",
            ):
                if candidate.is_dir():
                    roots.append(candidate)

    for candidate in (
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.9/lib64"),
    ):
        if candidate.is_dir():
            roots.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def _preload_shared_libraries() -> None:
    global _PRELOAD_DONE
    if _PRELOAD_DONE:
        return

    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "librapids_logger.so",
        "librmm.so",
        "libraft.so",
        "libcuvs_c.so",
    ]
    loaded: list[str] = []
    failures: list[str] = []

    for root in _shared_library_roots():
        for name in candidates:
            for path in sorted(root.glob(f"{name}*")):
                if not path.is_file():
                    continue
                try:
                    ctypes.CDLL(os.fspath(path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.fspath(path))
                except OSError as exc:
                    failures.append(f"{path}: {exc}")

    _PRELOAD_DONE = True
    if not loaded and failures:
        raise ImportError(
            "Failed to preload required cuVS shared libraries. "
            + " | ".join(failures[:8])
        )


_preload_shared_libraries()

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
        Destination directory for the partition-local artifact. A temporary
        directory is created when omitted.
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
