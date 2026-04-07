from __future__ import annotations

import ctypes
import os
import site
import tempfile
from pathlib import Path
from typing import Any

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

from ._native import build_ivf_pq_artifact as _build_ivf_pq_artifact


def _dataset_uri(dataset_or_uri: Any) -> str:
    if isinstance(dataset_or_uri, (str, os.PathLike)):
        return os.fspath(dataset_or_uri)

    uri = getattr(dataset_or_uri, "uri", None)
    if isinstance(uri, str):
        return uri

    raise TypeError(
        "dataset_or_uri must be a Lance dataset object with a string 'uri' property "
        "or a dataset URI/path"
    )


def _dataset_object(dataset_or_uri: Any):
    if hasattr(dataset_or_uri, "create_index") and isinstance(
        getattr(dataset_or_uri, "uri", None), str
    ):
        return dataset_or_uri

    import lance

    return lance.dataset(_dataset_uri(dataset_or_uri))


def build_ivf_pq_artifact(
    dataset_or_uri: Any,
    column: str,
    *,
    artifact_uri: str | os.PathLike[str] | None = None,
    metric_type: str = "L2",
    num_partitions: int,
    num_sub_vectors: int,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    batch_size: int = 1024 * 128,
    filter_nan: bool = True,
) -> dict[str, Any]:
    if artifact_uri is None:
        artifact_uri = tempfile.mkdtemp(prefix="lance-cuvs-artifact-")

    return _build_ivf_pq_artifact(
        dataset_uri=_dataset_uri(dataset_or_uri),
        column=column,
        artifact_uri=os.fspath(artifact_uri),
        metric_type=metric_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        batch_size=batch_size,
        filter_nan=filter_nan,
    )


def create_ivf_pq_index(
    dataset_or_uri: Any,
    column: str,
    *,
    metric_type: str = "L2",
    num_partitions: int,
    num_sub_vectors: int,
    name: str | None = None,
    replace: bool = False,
    artifact_uri: str | os.PathLike[str] | None = None,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    batch_size: int = 1024 * 128,
    filter_nan: bool = True,
) -> dict[str, Any]:
    if num_bits != 8:
        raise ValueError(
            "create_ivf_pq_index currently requires num_bits=8 because Lance Python "
            "create_index expects a PQ codebook with 256 centroids per sub-vector"
        )

    dataset = _dataset_object(dataset_or_uri)
    output = build_ivf_pq_artifact(
        dataset,
        column,
        artifact_uri=artifact_uri,
        metric_type=metric_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        batch_size=batch_size,
        filter_nan=filter_nan,
    )
    dataset.create_index(
        column,
        "IVF_PQ",
        name=name,
        metric=metric_type,
        replace=replace,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        ivf_centroids=output["ivf_centroids"],
        pq_codebook=output["pq_codebook"],
        precomputed_partition_artifact_uri=output["artifact_uri"],
    )
    return output


__all__ = [
    "build_ivf_pq_artifact",
    "create_ivf_pq_index",
]
