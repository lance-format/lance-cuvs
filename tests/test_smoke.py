# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import lance
import lance_cuvs
import pyarrow as pa
import pytest

DIM = 16
ROWS = 4096
NUM_PARTITIONS = 8
NUM_SUB_VECTORS = 4
NUM_BITS = 8


def _has_visible_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False

    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _require_gpu() -> None:
    if _has_visible_gpu():
        return

    message = "pylance-cuvs smoke test requires a CUDA-capable GPU"
    if os.environ.get("LANCE_CUVS_REQUIRE_GPU") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _vector_array(rows: int, dim: int) -> pa.FixedSizeListArray:
    values = pa.array(
        [
            row + axis / 100.0
            for row in range(rows)
            for axis in range(dim)
        ],
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(values, dim)


@pytest.mark.gpu
def test_train_and_build_ivf_pq_artifact(tmp_path: Path) -> None:
    _require_gpu()

    dataset_uri = tmp_path / "dataset.lance"
    artifact_uri = tmp_path / "artifact"

    table = pa.table({"vector": _vector_array(ROWS, DIM)})
    lance.write_dataset(table, dataset_uri)

    training = lance_cuvs.train_ivf_pq(
        dataset_uri,
        "vector",
        metric_type="L2",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        sample_rate=4,
        max_iters=20,
        num_bits=NUM_BITS,
        filter_nan=False,
    )

    assert training.num_partitions == NUM_PARTITIONS
    assert training.num_sub_vectors == NUM_SUB_VECTORS
    assert training.num_bits == NUM_BITS
    assert training.metric_type == "L2"

    ivf_centroids = training.ivf_centroids()
    pq_codebook = training.pq_codebook()

    assert isinstance(ivf_centroids, pa.FixedSizeListArray)
    assert len(ivf_centroids) == NUM_PARTITIONS
    assert ivf_centroids.type.list_size == DIM

    assert isinstance(pq_codebook, pa.FixedSizeListArray)
    assert len(pq_codebook) == NUM_SUB_VECTORS * (1 << NUM_BITS)
    assert pq_codebook.type.list_size == DIM // NUM_SUB_VECTORS

    artifact = lance_cuvs.build_ivf_pq_artifact(
        dataset_uri,
        "vector",
        training=training,
        artifact_uri=artifact_uri,
        batch_size=1024,
        filter_nan=False,
    )

    assert artifact.artifact_uri == str(artifact_uri)
    assert artifact.files
    assert artifact_uri.is_dir()

    for relative_path in artifact.files:
        assert (artifact_uri / relative_path).exists(), relative_path
