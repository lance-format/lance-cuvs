# lance-cuvs

`lance-cuvs` provides cuVS-backed IVF_PQ training and artifact building for Lance datasets.

It covers one narrow part of the indexing pipeline:

1. Train an IVF_PQ model with cuVS.
2. Build a partition-local artifact from a Lance dataset.
3. Pass the training output and artifact back to Lance for finalization.

It does **not** create or register a Lance index on your behalf.

## Requirements

- Linux
- Python 3.12+
- CUDA 12 runtime available on the machine
- cuVS runtime `libcuvs-cu12==26.2.0`
- A matching backend package: `lance-cuvs-backend-cuvs-26-02`
- A Lance build that includes the vector-build APIs used by this project

## Installation

Install the loader, the matching backend package, and the cuVS runtime into the
same Python environment:

- `lance-cuvs`
- `lance-cuvs-backend-cuvs-26-02`
- `libcuvs-cu12==26.2.0`

If you are working from this repository, the shortest local setup is:

```bash
just sync-dev
just backend-develop
```

The loader chooses the backend from the installed cuVS runtime version. You can
override detection when needed:

```bash
export LANCE_CUVS_BACKEND=cuvs-26-02
```

## Quick Start

```python
from pathlib import Path

import lance
import lance_cuvs
import pyarrow as pa


def vector_array(rows: int, dim: int) -> pa.FixedSizeListArray:
    values = pa.array(
        [row + axis / 100.0 for row in range(rows) for axis in range(dim)],
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(values, dim)


dataset_uri = Path("/tmp/example.lance")
artifact_uri = Path("/tmp/example-artifact")

table = pa.table({"vector": vector_array(rows=4096, dim=16)})
lance.write_dataset(table, dataset_uri)

training = lance_cuvs.train_ivf_pq(
    dataset_uri,
    "vector",
    metric_type="L2",
    num_partitions=8,
    num_sub_vectors=4,
    sample_rate=4,
    max_iters=20,
)

artifact = lance_cuvs.build_ivf_pq_artifact(
    dataset_uri,
    "vector",
    training=training,
    artifact_uri=artifact_uri,
)

print(type(training.ivf_centroids()))
print(type(training.pq_codebook()))
print(artifact.artifact_uri)
print(artifact.files)
```

## Python API

### `lance_cuvs.train_ivf_pq(...)`

Trains an IVF_PQ model with cuVS and returns `IvfPqTrainingOutput`.

Key fields and methods:

- `num_partitions`
- `num_sub_vectors`
- `num_bits`
- `metric_type`
- `ivf_centroids()` -> `pyarrow.FixedSizeListArray`
- `pq_codebook()` -> `pyarrow.FixedSizeListArray`

### `lance_cuvs.build_ivf_pq_artifact(...)`

Builds a partition-local artifact from a Lance dataset and a previously trained
model, then returns `IvfPqArtifactOutput`.

Key fields:

- `artifact_uri`
- `files`

`artifact_uri` must resolve to the local filesystem.

## Scope

Use `lance-cuvs` when you want cuVS to do the expensive GPU-side training and
encoding work, but you still want Lance to own index finalization.

Do not expect this package to:

- finalize an index
- register an index in a dataset
- provide a generic Lance wrapper beyond IVF_PQ training and artifact build

## License

Apache License 2.0. See [LICENSE](LICENSE).
