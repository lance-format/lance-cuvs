# lance-cuvs

`lance-cuvs` is an experimental cuVS-backed build backend for Lance vector indices.

The package is intentionally narrow:

- train an `IVF_PQ` model with cuVS
- encode a Lance dataset into a partition-local artifact
- return Arrow-native training outputs

It does **not** finalize or register Lance indices on behalf of callers. Callers are expected to pass the training outputs and artifact URI back into Lance's own index creation APIs.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Status

- Current package version: `0.1.0`
- Current backend scope: `IVF_PQ`
- Current dependency baseline: Lance git versions that expose unreleased partition-artifact build APIs

## Installation

Use Python `3.12+` so the installed RAPIDS wheels match the `cuvs-sys` `26.2.0` toolchain.

Create a local development environment with `uv`:

```bash
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv sync --group dev --no-install-project
eval "$(uv run python tools/rapids_env.py --format shell)"
uv run maturin develop --release
```

You will also need:

- CUDA toolkit
- cuVS / RAPIDS Python wheels installed in the active environment

## Public Python API

### `lance_cuvs.train_ivf_pq(...)`

Train an `IVF_PQ` model and return a reusable training object.

The returned `IvfPqTrainingOutput` exposes:

- `num_partitions`
- `num_sub_vectors`
- `num_bits`
- `metric_type`
- `ivf_centroids()` as `pyarrow.FixedSizeListArray`
- `pq_codebook()` as `pyarrow.FixedSizeListArray`

### `lance_cuvs.build_ivf_pq_artifact(...)`

Encode a dataset with a previously trained model and materialize a partition-local artifact.

The returned `IvfPqArtifactOutput` exposes:

- `artifact_uri`
- `files`

## Example

```python
import lance
import lance_cuvs

training = lance_cuvs.train_ivf_pq(
    "/data/example.lance",
    "vector",
    num_partitions=256,
    num_sub_vectors=16,
)

artifact = lance_cuvs.build_ivf_pq_artifact(
    "/data/example.lance",
    "vector",
    training=training,
    artifact_uri="/tmp/example-artifact",
)
```

Lance finalization stays outside this package. `lance-cuvs` only returns the
Arrow-native training outputs and the partition-local artifact needed by the
caller-managed finalize step.

## Development

Format the Rust sources with:

```bash
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv sync --group dev --no-install-project
uv run cargo fmt --all
```

Check the Python package surface with:

```bash
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv sync --group dev --no-install-project
uv run python -m py_compile python/lance_cuvs/__init__.py
```

Run the Python smoke on a GPU-capable machine with:

```bash
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv sync --group dev --no-install-project
eval "$(uv run python tools/rapids_env.py --format shell)"
uv run maturin develop --release
uv run pytest -q tests/test_smoke.py
```

## Rust Layout

- `src/backend.rs`: public backend API and Lance-side orchestration
- `src/cuda.rs`: CUDA / cuVS wrappers and tensor helpers
- `src/python.rs`: PyO3 bindings
- `python/lance_cuvs/__init__.py`: thin Python package surface
