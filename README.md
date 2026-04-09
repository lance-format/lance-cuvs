# lance-cuvs

`lance-cuvs` is an experimental cuVS backend loader for Lance vector indices.

The repository is intentionally split into two layers:

- `lance-cuvs`
  - stable Python entrypoint
  - cuVS runtime detection
  - backend selection
- `lance-cuvs-backend-cuvs-26-02`
  - cuVS `26.02` native implementation
  - IVF_PQ training
  - partition-local artifact build

The public API stays narrow:

- train an `IVF_PQ` model with cuVS
- encode a Lance dataset into a partition-local artifact
- return Arrow-native training outputs

It does **not** finalize or register Lance indices on behalf of callers. Callers are expected to pass the training outputs and artifact URI back into Lance's own index creation APIs.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Status

- Current loader package version: `0.1.0`
- Current backend package version: `0.1.0`
- Current supported runtime backend: `cuvs-26-02`
- Current dependency baseline: Lance git versions that expose unreleased vector-build APIs

## Installation

The root package is only the loader. Install both the loader and a matching backend package.

Current supported backend:

- `lance-cuvs-backend-cuvs-26-02`
- `libcuvs-cu12==26.2.0`

The loader selects the backend from the installed cuVS runtime version. You can override auto-detection with:

```bash
export LANCE_CUVS_BACKEND=cuvs-26-02
```

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
The artifact destination must resolve to the local filesystem.

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

List available tasks with:

```bash
just
```

Run commands inside the shared development container with:

```bash
just container-shell
```

Use `--platform linux/amd64` only when you explicitly want to match the GitHub-hosted runner architecture:

```bash
tools/run_in_container.sh --platform linux/amd64 -- bash
```

Run the CI-equivalent CPU build locally with:

```bash
just container-python-build
```

Run the Rust-only build locally with:

```bash
just container-rust-build
```

Build all release artifacts locally with:

```bash
just container-python-release
```

Create the root development environment with:

```bash
just sync-dev
```

Run loader-only tests with:

```bash
just loader-test
```

Build the root wheel with:

```bash
just python-build
```

Build the backend wheel with:

```bash
just backend-wheel
```

Build the `cuvs-26-02` backend in-place with:

```bash
just backend-develop
```

Build the release distributions with:

```bash
just python-release
```

Run the Python smoke on a GPU-capable machine with:

```bash
just container-gpu-smoke
```

## Repository Layout

- `python/lance_cuvs`
  - root loader package
- `backends/cuvs_26_02`
  - cuVS `26.02` backend package
- `backends/cuvs_26_02/src/backend.rs`
  - Lance-facing orchestration
- `backends/cuvs_26_02/src/cuda.rs`
  - CUDA / cuVS wrappers and tensor helpers
- `backends/cuvs_26_02/src/python.rs`
  - PyO3 bindings
