# AGENTS.md

## Scope

`lance-cuvs` is a narrow backend loader project:

- It exposes a stable Python API for Lance's cuVS integration.
- It selects a versioned native backend based on the installed cuVS runtime.
- It trains IVF_PQ models with cuVS.
- It encodes a Lance dataset into a partition-local artifact.
- It does **not** finalize or register Lance indices on behalf of callers.

Keep that boundary intact. Do not reintroduce `pylance` convenience wrappers or Lance-side finalization helpers into this repository.

## Architecture

- `python/lance_cuvs`
  - Pure Python loader.
  - cuVS runtime detection.
  - Backend dispatch.
- `backends/cuvs_26_02/src/backend.rs`
  - Lance-facing orchestration.
  - Public backend API types.
  - Training and artifact build entrypoints.
- `backends/cuvs_26_02/src/cuda.rs`
  - CUDA and cuVS low-level wrappers.
  - Device tensors, pinned host buffers, DLPack tensor views.
- `backends/cuvs_26_02/src/python.rs`
  - PyO3 bindings only.
  - No build logic beyond argument conversion and runtime bridging.
- `backends/cuvs_26_02/python/lance_cuvs_backend_cuvs_26_02`
  - Version-specific Python package surface for cuVS `26.02`.

## API Rules

- Keep the public Python API Arrow-first.
  - Training outputs must stay Arrow-native.
  - Do not add NumPy-only outputs back.
- Keep Python bindings thin.
  - Validation and build logic belong in Rust.
- Prefer stable, explicit outputs over convenience orchestration.
  - `train_ivf_pq(...)`
  - `build_ivf_pq_artifact(..., training=...)`
- Keep the root package version-agnostic.
  - No direct `cuvs-sys` dependency in the root package.
  - No root `_native` extension module.

## Dependency Rules

- The root package must stay pure Python.
- Version-pinned cuVS bindings belong only in versioned backend packages.
- The current backend baseline is `cuvs == 26.2.0`.
- This project currently depends on unreleased Lance build APIs.
- Use Lance git dependencies in source control.
- For local or EC2 validation against an unmerged Lance checkout, patch dependencies to path-based Lance crates outside the committed tree.
- Do not commit machine-specific path dependencies.

## Build and Test

### Task runner

```bash
just
```

### Shared development container

```bash
just container-shell
```

Add `--platform linux/amd64` only when you specifically need to mirror GitHub-hosted runner architecture.

### Root development environment

```bash
just sync-dev
```

### Loader tests

```bash
just loader-test
```

### Root package build

```bash
just python-build
```

### Backend package build

```bash
just backend-wheel
just backend-develop
```

### CI-equivalent container validation

```bash
just container-python-build
just container-rust-build
just container-python-release
```

### Smoke expectation

The minimal smoke should verify:

1. `train_ivf_pq(...)` succeeds.
2. Training outputs are Arrow arrays.
3. `build_ivf_pq_artifact(...)` succeeds.
4. The returned artifact files are materialized on disk.

### GPU smoke in container

```bash
just container-gpu-smoke
```

## CUDA / RAPIDS Notes

- Python `3.12` is the current expected baseline.
- The root package detects cuVS runtime versions from installed Python package metadata.
- The current supported backend key is `cuvs-26-02`.
- The build requires CUDA toolkit, cuVS wheel-provided CMake packages, and dynamic libraries to be discoverable.
- When debugging builds, separate:
  - loader / backend selection problems
  - backend source/layout problems
  - Lance dependency baseline problems
  - CUDA / linker environment problems

Do not "fix" environment issues by committing machine-specific linker paths into source files.
