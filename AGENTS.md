# AGENTS.md

## Scope

`lance-cuvs` is a narrow backend package:

- It trains IVF_PQ models with cuVS.
- It encodes a Lance dataset into a partition-local artifact.
- It does **not** finalize or register Lance indices on behalf of callers.

Keep that boundary intact. Do not reintroduce `pylance` convenience wrappers or Lance-side finalization helpers into this repository.

## Architecture

- `src/backend.rs`
  - Lance-facing orchestration.
  - Public backend API types.
  - Training and artifact build entrypoints.
- `src/cuda.rs`
  - CUDA and cuVS low-level wrappers.
  - Device tensors, pinned host buffers, DLPack tensor views.
- `src/python.rs`
  - PyO3 bindings only.
  - No build logic beyond argument conversion and runtime bridging.
- `python/lance_cuvs/__init__.py`
  - Thin Python package surface.
  - Shared library preloading and public docstring-facing wrappers.

## API Rules

- Keep the public Python API Arrow-first.
  - Training outputs must stay Arrow-native.
  - Do not add NumPy-only outputs back.
- Keep Python bindings thin.
  - Validation and build logic belong in Rust.
- Prefer stable, explicit outputs over convenience orchestration.
  - `train_ivf_pq(...)`
  - `build_ivf_pq_artifact(..., training=...)`

## Dependency Rules

- This package currently depends on unreleased Lance build APIs.
- Use Lance git dependencies in source control.
- For local or EC2 validation against an unmerged Lance checkout, patch dependencies to path-based Lance crates outside the committed tree.
- Do not commit machine-specific path dependencies.

## Build and Test

### Local formatting

```bash
cargo fmt --all
```

### Python syntax check

```bash
python3 -m py_compile python/lance_cuvs/__init__.py
```

### Python package build

```bash
maturin develop --release
```

### Smoke expectation

The minimal smoke should verify:

1. `train_ivf_pq(...)` succeeds.
2. Training outputs are Arrow arrays.
3. `build_ivf_pq_artifact(...)` succeeds.
4. A caller can pass the outputs back to Lance finalization.

## CUDA / RAPIDS Notes

- Python 3.12 is the current expected baseline for matching RAPIDS wheels.
- The build requires CUDA toolkit, cuVS wheel-provided CMake packages, and dynamic libraries to be discoverable.
- When debugging builds, separate:
  - source/layout problems
  - Lance dependency baseline problems
  - CUDA / linker environment problems

Do not "fix" environment issues by committing machine-specific linker paths into source files.
