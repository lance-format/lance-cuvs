#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ci_common.sh"

sync_dev_env

log_step "Prepare dist output"
rm -rf "$ROOT_DIR/dist"
mkdir -p "$ROOT_DIR/dist"

log_step "Check Python package syntax"
(
  cd "$ROOT_DIR"
  .venv/bin/python -m py_compile \
    python/lance_cuvs/__init__.py \
    python/lance_cuvs/_loader.py \
    backends/cuvs_26_02/python/lance_cuvs_backend_cuvs_26_02/__init__.py
)

log_step "Run loader tests"
(
  cd "$ROOT_DIR"
  .venv/bin/pytest -q tests/test_loader.py
)

log_step "Build root wheel"
(
  cd "$ROOT_DIR"
  uv build --wheel --out-dir dist
)

export_rapids_env

log_step "Build backend wheel"
(
  cd "$ROOT_DIR/backends/cuvs_26_02"
  ../../.venv/bin/maturin build --release --locked --out ../../dist
)
