#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ci_common.sh"

sync_dev_env
export_rapids_env

log_step "Prepare dist output"
rm -rf "$ROOT_DIR/dist"
mkdir -p "$ROOT_DIR/dist"

log_step "Build root wheel"
(
  cd "$ROOT_DIR"
  uv build --wheel --out-dir dist
)

log_step "Build root source distribution"
(
  cd "$ROOT_DIR"
  uv build --sdist --out-dir dist
)

log_step "Build backend wheel"
(
  cd "$ROOT_DIR/backends/cuvs_26_02"
  ../../.venv/bin/maturin build --release --locked --out ../../dist
)

log_step "Build backend source distribution"
(
  cd "$ROOT_DIR/backends/cuvs_26_02"
  ../../.venv/bin/maturin sdist --out ../../dist
)
