#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ci_common.sh"

sync_dev_env
export_rapids_env

log_step "Build backend extension in place"
(
  cd "$ROOT_DIR/backends/cuvs_26_02"
  ../../.venv/bin/maturin develop --release --locked
)

log_step "Run GPU smoke test"
(
  cd "$ROOT_DIR"
  LANCE_CUVS_REQUIRE_GPU="${LANCE_CUVS_REQUIRE_GPU:-1}" \
    .venv/bin/pytest -q tests/test_smoke.py
)
