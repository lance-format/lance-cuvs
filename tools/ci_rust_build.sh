#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ci_common.sh"

sync_dev_env --no-install-project
export_rapids_env

log_step "Build Rust crate"
(
  cd "$ROOT_DIR"
  cargo build --manifest-path backends/cuvs_26_02/Cargo.toml --locked --all-targets
)
