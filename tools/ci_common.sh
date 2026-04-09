#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT_DIR
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

log_step() {
  printf '\n==> %s\n' "$*"
}

sync_dev_env() {
  log_step "Sync development environment"
  (
    cd "$ROOT_DIR"
    uv sync --group dev "$@"
  )
}

uv_project() {
  uv run --project "$ROOT_DIR" --no-sync "$@"
}

export_rapids_env() {
  log_step "Export RAPIDS build environment"
  export CMAKE
  CMAKE="$(uv_project python -c 'import shutil; print(shutil.which("cmake") or "")')"
  local shell_env
  shell_env="$(uv_project python "$ROOT_DIR/tools/rapids_env.py" --format shell)"
  # shellcheck disable=SC1090
  source /dev/stdin <<<"$shell_env"
}

emit_github_env() {
  {
    printf 'CMAKE=%s\n' "$(uv_project python -c 'import shutil; print(shutil.which("cmake") or "")')"
    uv_project python "$ROOT_DIR/tools/rapids_env.py" --format github-env
  }
}
