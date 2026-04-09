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

export_local_python() {
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
  export CMAKE="$ROOT_DIR/.venv/bin/cmake"
}

export_rapids_env() {
  log_step "Export RAPIDS build environment"
  export_local_python
  local shell_env
  shell_env="$("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/tools/rapids_env.py" --format shell)"
  # shellcheck disable=SC1090
  source /dev/stdin <<<"$shell_env"
}

emit_github_env() {
  export_local_python
  {
    printf 'CMAKE=%s\n' "$ROOT_DIR/.venv/bin/cmake"
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/tools/rapids_env.py" --format github-env
  }
}
