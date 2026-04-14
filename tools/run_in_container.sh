#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${LANCE_CUVS_CONTAINER_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
PLATFORM="${LANCE_CUVS_CONTAINER_PLATFORM:-}"
GPU_ARGS=()
TTY_ARGS=()
CACHE_ROOT="${LANCE_CUVS_CONTAINER_CACHE:-${HOME}/.cache/lance-cuvs-container}"

if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ARGS=(--gpus all)
      shift
      ;;
    --image)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p "${HOME}/.cache/uv" "${HOME}/.cargo/registry" "${HOME}/.cargo/git"
mkdir -p "$CACHE_ROOT/cargo" "$CACHE_ROOT/rustup" "$CACHE_ROOT/local-bin"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

DOCKER_ARGS=(run --rm)
if ((${#TTY_ARGS[@]} > 0)); then
  DOCKER_ARGS+=("${TTY_ARGS[@]}")
fi
if ((${#GPU_ARGS[@]} > 0)); then
  DOCKER_ARGS+=("${GPU_ARGS[@]}")
fi
if [[ -n "$PLATFORM" ]]; then
  DOCKER_ARGS+=(--platform "$PLATFORM")
fi

docker "${DOCKER_ARGS[@]}" \
  -e UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.nvidia.com}" \
  -e CARGO_HOME=/root/.cargo \
  -e RUSTUP_HOME=/root/.rustup \
  -v "$ROOT_DIR:/work" \
  -v "${HOME}/.cache/uv:/root/.cache/uv" \
  -v "$CACHE_ROOT/cargo:/root/.cargo" \
  -v "$CACHE_ROOT/rustup:/root/.rustup" \
  -v "$CACHE_ROOT/local-bin:/root/.local/bin" \
  -w /work \
  "$IMAGE_TAG" \
  bash -lc '
    set -euo pipefail
    export PATH=/root/.cargo/bin:/root/.local/bin:/opt/python/cp312-cp312/bin:$PATH
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/work/backends/cuvs_26_02/target-manylinux_2_28}"
    /work/tools/bootstrap_build_env.sh
    exec "$@"
  ' bash "$@"
