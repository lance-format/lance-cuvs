#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${LANCE_CUVS_CONTAINER_IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu24.04}"
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
    export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      libprotobuf-dev \
      libssl-dev \
      patchelf \
      pkg-config \
      protobuf-compiler
    if ! command -v cargo >/dev/null 2>&1; then
      curl -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal
      rustup toolchain install stable
    fi
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    uv python install 3.12
    exec "$@"
  ' bash "$@"
