#!/usr/bin/env bash

set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
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
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y \
    bash \
    ca-certificates \
    clang \
    cmake \
    curl \
    gcc \
    gcc-c++ \
    git \
    openssl-devel \
    patchelf \
    pkgconf-pkg-config \
    protobuf-compiler \
    protobuf-devel
elif command -v yum >/dev/null 2>&1; then
  yum install -y \
    bash \
    ca-certificates \
    clang \
    cmake \
    curl \
    gcc \
    gcc-c++ \
    git \
    openssl-devel \
    patchelf \
    pkgconfig \
    protobuf-compiler \
    protobuf-devel
else
  echo "Unsupported package manager" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="${HOME}/.cargo/bin:${PATH}"
  rustup toolchain install stable
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v just >/dev/null 2>&1; then
  cargo install just --locked
fi

if command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
  if [[ ! -x /usr/local/cuda/bin/nvcc && ! -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    rpm --import https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/D42D0685.pub
    if command -v dnf >/dev/null 2>&1; then
      dnf install -y cuda-toolkit-12-9
    else
      yum install -y cuda-toolkit-12-9
    fi
  fi
fi

if [[ -d /opt/python/cp312-cp312/bin ]]; then
  export PATH="/opt/python/cp312-cp312/bin:${PATH}"
fi
