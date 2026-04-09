set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
uv_project := "uv run --project '" + root + "' --no-sync"
cmake_cmd := uv_project + " python -c 'import shutil; print(shutil.which(\"cmake\") or \"\")'"
rapids_env := "export CMAKE=\"$(" + cmake_cmd + ")\"; eval \"$(" + uv_project + " python tools/rapids_env.py --format shell)\""

default:
  @just --list

sync-dev:
  uv sync --group dev

sync-dev-no-project:
  uv sync --group dev --no-install-project

loader-test: sync-dev
  {{uv_project}} pytest -q tests/test_loader.py

backend-wheel: sync-dev
  rm -rf dist
  mkdir -p dist
  {{rapids_env}}
  cd backends/cuvs_26_02
  {{uv_project}} maturin build --release --locked --out ../../dist

backend-develop: sync-dev
  {{rapids_env}}
  cd backends/cuvs_26_02
  {{uv_project}} maturin develop --release --locked

python-build: sync-dev
  rm -rf dist
  mkdir -p dist
  {{uv_project}} python -m py_compile \
    python/lance_cuvs/__init__.py \
    python/lance_cuvs/_loader.py \
    backends/cuvs_26_02/python/lance_cuvs_backend_cuvs_26_02/__init__.py
  {{uv_project}} pytest -q tests/test_loader.py
  uv build --wheel --out-dir dist
  {{rapids_env}}
  cd backends/cuvs_26_02
  {{uv_project}} maturin build --release --locked --out ../../dist

rust-build: sync-dev-no-project
  {{rapids_env}}
  cargo build --manifest-path backends/cuvs_26_02/Cargo.toml --locked --all-targets

python-release: sync-dev
  rm -rf dist
  mkdir -p dist
  {{rapids_env}}
  uv build --wheel --out-dir dist
  uv build --sdist --out-dir dist
  cd backends/cuvs_26_02
  {{uv_project}} maturin build --release --locked --out ../../dist
  {{uv_project}} maturin sdist --out ../../dist

gpu-smoke: sync-dev
  {{rapids_env}}
  cd backends/cuvs_26_02
  {{uv_project}} maturin develop --release --locked
  cd {{root}}
  LANCE_CUVS_REQUIRE_GPU="${LANCE_CUVS_REQUIRE_GPU:-1}" {{uv_project}} pytest -q tests/test_smoke.py

container-shell:
  tools/run_in_container.sh -- bash

container-python-build:
  tools/run_in_container.sh -- just python-build

container-rust-build:
  tools/run_in_container.sh -- just rust-build

container-python-release:
  tools/run_in_container.sh -- just python-release

container-gpu-smoke:
  tools/run_in_container.sh --gpu -- just gpu-smoke

container-backend-wheel:
  tools/run_in_container.sh -- just backend-wheel
