"""Backend selection for versioned cuVS runtime integrations."""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import os
import re
import site
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class BackendSpec:
    key: str
    module: str
    distribution: str
    runtime_major: int
    runtime_minor: int


SUPPORTED_BACKENDS: dict[str, BackendSpec] = {
    "cuvs_26_02": BackendSpec(
        key="cuvs_26_02",
        module="lance_cuvs_backend_cuvs_26_02",
        distribution="lance-cuvs-backend-cuvs_26_02",
        runtime_major=26,
        runtime_minor=2,
    ),
}

_BACKEND: ModuleType | None = None
_PRELOAD_DONE = False


def _iter_site_roots() -> list[Path]:
    roots: list[Path] = []

    try:
        site_packages = list(site.getsitepackages())
    except AttributeError:
        site_packages = []

    user_site = site.getusersitepackages()
    if user_site:
        site_packages.append(user_site)

    for entry in site_packages:
        path = Path(entry)
        if path.is_dir():
            roots.append(path)

    return roots


def _shared_library_roots() -> list[Path]:
    roots: list[Path] = []

    for site_root in _iter_site_roots():
        for package_name in ("rapids_logger", "librmm", "libraft", "libcuvs"):
            for suffix in ("lib64", "lib"):
                candidate = site_root / package_name / suffix
                if candidate.is_dir():
                    roots.append(candidate)
        for candidate in site_root.iterdir():
            if (
                candidate.is_dir()
                and candidate.name.endswith(".libs")
                and any(
                    token in candidate.name.lower()
                    for token in ("cuvs", "raft", "rmm", "rapids")
                )
            ):
                roots.append(candidate)

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home:
            cuda_root = Path(cuda_home)
            for candidate in (
                cuda_root / "targets" / "x86_64-linux" / "lib",
                cuda_root / "lib64",
                cuda_root / "lib",
            ):
                if candidate.is_dir():
                    roots.append(candidate)

    for candidate in (
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.9/lib64"),
    ):
        if candidate.is_dir():
            roots.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def _preload_shared_libraries() -> None:
    global _PRELOAD_DONE
    if _PRELOAD_DONE:
        return

    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "librapids_logger.so",
        "librmm.so",
        "libraft.so",
        "libcuvs_c.so",
    ]
    loaded: list[str] = []
    failures: list[str] = []

    for root in _shared_library_roots():
        for name in candidates:
            for path in sorted(root.glob(f"{name}*")):
                if not path.is_file():
                    continue
                try:
                    ctypes.CDLL(os.fspath(path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.fspath(path))
                except OSError as exc:
                    failures.append(f"{path}: {exc}")

    _PRELOAD_DONE = True
    if not loaded and failures:
        raise ImportError(
            "Failed to preload required cuVS shared libraries. "
            + " | ".join(failures[:8])
        )


def _normalize_backend_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized.startswith("lance_cuvs_backend_"):
        normalized = normalized.removeprefix("lance_cuvs_backend_")

    match = re.fullmatch(r"cuvs[_]?(\d+)[_.]?(\d+)", normalized)
    if match:
        return f"cuvs_{int(match.group(1)):02d}_{int(match.group(2)):02d}"

    if normalized in SUPPORTED_BACKENDS:
        return normalized

    supported = ", ".join(sorted(SUPPORTED_BACKENDS))
    raise ImportError(
        f"Unsupported LANCE_CUVS_BACKEND={value!r}. Supported backends: {supported}"
    )


def detect_cuvs_runtime_version() -> str | None:
    for distribution in ("libcuvs-cu12", "libcuvs-cu11", "libcuvs", "cuvs"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def backend_key_for_runtime(version: str) -> str:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if match is None:
        raise ImportError(f"Unable to parse cuVS runtime version {version!r}")

    key = f"cuvs_{int(match.group(1)):02d}_{int(match.group(2)):02d}"
    if key not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ImportError(
            f"Detected cuVS runtime {version}, but no lance-cuvs backend is "
            f"available for {key}. Supported backends: {supported}"
        )
    return key


def resolve_backend() -> BackendSpec:
    override = os.environ.get("LANCE_CUVS_BACKEND")
    if override:
        return SUPPORTED_BACKENDS[_normalize_backend_key(override)]

    runtime_version = detect_cuvs_runtime_version()
    if runtime_version is None:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ImportError(
            "Unable to detect an installed cuVS runtime. Install a supported "
            "libcuvs Python package and matching lance-cuvs backend, or set "
            f"LANCE_CUVS_BACKEND explicitly. Supported backends: {supported}"
        )

    return SUPPORTED_BACKENDS[backend_key_for_runtime(runtime_version)]


def load_backend() -> ModuleType:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    spec = resolve_backend()
    _preload_shared_libraries()

    try:
        _BACKEND = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        if exc.name == spec.module:
            raise ImportError(
                f"Detected cuVS runtime backend {spec.key}, but Python package "
                f"{spec.distribution!r} is not installed."
            ) from exc
        raise

    return _BACKEND
