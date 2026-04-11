"""Backend selection for CUDA runtime-specific cuVS integrations."""

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
    runtime_distribution: str
    runtime_major: int
    runtime_minor: int


@dataclass(frozen=True)
class RuntimeSpec:
    distribution: str
    version: str


SUPPORTED_BACKENDS: dict[str, BackendSpec] = {
    "cu12": BackendSpec(
        key="cu12",
        module="lance_cuvs_backend_cu12",
        distribution="pylance-cuvs-cu12",
        runtime_distribution="libcuvs-cu12",
        runtime_major=26,
        runtime_minor=2,
    ),
}

_LEGACY_BACKEND_KEYS = {
    "cuvs_26_02": "cu12",
    "cuvs26_02": "cu12",
    "cuvs2602": "cu12",
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

    candidate_groups = [
        ("libcudart", ("libcudart.so.12", "libcudart.so")),
        ("librapids_logger", ("librapids_logger.so",)),
        ("librmm", ("librmm.so",)),
        ("libraft", ("libraft.so",)),
        ("libcuvs_c", ("libcuvs_c.so",)),
    ]
    loaded: list[str] = []
    failures: list[str] = []

    for _, names in candidate_groups:
        loaded_this_group = False
        for root in _shared_library_roots():
            paths: list[Path] = []
            seen_paths: set[Path] = set()
            for name in names:
                for path in [root / name, *sorted(root.glob(f"{name}*"))]:
                    if not path.is_file() or path in seen_paths:
                        continue
                    seen_paths.add(path)
                    paths.append(path)

            for path in paths:
                try:
                    ctypes.CDLL(os.fspath(path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.fspath(path))
                    loaded_this_group = True
                    break
                except OSError as exc:
                    failures.append(f"{path}: {exc}")

            if loaded_this_group:
                break

    _PRELOAD_DONE = True
    if not loaded and failures:
        raise ImportError(
            "Failed to preload required cuVS shared libraries. "
            + " | ".join(failures[:8])
        )


def _supported_backend_summary() -> str:
    return ", ".join(
        f"{spec.key} ({spec.runtime_distribution} {spec.runtime_major}.{spec.runtime_minor:02d})"
        for spec in sorted(SUPPORTED_BACKENDS.values(), key=lambda item: item.key)
    )


def _normalize_backend_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    for prefix in (
        "pylance_cuvs_backend_",
        "lance_cuvs_backend_",
        "pylance_cuvs_",
    ):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break

    if normalized in SUPPORTED_BACKENDS:
        return normalized
    if normalized in _LEGACY_BACKEND_KEYS:
        return _LEGACY_BACKEND_KEYS[normalized]

    match = re.fullmatch(r"cu(\d+)", normalized)
    if match:
        return f"cu{int(match.group(1))}"

    supported = _supported_backend_summary()
    raise ImportError(
        f"Unsupported LANCE_CUVS_BACKEND={value!r}. Supported backends: {supported}"
    )


def detect_cuvs_runtime() -> RuntimeSpec | None:
    for distribution in ("libcuvs-cu13", "libcuvs-cu12", "libcuvs", "cuvs"):
        try:
            return RuntimeSpec(
                distribution=distribution,
                version=importlib.metadata.version(distribution),
            )
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def backend_key_for_runtime(runtime: RuntimeSpec) -> str:
    for key, spec in SUPPORTED_BACKENDS.items():
        if runtime.distribution == spec.runtime_distribution:
            return key

    if runtime.distribution in {"libcuvs", "cuvs"}:
        raise ImportError(
            f"Detected cuVS runtime package {runtime.distribution}=={runtime.version}, "
            "but its CUDA runtime line is ambiguous. Install a runtime-specific "
            "package such as libcuvs-cu12, or set LANCE_CUVS_BACKEND explicitly."
        )

    supported = _supported_backend_summary()
    raise ImportError(
        f"Detected cuVS runtime package {runtime.distribution}=={runtime.version}, "
        f"but no pylance-cuvs backend is available for it. Supported backends: {supported}"
    )


def resolve_backend() -> BackendSpec:
    override = os.environ.get("LANCE_CUVS_BACKEND")
    if override:
        return SUPPORTED_BACKENDS[_normalize_backend_key(override)]

    runtime = detect_cuvs_runtime()
    if runtime is None:
        supported = _supported_backend_summary()
        raise ImportError(
            "Unable to detect an installed cuVS runtime. Install a supported "
            "libcuvs Python package and matching pylance-cuvs backend, or set "
            f"LANCE_CUVS_BACKEND explicitly. Supported backends: {supported}"
        )

    return SUPPORTED_BACKENDS[backend_key_for_runtime(runtime)]


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
                f"Detected CUDA backend {spec.key}, but Python package "
                f"{spec.distribution!r} is not installed."
            ) from exc
        raise

    return _BACKEND
