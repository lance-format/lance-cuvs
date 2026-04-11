# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from pathlib import Path
import types

import pytest

from lance_cuvs import _loader


def test_backend_key_for_runtime() -> None:
    runtime = _loader.RuntimeSpec(distribution="libcuvs-cu12", version="26.2.0")
    assert _loader.backend_key_for_runtime(runtime) == "cu12"


def test_normalize_backend_override() -> None:
    assert _loader._normalize_backend_key("cu12") == "cu12"
    assert _loader._normalize_backend_key("pylance-cuvs-cu12") == "cu12"
    assert _loader._normalize_backend_key("lance-cuvs-backend-cuvs-26-02") == "cu12"
    assert _loader._normalize_backend_key("cuvs-26-02") == "cu12"
    assert _loader._normalize_backend_key("cuvs_26_02") == "cu12"


def test_resolve_backend_from_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCE_CUVS_BACKEND", raising=False)
    monkeypatch.setattr(
        _loader,
        "detect_cuvs_runtime",
        lambda: _loader.RuntimeSpec(distribution="libcuvs-cu12", version="26.2.0"),
    )

    spec = _loader.resolve_backend()

    assert spec.key == "cu12"
    assert spec.module == "lance_cuvs_backend_cu12"


def test_unsupported_runtime_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCE_CUVS_BACKEND", raising=False)
    monkeypatch.setattr(
        _loader,
        "detect_cuvs_runtime",
        lambda: _loader.RuntimeSpec(distribution="libcuvs-cu13", version="26.2.0"),
    )

    with pytest.raises(ImportError, match="no pylance-cuvs backend"):
        _loader.resolve_backend()


def test_load_backend_imports_selected_module(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = types.SimpleNamespace(
        IvfPqArtifactOutput=object,
        IvfPqTrainingOutput=object,
        build_ivf_pq_artifact=lambda *args, **kwargs: (args, kwargs),
        train_ivf_pq=lambda *args, **kwargs: (args, kwargs),
    )

    monkeypatch.setenv("LANCE_CUVS_BACKEND", "cu12")
    monkeypatch.setattr(_loader, "_BACKEND", None)
    monkeypatch.setattr(_loader, "_preload_shared_libraries", lambda _spec: None)
    monkeypatch.setattr(_loader.importlib, "import_module", lambda name: backend)

    loaded = _loader.load_backend()

    assert loaded is backend


def test_preload_shared_libraries_prefers_first_matching_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    for root in (first, second):
        for name in (
            "libcudart.so.12",
            "librapids_logger.so",
            "librmm.so",
            "libraft.so",
            "libcuvs.so",
            "libcuvs_c.so",
        ):
            (root / name).write_bytes(b"")

    calls: list[str] = []

    def fake_cdll(path: str, *, mode: int) -> object:
        calls.append(path)
        return object()

    monkeypatch.setattr(_loader, "_PRELOAD_DONE", False)
    monkeypatch.setattr(
        _loader, "_backend_private_library_roots", lambda _spec: []
    )
    monkeypatch.setattr(_loader, "_cuda_library_roots", lambda: [first, second])
    monkeypatch.setattr(_loader, "_runtime_library_roots", lambda: [first, second])
    monkeypatch.setattr(_loader.ctypes, "CDLL", fake_cdll)

    _loader._preload_shared_libraries(_loader.SUPPORTED_BACKENDS["cu12"])

    assert calls == [
        str(first / "libcudart.so.12"),
        str(first / "librapids_logger.so"),
        str(first / "librmm.so"),
        str(first / "libraft.so"),
        str(first / "libcuvs.so"),
        str(first / "libcuvs_c.so"),
    ]


def test_preload_shared_libraries_matches_auditwheel_hashed_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cuda_root = tmp_path / "cuda"
    runtime_root = tmp_path / "runtime"
    cuda_root.mkdir()
    runtime_root.mkdir()

    (cuda_root / "libcudart.so.12").write_bytes(b"")
    for name in (
        "librapids_logger-abc123.so.0",
        "librmm-deadbeef.so.26",
        "libraft-facefeed.so.26",
        "libcuvs-c0ffee.so.26",
        "libcuvs_c-badf00d.so.26",
    ):
        (runtime_root / name).write_bytes(b"")

    calls: list[str] = []

    def fake_cdll(path: str, *, mode: int) -> object:
        calls.append(path)
        return object()

    monkeypatch.setattr(_loader, "_PRELOAD_DONE", False)
    monkeypatch.setattr(
        _loader, "_backend_private_library_roots", lambda _spec: []
    )
    monkeypatch.setattr(_loader, "_cuda_library_roots", lambda: [cuda_root])
    monkeypatch.setattr(_loader, "_runtime_library_roots", lambda: [runtime_root])
    monkeypatch.setattr(_loader.ctypes, "CDLL", fake_cdll)

    _loader._preload_shared_libraries(_loader.SUPPORTED_BACKENDS["cu12"])

    assert calls == [
        str(cuda_root / "libcudart.so.12"),
        str(runtime_root / "librapids_logger-abc123.so.0"),
        str(runtime_root / "librmm-deadbeef.so.26"),
        str(runtime_root / "libraft-facefeed.so.26"),
        str(runtime_root / "libcuvs-c0ffee.so.26"),
        str(runtime_root / "libcuvs_c-badf00d.so.26"),
    ]


def test_preload_shared_libraries_skips_private_backend_wheel_libs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cuda_root = tmp_path / "cuda"
    private_root = tmp_path / "pylance_cuvs_cu12.libs"
    cuda_root.mkdir()
    private_root.mkdir()

    (cuda_root / "libcudart.so.12").write_bytes(b"")
    for name in (
        "librapids_logger-abc123.so.0",
        "librmm-deadbeef.so.26",
        "libraft-facefeed.so.26",
        "libcuvs-c0ffee.so.26",
        "libcuvs_c-badf00d.so.26",
    ):
        (private_root / name).write_bytes(b"")

    calls: list[str] = []

    def fake_cdll(path: str, *, mode: int) -> object:
        calls.append(path)
        return object()

    monkeypatch.setattr(_loader, "_PRELOAD_DONE", False)
    monkeypatch.setattr(
        _loader, "_backend_private_library_roots", lambda _spec: [private_root]
    )
    monkeypatch.setattr(_loader, "_cuda_library_roots", lambda: [cuda_root])
    monkeypatch.setattr(_loader, "_runtime_library_roots", lambda: [private_root])
    monkeypatch.setattr(_loader.ctypes, "CDLL", fake_cdll)

    _loader._preload_shared_libraries(_loader.SUPPORTED_BACKENDS["cu12"])

    assert calls == [str(cuda_root / "libcudart.so.12")]
