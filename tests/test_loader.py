# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import types

import pytest

from lance_cuvs import _loader


def test_backend_key_for_runtime() -> None:
    assert _loader.backend_key_for_runtime("26.2.0") == "cuvs-26-02"


def test_normalize_backend_override() -> None:
    assert _loader._normalize_backend_key("cuvs-26-02") == "cuvs-26-02"
    assert _loader._normalize_backend_key("cuvs_26_02") == "cuvs-26-02"
    assert (
        _loader._normalize_backend_key("lance-cuvs-backend-cuvs-26-02")
        == "cuvs-26-02"
    )


def test_resolve_backend_from_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCE_CUVS_BACKEND", raising=False)
    monkeypatch.setattr(_loader, "detect_cuvs_runtime_version", lambda: "26.2.0")

    spec = _loader.resolve_backend()

    assert spec.key == "cuvs-26-02"
    assert spec.module == "lance_cuvs_backend_cuvs_26_02"


def test_unsupported_runtime_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCE_CUVS_BACKEND", raising=False)
    monkeypatch.setattr(_loader, "detect_cuvs_runtime_version", lambda: "26.4.0")

    with pytest.raises(ImportError, match="no lance-cuvs backend"):
        _loader.resolve_backend()


def test_load_backend_imports_selected_module(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = types.SimpleNamespace(
        IvfPqArtifactOutput=object,
        IvfPqTrainingOutput=object,
        build_ivf_pq_artifact=lambda *args, **kwargs: (args, kwargs),
        train_ivf_pq=lambda *args, **kwargs: (args, kwargs),
    )

    monkeypatch.setenv("LANCE_CUVS_BACKEND", "cuvs-26-02")
    monkeypatch.setattr(_loader, "_BACKEND", None)
    monkeypatch.setattr(_loader, "_preload_shared_libraries", lambda: None)
    monkeypatch.setattr(_loader.importlib, "import_module", lambda name: backend)

    loaded = _loader.load_backend()

    assert loaded is backend
