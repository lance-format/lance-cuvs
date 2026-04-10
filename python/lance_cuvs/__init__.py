"""Version-dispatched public API for `pylance-cuvs`."""

from __future__ import annotations

from typing import Any

from ._loader import load_backend

_BACKEND = None


def _backend() -> Any:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = load_backend()
    return _BACKEND


def train_ivf_pq(*args: Any, **kwargs: Any) -> Any:
    return _backend().train_ivf_pq(*args, **kwargs)


def build_ivf_pq_artifact(*args: Any, **kwargs: Any) -> Any:
    return _backend().build_ivf_pq_artifact(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {"IvfPqArtifactOutput", "IvfPqTrainingOutput"}:
        return getattr(_backend(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(
        {
            "__all__",
            "IvfPqArtifactOutput",
            "IvfPqTrainingOutput",
            "build_ivf_pq_artifact",
            "train_ivf_pq",
        }
    )


__all__ = [
    "IvfPqArtifactOutput",
    "IvfPqTrainingOutput",
    "train_ivf_pq",
    "build_ivf_pq_artifact",
]
