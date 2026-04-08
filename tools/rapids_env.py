#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import site
import sys
from pathlib import Path

PACKAGE_NAMES = ("rapids_logger", "librmm", "libraft", "libcuvs")
LIB_SUFFIXES = ("lib64", "lib")
TOKEN_HINTS = ("cuvs", "raft", "rmm", "rapids")


def dedupe(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def discover_roots() -> tuple[list[Path], list[Path]]:
    package_roots: list[Path] = []
    library_roots: list[Path] = []

    for site_packages in site.getsitepackages():
        site_root = Path(site_packages)
        if not site_root.is_dir():
            continue

        for package_name in PACKAGE_NAMES:
            package_root = site_root / package_name
            if not package_root.is_dir():
                continue
            package_roots.append(package_root)
            for suffix in LIB_SUFFIXES:
                candidate = package_root / suffix
                if candidate.is_dir():
                    library_roots.append(candidate)

        for candidate in site_root.iterdir():
            if (
                candidate.is_dir()
                and candidate.name.endswith(".libs")
                and any(token in candidate.name.lower() for token in TOKEN_HINTS)
            ):
                library_roots.append(candidate)

    return dedupe(package_roots), dedupe(library_roots)


def merge_paths(new_paths: list[Path], existing: str | None) -> str:
    values = [os.fspath(path) for path in new_paths]
    if existing:
        values.append(existing)
    return ":".join(values)


def build_env() -> dict[str, str]:
    package_roots, library_roots = discover_roots()
    if not package_roots:
        raise RuntimeError(
            "Failed to locate RAPIDS package roots. "
            "Install the development dependencies first, for example with "
            "`UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv sync --group dev --no-install-project`."
        )
    if not library_roots:
        raise RuntimeError(
            "Failed to locate RAPIDS shared-library roots after dependency installation."
        )

    return {
        "CMAKE_PREFIX_PATH": merge_paths(
            package_roots, os.environ.get("CMAKE_PREFIX_PATH")
        ),
        "LD_LIBRARY_PATH": merge_paths(
            library_roots, os.environ.get("LD_LIBRARY_PATH")
        ),
        "LIBRARY_PATH": merge_paths(library_roots, os.environ.get("LIBRARY_PATH")),
    }


def emit_shell(env: dict[str, str]) -> None:
    for key, value in env.items():
        print(f"export {key}={shlex.quote(value)}")


def emit_github_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        print(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit RAPIDS/CUDA build environment variables for local shells or GitHub Actions."
    )
    parser.add_argument(
        "--format",
        choices=("shell", "github-env"),
        default="shell",
        help="Output format.",
    )
    args = parser.parse_args()

    try:
        env = build_env()
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.format == "shell":
        emit_shell(env)
    else:
        emit_github_env(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
