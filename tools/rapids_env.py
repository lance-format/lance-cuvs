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
PACKAGE_DIR_VARS = {
    "cuvs": "cuvs_DIR",
    "raft": "raft_DIR",
    "rmm": "rmm_DIR",
    "rapids_logger": "rapids_logger_DIR",
}
CONFIG_PATTERNS = ("*Config.cmake", "*-config.cmake", "*.cps")


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


def discover_roots() -> tuple[list[Path], list[Path], dict[str, Path]]:
    package_roots: list[Path] = []
    library_roots: list[Path] = []
    package_dirs: dict[str, Path] = {}

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
                    for pattern in CONFIG_PATTERNS:
                        for config_file in candidate.rglob(pattern):
                            config_name = config_file.stem
                            if config_name.endswith("-config"):
                                package_key = config_name[: -len("-config")]
                            elif config_name.endswith("Config"):
                                package_key = config_name[: -len("Config")]
                            else:
                                package_key = config_name
                            package_key = package_key.lower()
                            if package_key in PACKAGE_DIR_VARS:
                                package_dirs.setdefault(package_key, config_file.parent)

        for candidate in site_root.iterdir():
            if (
                candidate.is_dir()
                and candidate.name.endswith(".libs")
                and any(token in candidate.name.lower() for token in TOKEN_HINTS)
            ):
                library_roots.append(candidate)

    return dedupe(package_roots), dedupe(library_roots), package_dirs


def merge_paths(new_paths: list[Path], existing: str | None) -> str:
    values = [os.fspath(path) for path in new_paths]
    if existing:
        values.append(existing)
    return ":".join(values)


def build_env() -> dict[str, str]:
    package_roots, library_roots, package_dirs = discover_roots()
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

    env = {
        "CMAKE_PREFIX_PATH": merge_paths(
            package_roots, os.environ.get("CMAKE_PREFIX_PATH")
        ),
        "LD_LIBRARY_PATH": merge_paths(
            library_roots, os.environ.get("LD_LIBRARY_PATH")
        ),
        "LIBRARY_PATH": merge_paths(library_roots, os.environ.get("LIBRARY_PATH")),
    }
    for package_key, env_name in PACKAGE_DIR_VARS.items():
        package_dir = package_dirs.get(package_key)
        if package_dir is not None:
            env[env_name] = os.fspath(package_dir)
    return env


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
