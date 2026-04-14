#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_PYTHON_VERSION = "0.0.0.dev0"
REPO_CARGO_VERSION = "0.0.0-dev"

TAG_PATTERN = re.compile(
    r"^v(?P<base>\d+\.\d+\.\d+)(?:(?P<phase>a|b|rc)(?P<number>\d+))?$"
)


@dataclass(frozen=True)
class ReleaseVersion:
    python: str
    cargo: str


def parse_release_tag(tag: str) -> ReleaseVersion:
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(
            f"unsupported release tag {tag!r}; expected forms like v0.1.0, v0.1.0b2, or v0.1.0rc1"
        )

    base = match.group("base")
    phase = match.group("phase")
    number = match.group("number")
    python = tag.removeprefix("v")

    if phase is None:
        return ReleaseVersion(python=python, cargo=base)

    cargo_phase = {"a": "alpha", "b": "beta", "rc": "rc"}[phase]
    return ReleaseVersion(python=python, cargo=f"{base}-{cargo_phase}.{number}")


def replace_once(content: str, pattern: str, replacement: str, path: Path) -> str:
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"failed to update {path}")
    return updated


def replace_package_version(
    content: str, package_name: str, version: str, path: Path
) -> str:
    pattern = (
        rf'(\[\[package\]\]\nname = "{re.escape(package_name)}"\nversion = ")'
        rf'([^"\n]+)(")'
    )
    updated, count = re.subn(pattern, rf"\g<1>{version}\3", content, count=1)
    if count != 1:
        raise ValueError(f"failed to update {package_name} version in {path}")
    return updated


def update_file(path: Path, transform) -> None:
    original = path.read_text()
    updated = transform(original)
    if updated != original:
        path.write_text(updated)


def apply_versions(root: Path, version: ReleaseVersion) -> None:
    update_file(
        root / "pyproject.toml",
        lambda content: replace_once(
            content,
            r'^version = "[^"\n]+"$',
            f'version = "{version.python}"',
            root / "pyproject.toml",
        ),
    )
    update_file(
        root / "backends/cuvs_26_02/pyproject.toml",
        lambda content: replace_once(
            content,
            r'^version = "[^"\n]+"$',
            f'version = "{version.python}"',
            root / "backends/cuvs_26_02/pyproject.toml",
        ),
    )
    update_file(
        root / "backends/cuvs_26_02/Cargo.toml",
        lambda content: replace_once(
            content,
            r'^version = "[^"\n]+"$',
            f'version = "{version.cargo}"',
            root / "backends/cuvs_26_02/Cargo.toml",
        ),
    )
    update_file(
        root / "uv.lock",
        lambda content: replace_package_version(
            content, "pylance-cuvs", version.python, root / "uv.lock"
        ),
    )
    update_file(
        root / "backends/cuvs_26_02/Cargo.lock",
        lambda content: replace_package_version(
            content,
            "pylance-cuvs-cu12",
            version.cargo,
            root / "backends/cuvs_26_02/Cargo.lock",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a release version derived from a git tag to local package metadata."
    )
    parser.add_argument("--tag", required=True, help="Git tag like v0.1.0b3")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="Repository root containing the package manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the tag and print derived versions without mutating files.",
    )
    args = parser.parse_args()

    try:
        version = parse_release_tag(args.tag)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 1

    print(f"tag={args.tag}")
    print(f"python_version={version.python}")
    print(f"cargo_version={version.cargo}")

    if not args.dry_run:
        apply_versions(args.root.resolve(), version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
