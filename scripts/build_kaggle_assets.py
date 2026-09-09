#!/usr/bin/env python3
"""Build a hash-locked offline inference dataset from the existing native release."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_file

ROOT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "psutil",
    "boto3",
)


def unpack_bundle(archive: Path, destination: Path) -> list[Path]:
    """Extract a verified model archive while rejecting unsafe or repeated paths."""
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate model archive member")
        for member in stream.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("unsafe model archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("model archive symlink is not allowed")
        stream.extractall(destination)
        return [
            destination / member.filename for member in stream.infolist() if not member.is_dir()
        ]


def locked_wheels(root: Path) -> list[dict[str, Any]]:
    """Resolve the complete locked inference dependency closure for this platform."""
    packages = {p["name"]: p for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    tags = {tag: position for position, tag in enumerate(sys_tags())}
    todo = list(ROOT_PACKAGES)
    selected: dict[str, dict[str, Any]] = {}
    while todo:
        name = todo.pop()
        if name in selected:
            continue
        package = packages[name]
        candidates = []
        for wheel in package["wheels"]:
            wheel_tags = parse_wheel_filename(wheel["url"].rsplit("/", 1)[1])[3]
            ranks = [tags[tag] for tag in wheel_tags if tag in tags]
            if ranks:
                candidates.append((min(ranks), wheel))
        if not candidates:
            raise ValueError(f"no compatible locked wheel for {name}")
        wheel = min(candidates, key=lambda item: item[0])[1]
        selected[name] = {"name": name, "version": package["version"], **wheel}
        todo.extend(dependency["name"] for dependency in package.get("dependencies", []))
    return [selected[name] for name in sorted(selected)]


def build(root: Path, bundle: Path, destination: Path) -> dict[str, Any]:
    """Recover verified downloads, then create a deterministic ZIP with no raw borrower data."""
    state = json.loads((root / "reports/model_release/state.json").read_text())
    expected = state["stages"]["bundle"]
    if sha256_file(bundle) != expected["archive"]["sha256"]:
        raise ValueError("frozen inference archive digest mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    wheels = locked_wheels(root)
    wheel_directory = destination / "wheels"
    wheel_directory.mkdir(exist_ok=True)
    for number, wheel in enumerate(wheels, 1):
        path = wheel_directory / wheel["url"].rsplit("/", 1)[1]
        expected_sha = wheel["hash"].removeprefix("sha256:")
        if not path.is_file() or sha256_file(path) != expected_sha:
            temporary = path.with_suffix(".download")
            with (
                urllib.request.urlopen(wheel["url"], timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            if sha256_file(temporary) != expected_sha:
                raise ValueError(f"locked wheel digest mismatch: {path.name}")
            temporary.replace(path)
        print(f"WHEEL_VERIFIED {number}/{len(wheels)} {path.name}", flush=True)
    (destination / "requirements.txt").write_text(
        "".join(f"{w['name']}=={w['version']} --hash={w['hash']}\n" for w in wheels)
    )
    # Kaggle expands nested ZIP files on upload; publish native members explicitly.
    model_directory = destination / "bundle"
    model_files = unpack_bundle(bundle, model_directory)
    shutil.copy2(root / "scripts/kaggle_inference.py", destination / "kaggle_inference.py")
    # Rebuilds ignore stale downloads, removed members and Python import caches.
    files = sorted(
        [
            *model_files,
            *(wheel_directory / w["url"].rsplit("/", 1)[1] for w in wheels),
            destination / "requirements.txt",
            destination / "kaggle_inference.py",
        ]
    )
    manifest = {
        "schema_version": 1,
        "python": "3.12",
        "platform": "linux_x86_64",
        "bundle_sha256": expected["manifest"]["sha256"],
        "model_fit_commit": state["stages"]["all_labels/lightgbm"]["fit_git_commit"],
        "dependency_lock_sha256": sha256_file(root / "uv.lock"),
        "versions": {w["name"]: w["version"] for w in wheels},
        "files": [
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    (destination / "runtime.json").write_bytes(canonical_json_bytes(manifest))
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted([*files, destination / "runtime.json"]):
            info = zipfile.ZipInfo(path.relative_to(destination).as_posix())
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, path.read_bytes())
    receipt = {
        "schema_version": 1,
        "runtime_sha256": sha256_file(destination / "runtime.json"),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "wheel_count": len(wheels),
        "model_fits": 0,
    }
    archive.with_suffix(".json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.bundle, args.destination)


if __name__ == "__main__":
    main()
