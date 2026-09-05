"""Deterministic file manifests for immutable data provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypedDict

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ManifestFile(TypedDict, total=False):
    """One file entry in a data manifest."""

    path: str
    size_bytes: int
    sha256: str


class DataManifest(TypedDict):
    """Stable manifest schema for a directory tree."""

    schema_version: int
    file_count: int
    total_bytes: int
    files: list[ManifestFile]


def sha256_file(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 digest for ``path`` using bounded memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, *, include_hashes: bool = False) -> DataManifest:
    """Build a path-stable manifest for all regular, non-symlink files under ``root``."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"manifest root must be a directory: {root}")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    files: list[ManifestFile] = []
    total_bytes = 0

    for path in paths:
        size_bytes = path.stat().st_size
        entry: ManifestFile = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": size_bytes,
        }
        if include_hashes:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
        total_bytes += size_bytes

    return {
        "schema_version": 1,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def write_manifest(manifest: DataManifest, output: Path) -> None:
    """Atomically write ``manifest`` as canonical, deterministic JSON."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
