"""Durable, content-addressed checkpoint manifests for model benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    """One content-addressed file referenced by a checkpoint manifest."""

    path: str
    bytes: int
    sha256: str
    object_key: str


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Immutable description of one fully verified local benchmark state."""

    schema_version: int
    run_key: str
    sequence: int
    completed_model_folds: int
    created_utc: str
    git_commit: str
    feature_manifest_sha256: str
    validation_protocol_sha256: str
    benchmark_config_sha256: str
    smoke: bool
    files: tuple[CheckpointFile, ...]


@dataclass(frozen=True, slots=True)
class CheckpointPointer:
    """Mutable S3 pointer to the latest immutable checkpoint manifest."""

    schema_version: int
    run_key: str
    sequence: int
    manifest_key: str
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize JSON deterministically for hashing and storage."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def derive_run_key(
    *,
    git_commit: str,
    feature_manifest_sha256: str,
    validation_protocol_sha256: str,
    benchmark_config_sha256: str,
    smoke: bool,
) -> str:
    """Derive an immutable run key from benchmark provenance."""
    values = (
        git_commit,
        feature_manifest_sha256,
        validation_protocol_sha256,
        benchmark_config_sha256,
    )
    if any(not value for value in values):
        raise ValueError("run-key provenance values must be non-empty")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "git_commit": git_commit,
                "feature_manifest_sha256": feature_manifest_sha256,
                "validation_protocol_sha256": validation_protocol_sha256,
                "benchmark_config_sha256": benchmark_config_sha256,
                "smoke": smoke,
            }
        )
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an ``s3://bucket/prefix`` URI into bucket and prefix."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return parsed.netloc, prefix


def object_key_for_sha(prefix: str, sha256: str) -> str:
    """Return a content-addressed object key."""
    if len(sha256) != 64:
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError("sha256 must be hexadecimal") from exc
    stem = prefix.rstrip("/")
    relative = f"objects/{sha256[:2]}/{sha256}"
    return f"{stem}/{relative}" if stem else relative


def latest_pointer_key(prefix: str) -> str:
    """Return the mutable pointer key for a checkpoint prefix."""
    stem = prefix.rstrip("/")
    return f"{stem}/latest.json" if stem else "latest.json"


def checkpoint_manifest_key(
    prefix: str,
    *,
    sequence: int,
    manifest_sha256: str,
) -> str:
    """Return an immutable checkpoint-manifest key."""
    if sequence < 0:
        raise ValueError("sequence cannot be negative")
    stem = prefix.rstrip("/")
    name = f"manifests/checkpoint-{sequence:03d}-{manifest_sha256}.json"
    return f"{stem}/{name}" if stem else name


def benchmark_manifest_key(
    prefix: str,
    *,
    manifest_sha256: str,
) -> str:
    """Return an immutable final benchmark-manifest key."""
    stem = prefix.rstrip("/")
    name = f"benchmark-{manifest_sha256}.json"
    return f"{stem}/{name}" if stem else name


def validate_benchmark_state(
    output_root: Path,
    *,
    git_commit: str,
    feature_manifest_sha256: str,
    validation_protocol_sha256: str,
    benchmark_config_sha256: str,
    smoke: bool,
) -> int:
    """Verify state identity and every referenced model/prediction artifact."""
    state_path = output_root / "benchmark_state.json"
    screen_path = output_root / "feature_screen.json"

    if not state_path.is_file():
        return 0
    if not screen_path.is_file():
        raise ValueError("feature_screen.json is missing for benchmark state")

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark state must be a JSON object")
    payload = cast(dict[str, Any], raw)

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported benchmark state schema_version")

    identity_raw = payload.get("identity")
    if not isinstance(identity_raw, dict):
        raise ValueError("benchmark state identity must be an object")
    identity = cast(dict[str, Any], identity_raw)

    expected = {
        "git_commit": git_commit,
        "feature_manifest_sha256": feature_manifest_sha256,
        "validation_protocol_sha256": validation_protocol_sha256,
        "benchmark_config_sha256": benchmark_config_sha256,
        "smoke": smoke,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"benchmark state provenance mismatch: {key}")

    observed_screen_sha = identity.get("feature_screen_sha256")
    if not isinstance(observed_screen_sha, str) or not observed_screen_sha:
        raise ValueError("benchmark state feature_screen_sha256 is invalid")
    if sha256_file(screen_path) != observed_screen_sha:
        raise ValueError("feature screen checkpoint hash mismatch")

    folds_raw = payload.get("folds", {})
    if not isinstance(folds_raw, dict):
        raise ValueError("benchmark state folds must be an object")
    folds = cast(dict[str, Any], folds_raw)

    for key, receipt_raw in folds.items():
        if not isinstance(receipt_raw, dict):
            raise ValueError(f"benchmark receipt must be an object: {key}")
        receipt = cast(dict[str, Any], receipt_raw)
        model = receipt.get("model")
        fold = receipt.get("fold")
        if not isinstance(model, str) or not model:
            raise ValueError(f"invalid model in benchmark receipt: {key}")
        if not isinstance(fold, int) or fold < 1:
            raise ValueError(f"invalid fold in benchmark receipt: {key}")
        if key != f"{model}:fold_{fold}":
            raise ValueError(f"benchmark receipt key mismatch: {key}")

        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("model_path", "model_sha256"),
        ):
            stored = receipt.get(path_key)
            expected_sha = receipt.get(sha_key)
            if not isinstance(stored, str) or not stored:
                raise ValueError(f"invalid {path_key}: {key}")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError(f"invalid {sha_key}: {key}")
            path = _resolve_under_root(output_root, stored)
            if not path.is_file():
                raise ValueError(f"missing benchmark artifact: {path}")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"benchmark artifact hash mismatch: {path}")

    return len(folds)


def build_checkpoint_manifest(
    output_root: Path,
    *,
    run_key: str,
    prefix: str,
    sequence: int,
    completed_model_folds: int,
    git_commit: str,
    feature_manifest_sha256: str,
    validation_protocol_sha256: str,
    benchmark_config_sha256: str,
    smoke: bool,
) -> CheckpointManifest:
    """Build a deterministic manifest over every durable output file."""
    if sequence < 0:
        raise ValueError("sequence cannot be negative")
    if completed_model_folds < 0:
        raise ValueError("completed_model_folds cannot be negative")

    output_root = output_root.resolve()
    members: list[CheckpointFile] = []

    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if _exclude_from_manifest(relative):
            continue
        digest = sha256_file(path)
        members.append(
            CheckpointFile(
                path=relative,
                bytes=path.stat().st_size,
                sha256=digest,
                object_key=object_key_for_sha(prefix, digest),
            )
        )

    return CheckpointManifest(
        schema_version=1,
        run_key=run_key,
        sequence=sequence,
        completed_model_folds=completed_model_folds,
        created_utc=_utc_now(),
        git_commit=git_commit,
        feature_manifest_sha256=feature_manifest_sha256,
        validation_protocol_sha256=validation_protocol_sha256,
        benchmark_config_sha256=benchmark_config_sha256,
        smoke=smoke,
        files=tuple(members),
    )


def manifest_bytes(manifest: CheckpointManifest) -> bytes:
    """Serialize a checkpoint manifest deterministically."""
    return canonical_json_bytes(asdict(manifest))


def pointer_bytes(pointer: CheckpointPointer) -> bytes:
    """Serialize the latest-pointer payload deterministically."""
    return canonical_json_bytes(asdict(pointer))


def load_manifest_bytes(payload: bytes) -> CheckpointManifest:
    """Parse and validate serialized checkpoint-manifest bytes."""
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    data = cast(dict[str, Any], raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest schema_version")

    files_raw = data.get("files")
    if not isinstance(files_raw, list):
        raise ValueError("checkpoint manifest files must be a list")

    files: list[CheckpointFile] = []
    seen_paths: set[str] = set()
    for item_raw in files_raw:
        if not isinstance(item_raw, dict):
            raise ValueError("checkpoint manifest file entry must be an object")
        item = cast(dict[str, Any], item_raw)
        path = str(item["path"])
        _validate_relative_path(path)
        if path in seen_paths:
            raise ValueError(f"duplicate checkpoint path: {path}")
        seen_paths.add(path)
        files.append(
            CheckpointFile(
                path=path,
                bytes=int(item["bytes"]),
                sha256=str(item["sha256"]),
                object_key=str(item["object_key"]),
            )
        )

    return CheckpointManifest(
        schema_version=1,
        run_key=str(data["run_key"]),
        sequence=int(data["sequence"]),
        completed_model_folds=int(data["completed_model_folds"]),
        created_utc=str(data["created_utc"]),
        git_commit=str(data["git_commit"]),
        feature_manifest_sha256=str(data["feature_manifest_sha256"]),
        validation_protocol_sha256=str(data["validation_protocol_sha256"]),
        benchmark_config_sha256=str(data["benchmark_config_sha256"]),
        smoke=bool(data["smoke"]),
        files=tuple(files),
    )


def load_pointer_bytes(payload: bytes) -> CheckpointPointer:
    """Parse and validate the latest-pointer payload."""
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint pointer must be a JSON object")
    data = cast(dict[str, Any], raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint pointer schema_version")
    return CheckpointPointer(
        schema_version=1,
        run_key=str(data["run_key"]),
        sequence=int(data["sequence"]),
        manifest_key=str(data["manifest_key"]),
        manifest_sha256=str(data["manifest_sha256"]),
    )


def verify_checkpoint_manifest(
    output_root: Path,
    manifest: CheckpointManifest,
) -> None:
    """Verify every local file described by a manifest."""
    output_root = output_root.resolve()
    for member in manifest.files:
        path = _resolve_under_root(output_root, member.path)
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {path}")
        if path.stat().st_size != member.bytes:
            raise ValueError(f"checkpoint file size mismatch: {path}")
        if sha256_file(path) != member.sha256:
            raise ValueError(f"checkpoint file hash mismatch: {path}")


def atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes atomically and fsync before rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_under_root(root: Path, stored: str) -> Path:
    _validate_relative_path(stored)
    candidate = (root / stored).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"checkpoint path escaped output root: {stored}") from exc
    return candidate


def _validate_relative_path(stored: str) -> None:
    path = PurePosixPath(stored)
    if not stored or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe checkpoint relative path: {stored}")


def _exclude_from_manifest(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        path.name.endswith(".tmp")
        or path.name == "checkpoint_manifest.json"
        or "__pycache__" in path.parts
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
