"""Atomic resumable state for long-running temporal model benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    """Immutable provenance that must match before benchmark resume."""

    git_commit: str
    feature_manifest_sha256: str
    validation_protocol_sha256: str
    benchmark_config_sha256: str
    feature_screen_sha256: str
    smoke: bool


@dataclass(frozen=True, slots=True)
class FoldReceipt:
    """One completed model-fold result and its immutable artifacts."""

    model: str
    fold: int
    prediction_path: str
    prediction_sha256: str
    model_path: str
    model_sha256: str
    metrics: dict[str, float | int]


class BenchmarkStateStore:
    """Hash-verifying JSON checkpoint for model-fold completion."""

    def __init__(
        self,
        path: Path,
        *,
        identity: BenchmarkIdentity,
    ) -> None:
        self.path = path
        self.identity = identity
        self._payload = self._load_or_create()

    def receipt(
        self,
        model: str,
        fold: int,
        *,
        output_root: Path,
    ) -> FoldReceipt | None:
        """Return one verified completed receipt or ``None``."""
        raw_folds = self._payload.get("folds", {})
        if not isinstance(raw_folds, dict):
            raise ValueError("benchmark state folds must be an object")
        key = _fold_key(model, fold)
        raw = raw_folds.get(key)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark state receipt must be an object: {key}")
        payload = cast(dict[str, Any], raw)
        receipt = FoldReceipt(
            model=str(payload["model"]),
            fold=int(payload["fold"]),
            prediction_path=str(payload["prediction_path"]),
            prediction_sha256=str(payload["prediction_sha256"]),
            model_path=str(payload["model_path"]),
            model_sha256=str(payload["model_sha256"]),
            metrics={
                str(name): cast(float | int, value)
                for name, value in cast(dict[str, Any], payload["metrics"]).items()
            },
        )
        prediction = _resolve_under_root(output_root, receipt.prediction_path)
        model_path = _resolve_under_root(output_root, receipt.model_path)
        if not prediction.is_file() or not model_path.is_file():
            return None
        if _sha256_file(prediction) != receipt.prediction_sha256:
            raise ValueError(f"benchmark prediction checkpoint hash mismatch: {prediction}")
        if _sha256_file(model_path) != receipt.model_sha256:
            raise ValueError(f"benchmark model checkpoint hash mismatch: {model_path}")
        return receipt

    def record(self, receipt: FoldReceipt) -> None:
        """Persist one completed fold atomically."""
        folds = self._payload.setdefault("folds", {})
        if not isinstance(folds, dict):
            raise ValueError("benchmark state folds must be an object")
        folds[_fold_key(receipt.model, receipt.fold)] = asdict(receipt)
        self._write()

    def completed_receipts(self) -> tuple[FoldReceipt, ...]:
        """Return stored receipts without re-reading model artifacts."""
        raw_folds = self._payload.get("folds", {})
        if not isinstance(raw_folds, dict):
            raise ValueError("benchmark state folds must be an object")
        receipts: list[FoldReceipt] = []
        for key in sorted(raw_folds):
            raw = raw_folds[key]
            if not isinstance(raw, dict):
                raise ValueError(f"benchmark state receipt must be an object: {key}")
            payload = cast(dict[str, Any], raw)
            receipts.append(
                FoldReceipt(
                    model=str(payload["model"]),
                    fold=int(payload["fold"]),
                    prediction_path=str(payload["prediction_path"]),
                    prediction_sha256=str(payload["prediction_sha256"]),
                    model_path=str(payload["model_path"]),
                    model_sha256=str(payload["model_sha256"]),
                    metrics={
                        str(name): cast(float | int, value)
                        for name, value in cast(dict[str, Any], payload["metrics"]).items()
                    },
                )
            )
        return tuple(receipts)

    def _load_or_create(self) -> dict[str, Any]:
        if not self.path.is_file():
            payload: dict[str, Any] = {
                "schema_version": 1,
                "identity": asdict(self.identity),
                "folds": {},
            }
            self._payload = payload
            self._write()
            return payload

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("benchmark state must be a JSON object")
        payload = cast(dict[str, Any], raw)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported benchmark state schema_version")
        observed_identity = payload.get("identity")
        if observed_identity != asdict(self.identity):
            raise ValueError(
                "benchmark checkpoint identity mismatch; refuse unsafe resume: "
                f"expected={asdict(self.identity)} observed={observed_identity}"
            )
        return payload

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        serialized = json.dumps(self._payload, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def relative_artifact(path: Path, *, output_root: Path) -> str:
    """Return an output-root-relative artifact path for checkpoint portability."""
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"benchmark artifact escaped output root: {path}") from exc


def sha256_file(path: Path) -> str:
    """Public SHA-256 helper for benchmark artifacts."""
    return _sha256_file(path)


def _fold_key(model: str, fold: int) -> str:
    if not model:
        raise ValueError("model name must be non-empty")
    if fold < 1:
        raise ValueError("fold must be positive")
    return f"{model}:fold_{fold}"


def _resolve_under_root(root: Path, stored: str) -> Path:
    path = Path(stored)
    resolved = path if path.is_absolute() else root / path
    try:
        resolved.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"benchmark checkpoint escaped output root: {stored}") from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
