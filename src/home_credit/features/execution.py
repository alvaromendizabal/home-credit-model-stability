"""Bounded-memory, resumable execution primitives for feature construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from home_credit.data.manifest import sha256_file

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class FeatureExecutionPolicy:
    """Frozen policy controlling bounded-memory feature execution."""

    schema_version: int
    mode: str
    partition_rows: int
    partition_threshold_rows: int
    partition_min_source_bytes: int
    max_threads: int
    resume: bool
    validate_intermediate_hashes: bool
    sort_partitions_by_case_id: bool
    retain_partition_files: bool
    work_directory_name: str
    notes: str

    @classmethod
    def load(cls, path: Path) -> tuple[FeatureExecutionPolicy, str]:
        """Load, validate, and fingerprint one canonical execution policy."""
        payload_bytes = path.read_bytes()
        raw = json.loads(payload_bytes)
        if not isinstance(raw, dict):
            raise ValueError("feature execution policy must be a JSON object")
        payload = cast(dict[str, Any], raw)

        policy = cls(
            schema_version=_required_int(payload, "schema_version"),
            mode=_required_str(payload, "mode"),
            partition_rows=_required_int(payload, "partition_rows"),
            partition_threshold_rows=_required_int(
                payload,
                "partition_threshold_rows",
            ),
            partition_min_source_bytes=_required_int(
                payload,
                "partition_min_source_bytes",
            ),
            max_threads=_required_int(payload, "max_threads"),
            resume=_required_bool(payload, "resume"),
            validate_intermediate_hashes=_required_bool(
                payload,
                "validate_intermediate_hashes",
            ),
            sort_partitions_by_case_id=_required_bool(
                payload,
                "sort_partitions_by_case_id",
            ),
            retain_partition_files=_required_bool(
                payload,
                "retain_partition_files",
            ),
            work_directory_name=_required_str(
                payload,
                "work_directory_name",
            ),
            notes=_required_str(payload, "notes"),
        )
        policy.validate()
        return policy, hashlib.sha256(payload_bytes).hexdigest()

    def validate(self) -> None:
        """Reject execution settings that can defeat memory bounds or resumption."""
        if self.schema_version != 1:
            raise ValueError("unsupported feature execution schema_version")
        if self.mode != "case_range_partitioned":
            raise ValueError("feature execution mode must be case_range_partitioned")
        if self.partition_rows < 1_000:
            raise ValueError("partition_rows must be at least 1000")
        if self.partition_threshold_rows < self.partition_rows:
            raise ValueError(
                "partition_threshold_rows must be greater than or equal to partition_rows"
            )
        if self.partition_min_source_bytes < 1:
            raise ValueError("partition_min_source_bytes must be positive")
        if self.max_threads < 1:
            raise ValueError("max_threads must be positive")
        if not self.resume:
            raise ValueError("resumable execution must remain enabled")
        if not self.validate_intermediate_hashes:
            raise ValueError("intermediate hash validation must remain enabled")
        if not self.sort_partitions_by_case_id:
            raise ValueError("partition outputs must remain deterministically sorted")
        if not self.work_directory_name or "/" in self.work_directory_name:
            raise ValueError("work_directory_name must be one local directory name")


@dataclass(frozen=True, slots=True)
class CasePartition:
    """One disjoint inclusive case-id range derived from the base population."""

    index: int
    case_id_min: int
    case_id_max: int
    expected_base_cases: int

    @property
    def name(self) -> str:
        """Return the deterministic partition name."""
        return f"part-{self.index:05d}"


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    """Immutable identity that prevents checkpoint reuse across incompatible runs."""

    git_commit: str
    raw_manifest_sha256: str
    validation_protocol_sha256: str
    feature_recipe_sha256: str
    feature_execution_sha256: str
    selected_sources: tuple[str, ...]

    def validate(self) -> None:
        """Validate all provenance fingerprints."""
        if not self.git_commit:
            raise ValueError("git_commit must be non-empty")
        for name, value in (
            ("raw_manifest_sha256", self.raw_manifest_sha256),
            ("validation_protocol_sha256", self.validation_protocol_sha256),
            ("feature_recipe_sha256", self.feature_recipe_sha256),
            ("feature_execution_sha256", self.feature_execution_sha256),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.selected_sources:
            raise ValueError("selected_sources must not be empty")


@dataclass(frozen=True, slots=True)
class PartitionReceipt:
    """Integrity receipt for one materialized case partition."""

    index: int
    case_id_min: int
    case_id_max: int
    expected_base_cases: int
    rows: int
    columns: int
    output: str
    output_bytes: int
    output_sha256: str


class BuildStateStore:
    """Atomic JSON checkpoint store for resumable feature builds."""

    def __init__(self, path: Path, identity: BuildIdentity) -> None:
        identity.validate()
        self.path = path
        self.identity = identity
        self._payload = self._load_or_initialize()

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.path.exists():
            payload: dict[str, Any] = {
                "schema_version": 1,
                "identity": asdict(self.identity),
                "blocks": {},
            }
            self._write(payload)
            return payload

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("build state must be a JSON object")
        payload = cast(dict[str, Any], raw)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported build state schema_version")

        current_identity = payload.get("identity")
        expected_identity = asdict(self.identity)
        if current_identity != expected_identity:
            raise ValueError(
                "build state identity does not match the current committed source/data locks"
            )

        blocks = payload.get("blocks")
        if not isinstance(blocks, dict):
            raise ValueError("build state blocks must be a JSON object")
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _block(self, logical_name: str) -> dict[str, Any]:
        blocks = cast(dict[str, Any], self._payload["blocks"])
        value = blocks.setdefault(
            logical_name,
            {
                "partitions": {},
                "feature_block": None,
            },
        )
        if not isinstance(value, dict):
            raise ValueError(f"invalid build state block: {logical_name}")
        return cast(dict[str, Any], value)

    def partition_receipt(
        self,
        logical_name: str,
        partition: CasePartition,
        *,
        output_root: Path,
        validate_hash: bool,
    ) -> PartitionReceipt | None:
        """Return a verified completed partition or ``None`` when it must be rebuilt."""
        block = self._block(logical_name)
        raw_partitions = block.get("partitions")
        if not isinstance(raw_partitions, dict):
            raise ValueError(f"invalid partition state for {logical_name}")
        raw = raw_partitions.get(partition.name)
        if not isinstance(raw, dict):
            return None

        try:
            receipt = PartitionReceipt(
                index=int(raw["index"]),
                case_id_min=int(raw["case_id_min"]),
                case_id_max=int(raw["case_id_max"]),
                expected_base_cases=int(raw["expected_base_cases"]),
                rows=int(raw["rows"]),
                columns=int(raw["columns"]),
                output=str(raw["output"]),
                output_bytes=int(raw["output_bytes"]),
                output_sha256=str(raw["output_sha256"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

        if (
            receipt.index != partition.index
            or receipt.case_id_min != partition.case_id_min
            or receipt.case_id_max != partition.case_id_max
            or receipt.expected_base_cases != partition.expected_base_cases
        ):
            return None

        path = output_root / receipt.output
        if not path.is_file() or path.stat().st_size != receipt.output_bytes:
            return None
        if validate_hash and sha256_file(path) != receipt.output_sha256:
            return None
        return receipt

    def record_partition(
        self,
        logical_name: str,
        receipt: PartitionReceipt,
    ) -> None:
        """Atomically record one completed partition."""
        block = self._block(logical_name)
        raw_partitions = block.setdefault("partitions", {})
        if not isinstance(raw_partitions, dict):
            raise ValueError(f"invalid partition state for {logical_name}")
        raw_partitions[f"part-{receipt.index:05d}"] = asdict(receipt)
        self._write(self._payload)

    def feature_block_payload(
        self,
        logical_name: str,
        *,
        output_root: Path,
        validate_hash: bool,
    ) -> dict[str, Any] | None:
        """Return a verified final feature-block payload when it can be resumed."""
        block = self._block(logical_name)
        raw = block.get("feature_block")
        if not isinstance(raw, dict):
            return None
        output = raw.get("output")
        output_bytes = raw.get("output_bytes")
        output_sha256 = raw.get("output_sha256")
        if not isinstance(output, str) or not isinstance(output_bytes, int):
            return None
        if not isinstance(output_sha256, str):
            return None
        path = output_root / output
        if not path.is_file() or path.stat().st_size != output_bytes:
            return None
        if validate_hash and sha256_file(path) != output_sha256:
            return None
        return cast(dict[str, Any], raw)

    def record_feature_block(
        self,
        logical_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Atomically record one completed final feature block."""
        block = self._block(logical_name)
        block["feature_block"] = dict(payload)
        self._write(self._payload)


def plan_case_partitions(
    case_ids: Sequence[int],
    *,
    partition_rows: int,
) -> tuple[CasePartition, ...]:
    """Create deterministic disjoint inclusive ranges from sorted unique case IDs."""
    if partition_rows <= 0:
        raise ValueError("partition_rows must be positive")
    if not case_ids:
        raise ValueError("case_ids must not be empty")

    normalized = tuple(int(value) for value in case_ids)
    if tuple(sorted(normalized)) != normalized:
        raise ValueError("case_ids must be sorted")
    if len(set(normalized)) != len(normalized):
        raise ValueError("case_ids must be unique")

    partitions: list[CasePartition] = []
    for index, start in enumerate(range(0, len(normalized), partition_rows)):
        chunk = normalized[start : start + partition_rows]
        partitions.append(
            CasePartition(
                index=index,
                case_id_min=chunk[0],
                case_id_max=chunk[-1],
                expected_base_cases=len(chunk),
            )
        )

    for previous, current in pairwise(partitions):
        if previous.case_id_max >= current.case_id_min:
            raise RuntimeError("case partition ranges overlap")

    return tuple(partitions)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"feature execution field {key!r} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"feature execution field {key!r} must be an integer")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"feature execution field {key!r} must be a boolean")
    return value


def should_partition_source(
    *,
    base_rows: int,
    source_bytes: int,
    is_base: bool,
    policy: FeatureExecutionPolicy,
) -> bool:
    """Return whether one source requires exact case-range partitioning."""
    if base_rows < 0 or source_bytes < 0:
        raise ValueError("row and byte counts must be non-negative")
    return (
        not is_base
        and base_rows >= policy.partition_threshold_rows
        and source_bytes >= policy.partition_min_source_bytes
    )
