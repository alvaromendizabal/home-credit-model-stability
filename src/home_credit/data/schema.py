"""Canonical filename parsing and schema fingerprints for Home Credit raw tables."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

Split = Literal["train", "test"]

_FILE_RE = re.compile(r"^(?P<split>train|test)_(?P<body>.+)\.parquet$")
_DEPTH_TOKENS = frozenset({"0", "1", "2"})


@dataclass(frozen=True, slots=True)
class TableIdentity:
    """Logical identity encoded in one competition Parquet filename."""

    split: Split
    family: str
    depth: int
    shard: int | None

    @property
    def logical_name(self) -> str:
        """Return the split-independent logical table name."""
        return f"{self.family}_depth{self.depth}"

    @property
    def split_logical_name(self) -> str:
        """Return the split-aware logical table name."""
        return f"{self.split}_{self.logical_name}"


def _parse_body(body: str, *, filename: str) -> tuple[str, int, int | None]:
    """Parse ``family_depth[_shard]`` without confusing depth and shard tokens."""
    if body == "base":
        return "base", 0, None

    parts = body.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot infer table depth from filename: {filename}")

    # Sharded files end in ``_<depth>_<shard>``. The final shard can itself be
    # 0, 1, or 2, so the parser must inspect the penultimate token first.
    if len(parts) >= 3 and parts[-2] in _DEPTH_TOKENS and parts[-1].isdigit():
        family = "_".join(parts[:-2])
        if not family:
            raise ValueError(f"missing table family in filename: {filename}")
        return family, int(parts[-2]), int(parts[-1])

    # Unsharded files end in ``_<depth>``.
    if parts[-1] in _DEPTH_TOKENS:
        family = "_".join(parts[:-1])
        if not family:
            raise ValueError(f"missing table family in filename: {filename}")
        return family, int(parts[-1]), None

    raise ValueError(f"cannot infer table depth from filename: {filename}")


def parse_table_identity(filename: str) -> TableIdentity:
    """Parse one official Home Credit Parquet path into its logical identity."""
    basename = filename.rsplit("/", maxsplit=1)[-1]
    match = _FILE_RE.fullmatch(basename)
    if match is None:
        raise ValueError(f"not a Home Credit train/test Parquet filename: {filename}")

    family, depth, shard = _parse_body(match.group("body"), filename=filename)
    return TableIdentity(
        split=cast(Split, match.group("split")),
        family=family,
        depth=depth,
        shard=shard,
    )


def canonical_schema_payload(fields: Sequence[tuple[str, str, bool]]) -> bytes:
    """Return a stable JSON representation for a schema field sequence."""
    payload = [
        {"name": name, "type": type_name, "nullable": nullable}
        for name, type_name, nullable in fields
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def schema_fingerprint(fields: Sequence[tuple[str, str, bool]]) -> str:
    """Return the SHA-256 fingerprint for an ordered schema definition."""
    return hashlib.sha256(canonical_schema_payload(fields)).hexdigest()
