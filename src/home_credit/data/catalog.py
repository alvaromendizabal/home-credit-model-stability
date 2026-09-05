"""Bounded-memory cataloging for the immutable Home Credit raw snapshot."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from home_credit.data.loader import RawManifestRecord, S3RawStore, parquet_records
from home_credit.data.schema import (
    Split,
    TableIdentity,
    parse_table_identity,
    schema_fingerprint,
)
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer

_CASE_ID = "case_id"
_CASE_BATCH_SIZE = 262_144


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Structural and cardinality profile for one raw Parquet object."""

    file: str
    s3_key: str
    source_bytes: int
    source_sha256: str
    split: Split
    family: str
    depth: int
    shard: int | None
    rows: int
    columns: int
    row_groups: int
    schema_sha256: str
    column_names: tuple[str, ...]
    column_types: tuple[str, ...]
    numeric_columns: int
    categorical_columns: int
    temporal_columns: int
    case_id_present: bool
    case_id_null_count: int
    unique_case_ids: int
    max_rows_per_case: int
    null_statistics_columns: int
    max_missing_fraction: float | None

    @property
    def identity(self) -> TableIdentity:
        """Return the parsed table identity for this entry."""
        return TableIdentity(
            split=self.split,
            family=self.family,
            depth=self.depth,
            shard=self.shard,
        )


@dataclass(frozen=True, slots=True)
class TableSummary:
    """Summary of one split-aware logical table across its Parquet shards."""

    split: Split
    family: str
    depth: int
    files: int
    rows: int
    source_bytes: int
    schema_sha256: tuple[str, ...]
    unique_schema_count: int


def _schema_fields(schema: Any) -> list[tuple[str, str, bool]]:
    return [(field.name, str(field.type), bool(field.nullable)) for field in schema]


def _column_type_counts(schema: Any) -> tuple[int, int, int]:
    numeric = 0
    categorical = 0
    temporal = 0
    for field in schema:
        data_type = field.type
        if (
            pa.types.is_integer(data_type)
            or pa.types.is_floating(data_type)
            or pa.types.is_decimal(data_type)
        ):
            numeric += 1
        elif (
            pa.types.is_string(data_type)
            or pa.types.is_large_string(data_type)
            or pa.types.is_dictionary(data_type)
            or pa.types.is_boolean(data_type)
        ):
            categorical += 1
        elif (
            pa.types.is_date(data_type)
            or pa.types.is_timestamp(data_type)
            or pa.types.is_time(data_type)
            or pa.types.is_duration(data_type)
        ):
            temporal += 1
    return numeric, categorical, temporal


def _null_statistics(metadata: Any, rows: int) -> tuple[int, float | None]:
    known_columns = 0
    max_fraction: float | None = None
    for column_index in range(metadata.num_columns):
        total_nulls = 0
        known = True
        for row_group_index in range(metadata.num_row_groups):
            statistics = metadata.row_group(row_group_index).column(column_index).statistics
            if statistics is None or statistics.null_count is None:
                known = False
                break
            total_nulls += int(statistics.null_count)
        if not known:
            continue
        known_columns += 1
        fraction = 0.0 if rows == 0 else total_nulls / rows
        if max_fraction is None or fraction > max_fraction:
            max_fraction = fraction
    return known_columns, max_fraction


def _case_id_profile(parquet_file: Any, schema: Any) -> tuple[bool, int, int, int]:
    if _CASE_ID not in schema.names:
        return False, 0, 0, 0

    counts: Counter[int] = Counter()
    null_count = 0
    for batch in parquet_file.iter_batches(
        batch_size=_CASE_BATCH_SIZE,
        columns=[_CASE_ID],
        use_threads=True,
    ):
        values = batch.column(0)
        null_count += int(values.null_count)
        if len(values) == values.null_count:
            continue
        non_null = pc.drop_null(values)
        value_counts = pc.value_counts(non_null)
        batch_values = value_counts.field("values").to_pylist()
        batch_counts = value_counts.field("counts").to_pylist()
        for case_id, count in zip(batch_values, batch_counts, strict=True):
            counts[int(case_id)] += int(count)

    max_rows = max(counts.values(), default=0)
    return True, null_count, len(counts), max_rows


def inspect_parquet(
    record: RawManifestRecord,
    *,
    open_input_file: Callable[[], Any],
    current_size: int,
) -> CatalogEntry:
    """Inspect one Parquet source with bounded memory and exact case-id cardinality."""
    if current_size != record.bytes:
        raise ValueError(
            f"S3 size does not match locked manifest for {record.file}: "
            f"manifest={record.bytes} current={current_size}"
        )

    identity = parse_table_identity(record.file)
    with open_input_file() as stream:
        parquet_file = pq.ParquetFile(stream)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow
        fields = _schema_fields(schema)
        rows = int(metadata.num_rows)
        numeric, categorical, temporal = _column_type_counts(schema)
        known_null_columns, max_missing_fraction = _null_statistics(metadata, rows)
        case_present, case_nulls, unique_cases, max_rows_per_case = _case_id_profile(
            parquet_file,
            schema,
        )

    return CatalogEntry(
        file=record.file,
        s3_key=record.s3_key,
        source_bytes=record.bytes,
        source_sha256=record.sha256,
        split=identity.split,
        family=identity.family,
        depth=identity.depth,
        shard=identity.shard,
        rows=rows,
        columns=len(schema),
        row_groups=int(metadata.num_row_groups),
        schema_sha256=schema_fingerprint(fields),
        column_names=tuple(name for name, _, _ in fields),
        column_types=tuple(type_name for _, type_name, _ in fields),
        numeric_columns=numeric,
        categorical_columns=categorical,
        temporal_columns=temporal,
        case_id_present=case_present,
        case_id_null_count=case_nulls,
        unique_case_ids=unique_cases,
        max_rows_per_case=max_rows_per_case,
        null_statistics_columns=known_null_columns,
        max_missing_fraction=max_missing_fraction,
    )


def summarize_tables(entries: Iterable[CatalogEntry]) -> list[TableSummary]:
    """Aggregate file-level catalog entries into split-aware logical tables."""
    grouped: dict[tuple[Split, str, int], list[CatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.split, entry.family, entry.depth), []).append(entry)

    summaries: list[TableSummary] = []
    for (split, family, depth), group in sorted(grouped.items()):
        schemas = tuple(sorted({entry.schema_sha256 for entry in group}))
        summaries.append(
            TableSummary(
                split=split,
                family=family,
                depth=depth,
                files=len(group),
                rows=sum(entry.rows for entry in group),
                source_bytes=sum(entry.source_bytes for entry in group),
                schema_sha256=schemas,
                unique_schema_count=len(schemas),
            )
        )
    return summaries


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_file_catalog(entries: Sequence[CatalogEntry], output: Path) -> None:
    """Atomically write the deterministic file catalog as JSONL."""
    payload = "".join(
        json.dumps(asdict(entry), sort_keys=True, default=list) + "\n"
        for entry in sorted(entries, key=lambda item: item.file)
    )
    _atomic_write_text(output, payload)


def write_table_catalog(entries: Sequence[CatalogEntry], output: Path) -> None:
    """Atomically write split-aware logical table summaries as canonical JSON."""
    summaries = summarize_tables(entries)
    payload = (
        json.dumps(
            [asdict(summary) for summary in summaries],
            indent=2,
            sort_keys=True,
            default=list,
        )
        + "\n"
    )
    _atomic_write_text(output, payload)


def write_json(payload: object, output: Path) -> None:
    """Atomically write one canonical JSON artifact."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n"
    _atomic_write_text(output, text)


def _bind_input_file(
    store: S3RawStore,
    key: str,
) -> Callable[[], Any]:
    """Bind an S3 object key to a typed zero-argument opener."""

    def open_input_file() -> Any:
        return store.open_input_file(key)

    return open_input_file


def build_s3_catalog(
    records: Sequence[RawManifestRecord],
    store: S3RawStore,
    *,
    logger: RunLogger,
) -> list[CatalogEntry]:
    """Build the complete Parquet catalog with progress events and heartbeats."""
    selected = list(parquet_records(records))
    total = len(selected)
    entries: list[CatalogEntry] = []
    started = time.monotonic()
    logger.event("catalog_started", files=total, bucket=store.bucket, region=store.region)

    for index, record in enumerate(selected, start=1):
        logger.event("catalog_file_started", index=index, total=total, file=record.file)
        with StageTimer(logger, f"catalog_file_{index}", heartbeat_seconds=30):
            entry = inspect_parquet(
                record,
                open_input_file=_bind_input_file(store, record.s3_key),
                current_size=store.verify_record(record),
            )
        entries.append(entry)
        logger.event(
            "catalog_file_completed",
            index=index,
            total=total,
            file=record.file,
            rows=entry.rows,
            columns=entry.columns,
            unique_case_ids=entry.unique_case_ids,
            max_rows_per_case=entry.max_rows_per_case,
        )

    logger.event(
        "catalog_completed",
        files=len(entries),
        total_rows=sum(entry.rows for entry in entries),
        total_elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return entries
