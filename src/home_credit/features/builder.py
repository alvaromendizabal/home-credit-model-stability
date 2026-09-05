"""Case-level feature-block construction from the immutable raw S3 snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from home_credit.data.loader import RawManifestRecord, S3RawStore
from home_credit.data.manifest import sha256_file
from home_credit.data.schema import Split, parse_table_identity
from home_credit.features.aggregation import aggregate_case_history, build_person_subgroups
from home_credit.features.semantics import (
    CASE_ID,
    DATE_DECISION,
    MONTH,
    NUM_GROUP1,
    NUM_GROUP2,
    TARGET,
    WEEK_NUM,
    SemanticColumns,
    assert_no_target_leakage,
    classify_columns,
    feature_prefix,
)
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer
from home_credit.validation.protocol import verify_protocol_sha256


@dataclass(frozen=True, slots=True)
class FeatureRecipe:
    """Frozen feature-engine recipe loaded from canonical JSON."""

    schema_version: int
    name: str
    engine: str
    compression: str
    scan_batch_rows: int
    time_windows_days: tuple[int, ...]
    person_subgroups: tuple[str, ...]
    numeric_aggregations: tuple[str, ...]
    date_aggregations: tuple[str, ...]
    categorical_aggregations: tuple[str, ...]
    global_frequency_encoding: str
    quantiles: str
    notes: str

    @classmethod
    def load(cls, path: Path) -> tuple[FeatureRecipe, str]:
        """Load and fingerprint one canonical feature recipe."""
        payload_bytes = path.read_bytes()
        raw = json.loads(payload_bytes)
        if not isinstance(raw, dict):
            raise ValueError("feature recipe must be a JSON object")
        payload = cast(dict[str, Any], raw)

        recipe = cls(
            schema_version=_required_int(payload, "schema_version"),
            name=_required_str(payload, "name"),
            engine=_required_str(payload, "engine"),
            compression=_required_str(payload, "compression"),
            scan_batch_rows=_required_int(payload, "scan_batch_rows"),
            time_windows_days=_int_tuple(payload, "time_windows_days"),
            person_subgroups=_str_tuple(payload, "person_subgroups"),
            numeric_aggregations=_str_tuple(payload, "numeric_aggregations"),
            date_aggregations=_str_tuple(payload, "date_aggregations"),
            categorical_aggregations=_str_tuple(payload, "categorical_aggregations"),
            global_frequency_encoding=_required_str(
                payload,
                "global_frequency_encoding",
            ),
            quantiles=_required_str(payload, "quantiles"),
            notes=_required_str(payload, "notes"),
        )
        recipe.validate()
        digest = hashlib.sha256(payload_bytes).hexdigest()
        return recipe, digest

    def validate(self) -> None:
        """Reject unsafe or unsupported recipe settings."""
        if self.schema_version != 1:
            raise ValueError("unsupported feature recipe schema_version")
        if self.engine != "polars_streaming":
            raise ValueError("feature recipe engine must be polars_streaming")
        if self.compression not in {"zstd", "lz4", "snappy"}:
            raise ValueError("unsupported Parquet compression")
        if self.scan_batch_rows < 1:
            raise ValueError("scan_batch_rows must be positive")
        if not self.time_windows_days:
            raise ValueError("at least one time window is required")
        if tuple(sorted(set(self.time_windows_days))) != self.time_windows_days:
            raise ValueError("time_windows_days must be unique and sorted")
        if any(window <= 0 for window in self.time_windows_days):
            raise ValueError("time windows must be positive")
        if self.global_frequency_encoding != "deferred_to_fold_fit":
            raise ValueError("global frequency encoding must remain fold-fitted")


@dataclass(frozen=True, slots=True)
class LogicalSource:
    """One split-aware logical raw table across one or more Parquet shards."""

    split: Split
    family: str
    depth: int
    records: tuple[RawManifestRecord, ...]

    @property
    def logical_name(self) -> str:
        """Return a deterministic logical source name."""
        return f"{self.split}_{self.family}_depth{self.depth}"


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """Metadata for one materialized case-level feature block."""

    split: Split
    family: str
    depth: int
    output: str
    source_files: tuple[str, ...]
    source_sha256: tuple[str, ...]
    rows: int
    columns: int
    feature_columns: int
    output_bytes: int
    output_sha256: str
    recipe_sha256: str
    protocol_sha256: str
    unsupported_columns: tuple[str, ...]


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"feature recipe field {key!r} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"feature recipe field {key!r} must be an integer")
    return value


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"feature recipe field {key!r} must be a list of strings")
    return tuple(cast(list[str], value))


def _int_tuple(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"feature recipe field {key!r} must be a list of integers")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"feature recipe field {key!r} must contain integers")
        result.append(item)
    return tuple(result)


def group_logical_sources(records: Sequence[RawManifestRecord]) -> tuple[LogicalSource, ...]:
    """Group immutable Parquet records by split/family/depth deterministically."""
    grouped: dict[tuple[Split, str, int], list[RawManifestRecord]] = defaultdict(list)

    for record in records:
        if not record.file.endswith(".parquet"):
            continue
        identity = parse_table_identity(record.file)
        grouped[(identity.split, identity.family, identity.depth)].append(record)

    sources = [
        LogicalSource(
            split=split,
            family=family,
            depth=depth,
            records=tuple(sorted(group, key=lambda record: record.file)),
        )
        for (split, family, depth), group in sorted(grouped.items())
    ]

    if not sources:
        raise ValueError("manifest contains no Parquet logical sources")
    return tuple(sources)


def select_sources(
    sources: Sequence[LogicalSource],
    *,
    splits: frozenset[str],
    families: frozenset[str] | None,
) -> tuple[LogicalSource, ...]:
    """Select build sources while always retaining the base table for each split."""
    selected: list[LogicalSource] = []
    for source in sources:
        if source.split not in splits:
            continue
        if source.family == "base" or families is None or source.family in families:
            selected.append(source)
    return tuple(selected)


def load_validation_protocol(path: Path, *, expected_sha256: str) -> dict[str, object]:
    """Load, verify, and lock the frozen validation protocol used by feature artifacts."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validation protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    if not verify_protocol_sha256(payload):
        raise ValueError("validation protocol embedded SHA-256 is invalid")
    actual = payload.get("protocol_sha256")
    if actual != expected_sha256:
        raise ValueError(
            f"validation protocol SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )
    return payload


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _dataset_for_record(record: RawManifestRecord, store: S3RawStore) -> Any:
    return ds.dataset(
        f"{store.bucket}/{record.s3_key}",
        filesystem=store.filesystem,
        format="parquet",
    )


def _field_pairs(schema: Any) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.type)) for field in schema)


def _is_numeric_arrow_type(type_name: str) -> bool:
    normalized = type_name.lower().replace(" ", "")
    return normalized.startswith(("int", "uint", "float", "double", "halffloat", "decimal"))


def _is_string_arrow_type(type_name: str) -> bool:
    normalized = type_name.lower().replace(" ", "")
    return "string" in normalized or "dictionary<" in normalized


def _is_temporal_arrow_type(type_name: str) -> bool:
    normalized = type_name.lower().replace(" ", "")
    return normalized.startswith(("date", "timestamp", "time"))


def _date_days_expression(name: str, type_name: str) -> pl.Expr:
    """Normalize one Home Credit date predictor to days relative to decision date."""
    if _is_temporal_arrow_type(type_name):
        parsed = pl.col(name).cast(pl.Date, strict=False)
        return (parsed - pl.col("_decision_date")).dt.total_days().cast(pl.Float32).alias(name)

    if _is_string_arrow_type(type_name) or type_name.lower().replace(" ", "") == "null":
        parsed = pl.col(name).cast(pl.String, strict=False).str.to_date(strict=False)
        return (parsed - pl.col("_decision_date")).dt.total_days().cast(pl.Float32).alias(name)

    if _is_numeric_arrow_type(type_name):
        # In the official tiny test sample some entirely-null D columns are encoded
        # as floating point. If a future source carries a numeric D representation,
        # preserve it as an already-relative numeric feature rather than guessing a date.
        return pl.col(name).cast(pl.Float32, strict=False).alias(name)

    return pl.lit(None, dtype=pl.Float32).alias(name)


def _normalized_scan(
    source: LogicalSource,
    *,
    store: S3RawStore,
    decision_frame: pl.LazyFrame | None,
    recipe: FeatureRecipe,
) -> tuple[pl.LazyFrame, SemanticColumns]:
    """Scan shards separately, normalize semantic dtypes, then concatenate lazily."""
    frames: list[pl.LazyFrame] = []
    all_numeric: set[str] = set()
    all_categorical: set[str] = set()
    all_dates: set[str] = set()
    all_unsupported: set[str] = set()

    for record in source.records:
        dataset = _dataset_for_record(record, store)
        fields = _field_pairs(dataset.schema)
        semantic = classify_columns(fields)
        all_numeric.update(semantic.numeric)
        all_categorical.update(semantic.categorical)
        all_dates.update(semantic.date)
        all_unsupported.update(semantic.unsupported)

        names = {name for name, _ in fields}
        assert_no_target_leakage(
            tuple(sorted(names)) if source.family != "base" else (),
            context=source.logical_name,
        )

        frame = pl.scan_pyarrow_dataset(
            dataset,
            allow_pyarrow_filter=True,
            batch_size=recipe.scan_batch_rows,
        )

        if CASE_ID not in names:
            raise ValueError(f"raw source is missing case_id: {record.file}")

        select_columns = [CASE_ID]
        for structural in (NUM_GROUP1, NUM_GROUP2):
            if structural in names:
                select_columns.append(structural)
        select_columns.extend(semantic.predictors)

        frame = frame.select(select_columns)

        if decision_frame is not None and semantic.date:
            frame = frame.join(decision_frame, on=CASE_ID, how="left", validate="m:1")

        type_by_name = dict(fields)
        normalizers: list[pl.Expr] = [pl.col(CASE_ID).cast(pl.Int64, strict=False)]

        for structural in (NUM_GROUP1, NUM_GROUP2):
            if structural in select_columns:
                normalizers.append(pl.col(structural).cast(pl.Int64, strict=False))

        normalizers.extend(
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in semantic.numeric
        )
        normalizers.extend(
            pl.col(column).cast(pl.String, strict=False).alias(column)
            for column in semantic.categorical
        )
        normalizers.extend(
            _date_days_expression(column, type_by_name[column]) for column in semantic.date
        )

        frame = frame.select(normalizers)
        frames.append(frame)

    if not frames:
        raise ValueError(f"logical source has no shards: {source.logical_name}")

    combined = pl.concat(frames, how="diagonal_relaxed", rechunk=False)
    semantic = SemanticColumns(
        numeric=tuple(sorted(all_numeric - all_dates)),
        categorical=tuple(sorted(all_categorical - all_dates)),
        date=tuple(sorted(all_dates)),
        unsupported=tuple(sorted(all_unsupported)),
    )
    return combined, semantic


def _base_source(sources: Sequence[LogicalSource], split: Split) -> LogicalSource:
    matches = [source for source in sources if source.split == split and source.family == "base"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {split} base source, found {len(matches)}")
    return matches[0]


def _scan_base(source: LogicalSource, store: S3RawStore, recipe: FeatureRecipe) -> pl.LazyFrame:
    if len(source.records) != 1:
        raise ValueError(f"base source must contain one file: {source.logical_name}")
    dataset = _dataset_for_record(source.records[0], store)
    return pl.scan_pyarrow_dataset(
        dataset,
        allow_pyarrow_filter=True,
        batch_size=recipe.scan_batch_rows,
    )


def _decision_frame(base: pl.LazyFrame) -> pl.LazyFrame:
    schema_names = base.collect_schema().names()
    required = {CASE_ID, DATE_DECISION}
    missing = sorted(required - set(schema_names))
    if missing:
        raise ValueError(f"base table is missing decision columns: {missing}")
    return base.select(
        pl.col(CASE_ID).cast(pl.Int64, strict=False),
        pl.col(DATE_DECISION).cast(pl.Date, strict=False).alias("_decision_date"),
    )


def _base_block(base: pl.LazyFrame, *, split: Split) -> pl.LazyFrame:
    schema_names = set(base.collect_schema().names())
    required = {CASE_ID, DATE_DECISION, WEEK_NUM, MONTH}
    if split == "train":
        required.add(TARGET)
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"base table is missing required columns: {missing}")
    if split == "test" and TARGET in schema_names:
        raise ValueError("test base unexpectedly contains target")

    decision = pl.col(DATE_DECISION).cast(pl.Date, strict=False)
    expressions: list[pl.Expr] = [
        pl.col(CASE_ID).cast(pl.Int64, strict=False),
        pl.col(WEEK_NUM).cast(pl.Int32, strict=False),
        pl.col(MONTH).cast(pl.Int16, strict=False),
        decision.dt.year().cast(pl.Int16).alias("base__decision_year"),
        decision.dt.month().cast(pl.Int8).alias("base__decision_month"),
        decision.dt.day().cast(pl.Int8).alias("base__decision_day"),
        decision.dt.weekday().cast(pl.Int8).alias("base__decision_weekday"),
        decision.dt.ordinal_day().cast(pl.Int16).alias("base__decision_ordinal_day"),
    ]
    if split == "train":
        expressions.append(pl.col(TARGET).cast(pl.Int8, strict=False))
    return base.select(expressions)


def _depth_zero_block(
    frame: pl.LazyFrame,
    semantic: SemanticColumns,
    *,
    source: LogicalSource,
) -> pl.LazyFrame:
    prefix = feature_prefix(source.family, source.depth)
    predictors = semantic.predictors
    expressions: list[pl.Expr] = [pl.col(CASE_ID)]
    expressions.extend(pl.col(column).alias(f"{prefix}__{column}") for column in predictors)

    if predictors:
        missing_count = pl.sum_horizontal(
            [pl.col(column).is_null().cast(pl.UInt16) for column in predictors]
        )
        expressions.extend(
            [
                missing_count.alias(f"{prefix}__missing_count"),
                (missing_count.cast(pl.Float32) / float(len(predictors))).alias(
                    f"{prefix}__missing_fraction"
                ),
            ]
        )

    return frame.select(expressions).unique(subset=[CASE_ID], keep="first", maintain_order=True)


def _history_block(
    frame: pl.LazyFrame,
    semantic: SemanticColumns,
    *,
    source: LogicalSource,
    recipe: FeatureRecipe,
) -> pl.LazyFrame:
    if source.family == "person":
        subgroup_blocks: list[pl.LazyFrame] = []
        allowed_subgroups = set(recipe.person_subgroups)
        for subgroup, subgroup_frame in build_person_subgroups(frame):
            if subgroup not in allowed_subgroups:
                continue
            subgroup_blocks.append(
                aggregate_case_history(
                    subgroup_frame,
                    family=source.family,
                    depth=source.depth,
                    numeric_columns=semantic.numeric,
                    categorical_columns=semantic.categorical,
                    date_columns=semantic.date,
                    time_windows_days=recipe.time_windows_days,
                    subgroup=subgroup,
                )
            )
        if not subgroup_blocks:
            raise ValueError("person feature recipe selected no subgroups")
        result = subgroup_blocks[0]
        for block in subgroup_blocks[1:]:
            result = result.join(block, on=CASE_ID, how="left", validate="1:1")
        return result

    return aggregate_case_history(
        frame,
        family=source.family,
        depth=source.depth,
        numeric_columns=semantic.numeric,
        categorical_columns=semantic.categorical,
        date_columns=semantic.date,
        time_windows_days=recipe.time_windows_days,
    )


def _materialize(frame: pl.LazyFrame, output: Path, *, recipe: FeatureRecipe) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.sink_parquet(
            temporary,
            compression=cast(Any, recipe.compression),
            statistics=True,
            maintain_order=True,
            mkdir=True,
            engine="streaming",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _block_contract(output: Path) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(output)
    rows = int(parquet_file.metadata.num_rows)
    columns = int(parquet_file.metadata.num_columns)
    names = parquet_file.schema_arrow.names
    if not names or names[0] != CASE_ID:
        raise ValueError(f"feature block must start with case_id: {output}")
    if TARGET in names and "base" not in output.name:
        raise ValueError(f"target leaked into non-base feature block: {output}")

    cardinality = (
        pl.scan_parquet(output)
        .select(
            pl.len().alias("rows"),
            pl.col(CASE_ID).n_unique().alias("unique_case_ids"),
            pl.col(CASE_ID).null_count().alias("null_case_ids"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if int(cardinality["null_case_ids"]) != 0:
        raise ValueError(f"feature block contains null case_id: {output}")
    if int(cardinality["rows"]) != int(cardinality["unique_case_ids"]):
        raise ValueError(f"feature block contains duplicate case_id: {output}")
    return rows, columns


def _source_feature_block(
    source: LogicalSource,
    *,
    all_sources: Sequence[LogicalSource],
    store: S3RawStore,
    recipe: FeatureRecipe,
    recipe_sha256: str,
    protocol_sha256: str,
    output_dir: Path,
    logger: RunLogger,
    heartbeat_seconds: float,
) -> FeatureBlock:
    base_source = _base_source(all_sources, source.split)
    base_frame = _scan_base(base_source, store, recipe)

    if source.family == "base":
        semantic = SemanticColumns((), (), (), ())
        block = _base_block(base_frame, split=source.split)
    else:
        decision_frame = _decision_frame(base_frame)
        frame, semantic = _normalized_scan(
            source,
            store=store,
            decision_frame=decision_frame,
            recipe=recipe,
        )
        block = (
            _depth_zero_block(frame, semantic, source=source)
            if source.depth == 0
            else _history_block(frame, semantic, source=source, recipe=recipe)
        )

    output = output_dir / "blocks" / source.split / f"{source.family}_depth{source.depth}.parquet"
    with StageTimer(
        logger,
        f"materialize_{source.logical_name}",
        heartbeat_seconds=heartbeat_seconds,
    ):
        _materialize(block, output, recipe=recipe)

    with StageTimer(logger, f"validate_{source.logical_name}"):
        rows, columns = _block_contract(output)
        digest = sha256_file(output)

    result = FeatureBlock(
        split=source.split,
        family=source.family,
        depth=source.depth,
        output=output.as_posix(),
        source_files=tuple(record.file for record in source.records),
        source_sha256=tuple(record.sha256 for record in source.records),
        rows=rows,
        columns=columns,
        feature_columns=max(columns - 1, 0),
        output_bytes=output.stat().st_size,
        output_sha256=digest,
        recipe_sha256=recipe_sha256,
        protocol_sha256=protocol_sha256,
        unsupported_columns=semantic.unsupported,
    )
    logger.event(
        "feature_block_completed",
        split=result.split,
        family=result.family,
        depth=result.depth,
        rows=result.rows,
        columns=result.columns,
        feature_columns=result.feature_columns,
        output_bytes=result.output_bytes,
        output=result.output,
    )
    return result


def build_feature_blocks(
    records: Sequence[RawManifestRecord],
    store: S3RawStore,
    *,
    recipe: FeatureRecipe,
    recipe_sha256: str,
    protocol_sha256: str,
    output_dir: Path,
    logger: RunLogger,
    heartbeat_seconds: float,
    splits: frozenset[str],
    families: frozenset[str] | None,
) -> tuple[FeatureBlock, ...]:
    """Build selected case-level feature blocks sequentially to bound peak memory."""
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")

    all_sources = group_logical_sources(records)
    selected = select_sources(all_sources, splits=splits, families=families)
    if not selected:
        raise ValueError("feature source selection is empty")

    blocks: list[FeatureBlock] = []
    started = time.monotonic()
    logger.event(
        "feature_build_started",
        logical_sources=len(selected),
        splits=sorted(splits),
        families="all" if families is None else sorted(families),
        recipe=recipe.name,
    )

    for index, source in enumerate(selected, start=1):
        logger.event(
            "feature_block_started",
            index=index,
            total=len(selected),
            split=source.split,
            family=source.family,
            depth=source.depth,
            shards=len(source.records),
        )
        blocks.append(
            _source_feature_block(
                source,
                all_sources=all_sources,
                store=store,
                recipe=recipe,
                recipe_sha256=recipe_sha256,
                protocol_sha256=protocol_sha256,
                output_dir=output_dir,
                logger=logger,
                heartbeat_seconds=heartbeat_seconds,
            )
        )

    logger.event(
        "feature_build_completed",
        blocks=len(blocks),
        total_feature_columns=sum(block.feature_columns for block in blocks),
        total_output_bytes=sum(block.output_bytes for block in blocks),
        total_elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return tuple(blocks)


def write_feature_manifest(
    blocks: Sequence[FeatureBlock],
    *,
    output: Path,
    manifest_uri: str,
    manifest_sha256: str,
    protocol_sha256: str,
    recipe_sha256: str,
) -> None:
    """Atomically write the feature-block provenance manifest."""
    payload = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "raw_manifest_uri": manifest_uri,
        "raw_manifest_sha256": manifest_sha256,
        "validation_protocol_sha256": protocol_sha256,
        "feature_recipe_sha256": recipe_sha256,
        "blocks": [asdict(block) for block in blocks],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def block_totals(blocks: Iterable[FeatureBlock]) -> dict[str, int]:
    """Return deterministic aggregate block counts for logging and acceptance checks."""
    materialized = tuple(blocks)
    return {
        "blocks": len(materialized),
        "feature_columns": sum(block.feature_columns for block in materialized),
        "output_bytes": sum(block.output_bytes for block in materialized),
    }
