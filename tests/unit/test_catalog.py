from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from home_credit.data.catalog import inspect_parquet, summarize_tables
from home_credit.data.loader import RawManifestRecord


def _write_parquet(path: Path, case_ids: list[int], *, include_target: bool = False) -> None:
    payload: dict[str, pa.Array] = {
        "case_id": pa.array(case_ids, type=pa.int64()),
        "WEEK_NUM": pa.array([1] * len(case_ids), type=pa.int64()),
        "valueA": pa.array([1.0] * len(case_ids), type=pa.float64()),
        "categoryM": pa.array(["a"] * len(case_ids), type=pa.string()),
    }
    if include_target:
        payload["target"] = pa.array([0] * len(case_ids), type=pa.int64())
    pq.write_table(pa.table(payload), path, row_group_size=2)


def _inspect(path: Path, file_name: str) -> object:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = RawManifestRecord(
        file=file_name,
        s3_key=f"raw/{path.name}",
        bytes=path.stat().st_size,
        sha256=digest,
    )
    return inspect_parquet(
        record,
        open_input_file=lambda: path.open("rb"),
        current_size=path.stat().st_size,
    )


def test_catalog_exact_case_cardinality_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "train_applprev_1.parquet"
    _write_parquet(path, [1, 1, 2, 3, 3, 3])
    entry = _inspect(path, "parquet_files/train/train_applprev_1.parquet")
    assert entry.rows == 6
    assert entry.unique_case_ids == 3
    assert entry.max_rows_per_case == 3
    assert entry.case_id_null_count == 0
    assert entry.numeric_columns >= 2
    assert entry.categorical_columns == 1
    assert len(entry.schema_sha256) == 64


def test_table_summary_detects_consistent_shards(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_parquet(first, [1, 2])
    _write_parquet(second, [3, 4])
    entries = [
        _inspect(first, "parquet_files/train/train_static_0_0.parquet"),
        _inspect(second, "parquet_files/train/train_static_0_1.parquet"),
    ]
    summaries = summarize_tables(entries)
    assert len(summaries) == 1
    assert summaries[0].files == 2
    assert summaries[0].rows == 4
    assert summaries[0].unique_schema_count == 1
