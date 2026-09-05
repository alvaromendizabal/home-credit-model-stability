from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from home_credit.features.aggregation import aggregate_case_history
from home_credit.features.execution import (
    BuildIdentity,
    BuildStateStore,
    CasePartition,
    FeatureExecutionPolicy,
    PartitionReceipt,
    plan_case_partitions,
    should_partition_source,
)


def _identity(*, git_commit: str = "abc123") -> BuildIdentity:
    return BuildIdentity(
        git_commit=git_commit,
        raw_manifest_sha256="1" * 64,
        validation_protocol_sha256="2" * 64,
        feature_recipe_sha256="3" * 64,
        feature_execution_sha256="4" * 64,
        selected_sources=("train_base_depth0", "train_demo_depth1"),
    )


def test_execution_policy_is_fingerprinted(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "case_range_partitioned",
        "partition_rows": 20_000,
        "partition_threshold_rows": 100_000,
        "partition_min_source_bytes": 134_217_728,
        "max_threads": 4,
        "resume": True,
        "validate_intermediate_hashes": True,
        "sort_partitions_by_case_id": True,
        "retain_partition_files": False,
        "work_directory_name": "work",
        "notes": "test",
    }
    path = tmp_path / "feature_execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    policy, digest = FeatureExecutionPolicy.load(path)

    assert policy.partition_rows == 20_000
    assert policy.max_threads == 4
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_policy_rejects_non_resumable_mode(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "case_range_partitioned",
        "partition_rows": 20_000,
        "partition_threshold_rows": 100_000,
        "partition_min_source_bytes": 134_217_728,
        "max_threads": 4,
        "resume": False,
        "validate_intermediate_hashes": True,
        "sort_partitions_by_case_id": True,
        "retain_partition_files": False,
        "work_directory_name": "work",
        "notes": "test",
    }
    path = tmp_path / "feature_execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="resumable"):
        FeatureExecutionPolicy.load(path)


def test_case_partition_plan_is_exact_and_disjoint() -> None:
    partitions = plan_case_partitions(
        (10, 20, 30, 40, 50, 60, 70),
        partition_rows=3,
    )

    assert partitions == (
        CasePartition(0, 10, 30, 3),
        CasePartition(1, 40, 60, 3),
        CasePartition(2, 70, 70, 1),
    )
    assert partitions[0].case_id_max < partitions[1].case_id_min
    assert partitions[1].case_id_max < partitions[2].case_id_min


def test_case_partition_plan_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        plan_case_partitions((1, 2, 2, 3), partition_rows=2)


def test_partitioned_history_equals_single_pass_history() -> None:
    frame = pl.DataFrame(
        {
            "case_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "num_group1": [0, 1, 0, 1, 0, 1, 0, 1],
            "amount_1A": [1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 6.0, 9.0],
            "status_1M": ["a", "b", "a", "a", "c", "c", "b", "d"],
            "event_1D": [-10.0, -5.0, -20.0, -2.0, -8.0, -1.0, -4.0, -3.0],
        }
    ).lazy()

    def aggregate(lazy: pl.LazyFrame) -> pl.DataFrame:
        return (
            aggregate_case_history(
                lazy,
                family="demo",
                depth=1,
                numeric_columns=("amount_1A",),
                categorical_columns=("status_1M",),
                date_columns=("event_1D",),
                time_windows_days=(30, 180),
            )
            .sort("case_id")
            .collect()
        )

    expected = aggregate(frame)
    partitions = plan_case_partitions((1, 2, 3, 4), partition_rows=2)
    actual = pl.concat(
        [
            aggregate(
                frame.filter(
                    pl.col("case_id").is_between(
                        partition.case_id_min,
                        partition.case_id_max,
                        closed="both",
                    )
                )
            )
            for partition in partitions
        ],
        how="vertical",
    ).sort("case_id")

    assert actual.schema == expected.schema
    assert actual.to_dicts() == expected.to_dicts()


def test_build_state_rejects_incompatible_identity(tmp_path: Path) -> None:
    path = tmp_path / "build_state.json"
    BuildStateStore(path, _identity())

    with pytest.raises(ValueError, match="identity"):
        BuildStateStore(path, _identity(git_commit="different"))


def test_partition_checkpoint_requires_verified_file(tmp_path: Path) -> None:
    output_root = tmp_path / "features"
    output_root.mkdir()
    state = BuildStateStore(output_root / "build_state.json", _identity())
    partition = CasePartition(0, 1, 100, 100)
    part_path = output_root / "work" / "train_demo_depth1" / "part-00000.parquet"
    part_path.parent.mkdir(parents=True)
    part_path.write_bytes(b"deterministic-partition")
    digest = hashlib.sha256(part_path.read_bytes()).hexdigest()

    receipt = PartitionReceipt(
        index=0,
        case_id_min=1,
        case_id_max=100,
        expected_base_cases=100,
        rows=80,
        columns=12,
        output=part_path.relative_to(output_root).as_posix(),
        output_bytes=part_path.stat().st_size,
        output_sha256=digest,
    )
    state.record_partition("train_demo_depth1", receipt)

    restored = state.partition_receipt(
        "train_demo_depth1",
        partition,
        output_root=output_root,
        validate_hash=True,
    )
    assert restored == receipt

    part_path.write_bytes(b"corrupted")
    assert (
        state.partition_receipt(
            "train_demo_depth1",
            partition,
            output_root=output_root,
            validate_hash=True,
        )
        is None
    )


def test_partition_decision_uses_population_and_source_size(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "case_range_partitioned",
        "partition_rows": 20_000,
        "partition_threshold_rows": 100_000,
        "partition_min_source_bytes": 134_217_728,
        "max_threads": 4,
        "resume": True,
        "validate_intermediate_hashes": True,
        "sort_partitions_by_case_id": True,
        "retain_partition_files": False,
        "work_directory_name": "work",
        "notes": "test",
    }
    path = tmp_path / "feature_execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    policy, _ = FeatureExecutionPolicy.load(path)

    assert should_partition_source(
        base_rows=1_500_000,
        source_bytes=500_000_000,
        is_base=False,
        policy=policy,
    )
    assert not should_partition_source(
        base_rows=1_500_000,
        source_bytes=10_000_000,
        is_base=False,
        policy=policy,
    )
    assert not should_partition_source(
        base_rows=10,
        source_bytes=500_000_000,
        is_base=False,
        policy=policy,
    )
    assert not should_partition_source(
        base_rows=1_500_000,
        source_bytes=500_000_000,
        is_base=True,
        policy=policy,
    )
