from __future__ import annotations

import pytest

from home_credit.data.schema import parse_table_identity, schema_fingerprint


def test_parse_base_identity() -> None:
    identity = parse_table_identity("parquet_files/train/train_base.parquet")
    assert identity.split == "train"
    assert identity.family == "base"
    assert identity.depth == 0
    assert identity.shard is None


def test_parse_sharded_depth_zero_identity() -> None:
    identity = parse_table_identity("parquet_files/test/test_static_0_2.parquet")
    assert identity.split == "test"
    assert identity.family == "static"
    assert identity.depth == 0
    assert identity.shard == 2


def test_parse_sharded_depth_one_identity() -> None:
    identity = parse_table_identity("parquet_files/train/train_applprev_1_0.parquet")
    assert identity.split == "train"
    assert identity.family == "applprev"
    assert identity.depth == 1
    assert identity.shard == 0


def test_parse_sharded_depth_two_identity() -> None:
    identity = parse_table_identity("parquet_files/train/train_credit_bureau_a_2_10.parquet")
    assert identity.split == "train"
    assert identity.family == "credit_bureau_a"
    assert identity.depth == 2
    assert identity.shard == 10


def test_parse_unsharded_identity() -> None:
    identity = parse_table_identity("parquet_files/train/train_tax_registry_a_1.parquet")
    assert identity.family == "tax_registry_a"
    assert identity.depth == 1
    assert identity.shard is None


def test_schema_fingerprint_is_deterministic_and_order_sensitive() -> None:
    first = [("case_id", "int64", True), ("value", "double", True)]
    second = list(first)
    reversed_fields = list(reversed(first))
    assert schema_fingerprint(first) == schema_fingerprint(second)
    assert schema_fingerprint(first) != schema_fingerprint(reversed_fields)


def test_invalid_filename_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_table_identity("sample_submission.csv")
