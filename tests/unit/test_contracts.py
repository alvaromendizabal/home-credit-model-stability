from __future__ import annotations

from dataclasses import replace

from home_credit.data.catalog import CatalogEntry
from home_credit.data.contracts import validate_catalog


def _entry(
    *,
    split: str,
    family: str,
    columns: tuple[str, ...],
    types: tuple[str, ...] | None = None,
    depth: int = 0,
    shard: int | None = None,
    rows: int = 10,
    max_rows: int = 1,
    schema_sha256: str = "b" * 64,
    file: str | None = None,
) -> CatalogEntry:
    column_types = types if types is not None else tuple("int64" for _ in columns)

    filename = (
        file if file is not None else (f"parquet_files/{split}/{split}_{family}_{depth}.parquet")
    )

    return CatalogEntry(
        file=filename,
        s3_key=("raw/" + filename.rsplit("/", maxsplit=1)[-1]),
        source_bytes=100,
        source_sha256="a" * 64,
        split=split,
        family=family,
        depth=depth,
        shard=shard,
        rows=rows,
        columns=len(columns),
        row_groups=1,
        schema_sha256=schema_sha256,
        column_names=columns,
        column_types=column_types,
        numeric_columns=0,
        categorical_columns=0,
        temporal_columns=0,
        case_id_present="case_id" in columns,
        case_id_null_count=0,
        unique_case_ids=rows if rows > 0 else 0,
        max_rows_per_case=(max_rows if rows > 0 else 0),
        null_statistics_columns=len(columns),
        max_missing_fraction=(0.0 if rows > 0 else None),
    )


def _base_pair() -> tuple[
    CatalogEntry,
    CatalogEntry,
]:
    return (
        _entry(
            split="train",
            family="base",
            columns=(
                "case_id",
                "WEEK_NUM",
                "target",
            ),
        ),
        _entry(
            split="test",
            family="base",
            columns=(
                "case_id",
                "WEEK_NUM",
            ),
        ),
    )


def test_contracts_accept_valid_base_pair() -> None:
    report = validate_catalog(list(_base_pair()))

    assert report.passed
    assert report.errors == ()


def test_contracts_reject_depth_zero_multiplicity() -> None:
    train, test = _base_pair()

    bad = replace(
        train,
        family="static",
        max_rows_per_case=2,
        file="train_static_0.parquet",
    )

    report = validate_catalog([train, test, bad])

    assert not report.passed
    assert any("depth-0" in error for error in report.errors)


def test_contracts_allow_known_empty_test_table() -> None:
    train, test = _base_pair()

    train_tax = _entry(
        split="train",
        family="tax_registry_c",
        depth=1,
        columns=("case_id", "amount"),
        max_rows=4,
    )

    test_tax = _entry(
        split="test",
        family="tax_registry_c",
        depth=1,
        columns=("case_id", "amount"),
        rows=0,
        max_rows=0,
        file=("parquet_files/test/test_tax_registry_c_1.parquet"),
    )

    report = validate_catalog(
        [
            train,
            test,
            train_tax,
            test_tax,
        ]
    )

    assert report.passed

    assert any(
        "known empty official test Parquet accepted" in warning for warning in report.warnings
    )


def test_contracts_reject_unexpected_empty_test_table() -> None:
    train, test = _base_pair()

    unexpected = _entry(
        split="test",
        family="person",
        depth=1,
        columns=("case_id", "value"),
        rows=0,
        max_rows=0,
    )

    report = validate_catalog(
        [
            train,
            test,
            unexpected,
        ]
    )

    assert not report.passed

    assert any("unexpected empty test" in error for error in report.errors)


def test_contracts_allow_numeric_width_drift() -> None:
    train, test = _base_pair()

    train_static = _entry(
        split="train",
        family="static",
        columns=("case_id", "amount"),
        types=("int64", "double"),
    )

    test_static = _entry(
        split="test",
        family="static",
        columns=("case_id", "amount"),
        types=("int32", "float"),
    )

    report = validate_catalog(
        [
            train,
            test,
            train_static,
            test_static,
        ]
    )

    assert report.passed

    assert any("physical type drift" in warning for warning in report.warnings)


def test_contracts_allow_null_typed_test_column() -> None:
    train, test = _base_pair()

    train_person = _entry(
        split="train",
        family="person",
        depth=1,
        columns=("case_id", "occupation"),
        types=("int64", "string"),
        max_rows=2,
    )

    test_person = _entry(
        split="test",
        family="person",
        depth=1,
        columns=("case_id", "occupation"),
        types=("int64", "null"),
        max_rows=2,
    )

    report = validate_catalog(
        [
            train,
            test,
            train_person,
            test_person,
        ]
    )

    assert report.passed


def test_contracts_allow_compatible_shard_variants() -> None:
    train, test = _base_pair()

    train_applprev = _entry(
        split="train",
        family="applprev",
        depth=1,
        columns=("case_id", "amount"),
        types=("int64", "double"),
        max_rows=3,
    )

    shard_a = _entry(
        split="test",
        family="applprev",
        depth=1,
        shard=0,
        columns=("case_id", "amount"),
        types=("int64", "double"),
        schema_sha256="c" * 64,
        file=("parquet_files/test/test_applprev_1_0.parquet"),
        max_rows=3,
    )

    shard_b = _entry(
        split="test",
        family="applprev",
        depth=1,
        shard=1,
        columns=("case_id", "amount"),
        types=("int64", "float"),
        schema_sha256="d" * 64,
        file=("parquet_files/test/test_applprev_1_1.parquet"),
        max_rows=3,
    )

    report = validate_catalog(
        [
            train,
            test,
            train_applprev,
            shard_a,
            shard_b,
        ]
    )

    assert report.passed

    assert any("physical schema variants observed" in warning for warning in report.warnings)


def test_contracts_reject_column_identity_mismatch() -> None:
    train, test = _base_pair()

    train_person = _entry(
        split="train",
        family="person",
        depth=1,
        columns=("case_id", "occupation"),
        max_rows=2,
    )

    test_person = _entry(
        split="test",
        family="person",
        depth=1,
        columns=("case_id", "employer"),
        max_rows=2,
    )

    report = validate_catalog(
        [
            train,
            test,
            train_person,
            test_person,
        ]
    )

    assert not report.passed

    assert any("train/test logical schema mismatch" in error for error in report.errors)


def test_contracts_reject_incompatible_type_families() -> None:
    train, test = _base_pair()

    shard_a = _entry(
        split="train",
        family="person",
        depth=1,
        shard=0,
        columns=("case_id", "value"),
        types=("int64", "string"),
        file=("parquet_files/train/train_person_1_0.parquet"),
        max_rows=2,
    )

    shard_b = _entry(
        split="train",
        family="person",
        depth=1,
        shard=1,
        columns=("case_id", "value"),
        types=("int64", "timestamp[us]"),
        file=("parquet_files/train/train_person_1_1.parquet"),
        max_rows=2,
    )

    report = validate_catalog(
        [
            train,
            test,
            shard_a,
            shard_b,
        ]
    )

    assert not report.passed

    assert any("logically schema-incompatible" in error for error in report.errors)


def test_contracts_accept_home_credit_date_semantic_drift() -> None:
    train, test = _base_pair()

    train_static = _entry(
        split="train",
        family="static",
        columns=("case_id", "validfrom_1069D"),
        types=("int64", "string"),
    )

    test_static = _entry(
        split="test",
        family="static",
        columns=("case_id", "validfrom_1069D"),
        types=("int64", "double"),
    )

    report = validate_catalog([train, test, train_static, test_static])

    assert report.passed
    assert report.errors == ()


def test_contracts_reject_same_drift_for_non_date_feature() -> None:
    train, test = _base_pair()

    train_static = _entry(
        split="train",
        family="static",
        columns=("case_id", "ordinary_feature"),
        types=("int64", "string"),
    )

    test_static = _entry(
        split="test",
        family="static",
        columns=("case_id", "ordinary_feature"),
        types=("int64", "double"),
    )

    report = validate_catalog([train, test, train_static, test_static])

    assert not report.passed
    assert any("schema" in error.lower() for error in report.errors)
