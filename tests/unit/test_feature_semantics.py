from __future__ import annotations

import pytest

from home_credit.features.semantics import (
    SemanticColumns,
    assert_no_target_leakage,
    classify_columns,
    classify_feature,
    feature_name,
    is_date_feature,
)


def test_date_suffix_overrides_physical_string_type() -> None:
    assert is_date_feature("validfrom_1069D")
    assert classify_feature("validfrom_1069D", "string") == "date"
    assert classify_feature("validfrom_1069D", "double") == "date"


def test_null_dtype_recovers_unambiguous_suffix_semantics() -> None:
    assert classify_feature("category_1M", "null") == "categorical"
    assert classify_feature("amount_2A", "null") == "numeric"
    assert classify_feature("count_3L", "null") == "numeric"
    assert classify_feature("mystery", "null") == "unsupported"


def test_column_classification_is_deterministic() -> None:
    actual = classify_columns(
        (
            ("case_id", "int64"),
            ("amount_1A", "double"),
            ("category_2M", "string"),
            ("event_3D", "date32[day]"),
            ("blob", "binary"),
        )
    )

    assert actual == SemanticColumns(
        numeric=("amount_1A",),
        categorical=("category_2M",),
        date=("event_3D",),
        unsupported=("blob",),
    )
    assert actual.predictors == ("amount_1A", "category_2M", "event_3D")


def test_target_leakage_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_target_leakage(
            ("case_id", "amount_1A", "target"),
            context="train_static_depth0",
        )


def test_feature_names_are_collision_resistant() -> None:
    assert (
        feature_name("credit_bureau_a", 2, "amount_1A", "mean")
        == "credit_bureau_a__d2__amount_1A__mean"
    )
    assert (
        feature_name(
            "person",
            1,
            "income_1A",
            "max",
            subgroup="applicant",
        )
        == "person__d1__applicant__income_1A__max"
    )


def test_semantic_resolution_harmonizes_mixed_shard_types() -> None:
    from home_credit.features.semantics import (
        resolve_semantic_columns,
    )

    resolved = resolve_semantic_columns(
        (
            SemanticColumns(
                numeric=(
                    "amount_1A",
                    "status_1L",
                ),
                categorical=("category_1M",),
                date=("event_1D",),
                unsupported=("blob",),
            ),
            SemanticColumns(
                numeric=(),
                categorical=(
                    "amount_1A",
                    "status_1L",
                ),
                date=("event_1D",),
                unsupported=("blob",),
            ),
        )
    )

    assert resolved.numeric == ("amount_1A",)

    assert resolved.categorical == (
        "category_1M",
        "status_1L",
    )

    assert resolved.date == ("event_1D",)

    assert resolved.unsupported == ("blob",)
