from __future__ import annotations

import polars as pl

from home_credit.features.aggregation import (
    aggregate_case_history,
    build_person_subgroups,
    history_order_columns,
)


def test_history_order_columns_respect_depth() -> None:
    assert history_order_columns(1, ["case_id", "num_group1"]) == ("num_group1",)
    assert history_order_columns(
        2,
        ["case_id", "num_group1", "num_group2"],
    ) == ("num_group1", "num_group2")
    assert history_order_columns(1, ["case_id"]) == ()


def test_case_history_aggregates_numeric_categorical_date_and_missingness() -> None:
    frame = pl.DataFrame(
        {
            "case_id": [1, 1, 1, 2],
            "num_group1": [2, 0, 1, 0],
            "amount_1A": [30.0, 10.0, None, 5.0],
            "category_1M": ["b", "a", "a", "z"],
            "event_1D": [-10.0, -400.0, -40.0, -5.0],
        }
    ).lazy()

    result = aggregate_case_history(
        frame,
        family="demo",
        depth=1,
        numeric_columns=("amount_1A",),
        categorical_columns=("category_1M",),
        date_columns=("event_1D",),
        time_windows_days=(30, 365),
    ).collect()

    assert result.height == 2
    assert result["case_id"].to_list() == [1, 2]

    case_one = result.filter(pl.col("case_id") == 1).row(0, named=True)
    assert case_one["demo__d1__rows__count"] == 3
    assert case_one["demo__d1__amount_1A__min"] == 10.0
    assert case_one["demo__d1__amount_1A__max"] == 30.0
    assert case_one["demo__d1__amount_1A__first"] == 10.0
    assert case_one["demo__d1__amount_1A__last"] == 30.0
    assert case_one["demo__d1__category_1M__n_unique"] == 2
    assert case_one["demo__d1__event_1D__count_last_30d"] == 1
    assert case_one["demo__d1__event_1D__count_last_365d"] == 2


def test_person_subgroups_keep_applicant_and_related_history_separate() -> None:
    frame = pl.DataFrame(
        {
            "case_id": [1, 1, 2],
            "num_group1": [0, 2, 0],
            "value": [10, 20, 30],
        }
    ).lazy()

    groups = dict(build_person_subgroups(frame))
    assert set(groups) == {"all", "applicant", "related"}
    assert groups["applicant"].collect()["value"].to_list() == [10, 30]
    assert groups["related"].collect()["value"].to_list() == [20]
