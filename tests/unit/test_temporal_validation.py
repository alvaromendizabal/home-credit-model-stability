from __future__ import annotations

import pandas as pd
import pytest

from home_credit.validation.temporal import (
    build_holdout_candidates,
    build_week_profile,
    prepare_base_frame,
    temporal_integrity,
)


def _frame(*, weeks: int = 40, rows_per_week: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    case_id = 1
    start = pd.Timestamp("2020-01-06")
    for week in range(weeks):
        for offset in range(rows_per_week):
            rows.append(
                {
                    "case_id": case_id,
                    "date_decision": start + pd.Timedelta(days=7 * week + offset),
                    "WEEK_NUM": week,
                    "target": offset % 2,
                }
            )
            case_id += 1
    return pd.DataFrame(rows)


def test_prepare_base_frame_accepts_valid_binary_case_data() -> None:
    prepared = prepare_base_frame(_frame(), require_target=True)
    assert len(prepared) == 160
    assert prepared["case_id"].is_unique
    assert set(prepared["target"].unique()) == {0, 1}


def test_prepare_base_frame_rejects_duplicate_case_id() -> None:
    frame = _frame()
    frame.loc[1, "case_id"] = frame.loc[0, "case_id"]
    with pytest.raises(ValueError, match="case_id values must be unique"):
        prepare_base_frame(frame, require_target=True)


def test_prepare_base_frame_rejects_nonbinary_target() -> None:
    frame = _frame()
    frame.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="binary classes"):
        prepare_base_frame(frame, require_target=True)


def test_week_profile_is_complete_and_metric_eligible() -> None:
    prepared = prepare_base_frame(_frame(), require_target=True)
    profile = build_week_profile(prepared)
    assert len(profile) == 40
    assert all(row.rows == 4 for row in profile)
    assert all(row.positives == 2 for row in profile)
    assert all(row.negatives == 2 for row in profile)
    assert all(row.metric_eligible for row in profile)


def test_holdout_candidates_are_contiguous_and_deterministic() -> None:
    prepared = prepare_base_frame(_frame(), require_target=True)
    profile = build_week_profile(prepared)
    first = build_holdout_candidates(profile)
    second = build_holdout_candidates(profile)
    assert first == second
    assert first
    for candidate in first:
        assert candidate.train_week_max < candidate.validation_week_min
        assert candidate.eligible
        assert candidate.validation_metric_eligible_weeks == candidate.validation_weeks


def test_temporal_integrity_detects_case_overlap() -> None:
    train = prepare_base_frame(_frame(), require_target=True)
    test_raw = train.loc[:3, ["case_id", "date_decision", "WEEK_NUM"]].copy()
    test_raw["WEEK_NUM"] = test_raw["WEEK_NUM"] + 100
    test_raw["date_decision"] = test_raw["date_decision"] + pd.Timedelta(days=700)
    test = prepare_base_frame(test_raw, require_target=False)
    integrity = temporal_integrity(train, test, build_week_profile(train))
    assert integrity.case_id_overlap == 4


def test_holdout_candidates_require_sufficient_temporal_history() -> None:
    prepared = prepare_base_frame(_frame(weeks=12), require_target=True)
    profile = build_week_profile(prepared)
    with pytest.raises(ValueError, match="not enough weeks"):
        build_holdout_candidates(profile)
