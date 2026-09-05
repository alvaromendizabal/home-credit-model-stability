"""Leakage-aware temporal diagnostics and holdout design for Home Credit."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REQUIRED_BASE_COLUMNS = frozenset({"case_id", "date_decision", "WEEK_NUM"})
_TARGET = "target"


@dataclass(frozen=True, slots=True)
class BaseSummary:
    """Case-level temporal summary for one competition split."""

    split: str
    rows: int
    unique_cases: int
    weeks: int
    week_min: int
    week_max: int
    date_min: str
    date_max: str
    positives: int | None
    negatives: int | None
    positive_rate: float | None


@dataclass(frozen=True, slots=True)
class WeekProfile:
    """Target and date profile for one training week."""

    week_num: int
    rows: int
    positives: int
    negatives: int
    positive_rate: float
    date_min: str
    date_max: str
    metric_eligible: bool


@dataclass(frozen=True, slots=True)
class TemporalIntegrity:
    """Cross-split and week/date ordering checks."""

    case_id_overlap: int
    week_start_reversals: int
    week_end_reversals: int
    test_starts_after_train_week: bool
    test_starts_after_train_date: bool


@dataclass(frozen=True, slots=True)
class HoldoutCandidate:
    """One contiguous trailing-week out-of-time validation candidate."""

    name: str
    requested_validation_week_fraction: float
    train_week_min: int
    train_week_max: int
    validation_week_min: int
    validation_week_max: int
    train_weeks: int
    validation_weeks: int
    train_rows: int
    validation_rows: int
    validation_row_fraction: float
    train_positive_rate: float
    validation_positive_rate: float
    target_rate_shift: float
    validation_metric_eligible_weeks: int
    min_validation_positives_per_week: int
    min_validation_negatives_per_week: int
    all_validation_weeks_metric_eligible: bool
    eligible: bool


def _as_datetime_series(values: pd.Series[Any]) -> pd.Series[Any]:
    converted = pd.to_datetime(values, errors="coerce")
    return pd.Series(converted, index=values.index, name=values.name)


def prepare_base_frame(frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Validate and normalize a case-level base table without mutating the input."""
    required = set(_REQUIRED_BASE_COLUMNS)
    if require_target:
        required.add(_TARGET)

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"base table is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("base table must not be empty")

    case_numeric = pd.to_numeric(frame["case_id"], errors="coerce")
    week_numeric = pd.to_numeric(frame["WEEK_NUM"], errors="coerce")
    dates = _as_datetime_series(frame["date_decision"])

    if case_numeric.isna().any():
        raise ValueError("case_id must be numeric and non-null")
    if week_numeric.isna().any():
        raise ValueError("WEEK_NUM must be numeric and non-null")
    if dates.isna().any():
        raise ValueError("date_decision must be parseable and non-null")

    case_values = np.asarray(case_numeric.to_numpy(), dtype=np.float64)
    week_values = np.asarray(week_numeric.to_numpy(), dtype=np.float64)

    if not np.isfinite(case_values).all():
        raise ValueError("case_id must contain only finite values")
    if not np.isfinite(week_values).all():
        raise ValueError("WEEK_NUM must contain only finite values")
    if not np.equal(case_values, np.floor(case_values)).all():
        raise ValueError("case_id must contain integer values")
    if not np.equal(week_values, np.floor(week_values)).all():
        raise ValueError("WEEK_NUM must contain integer values")

    normalized = pd.DataFrame(
        {
            "case_id": case_numeric.astype("int64"),
            "date_decision": dates,
            "WEEK_NUM": week_numeric.astype("int64"),
        }
    )

    if normalized["case_id"].duplicated().any():
        raise ValueError("base table case_id values must be unique")

    if require_target:
        target_numeric = pd.to_numeric(frame[_TARGET], errors="coerce")
        if target_numeric.isna().any():
            raise ValueError("target must be numeric and non-null")
        target_values = np.asarray(target_numeric.to_numpy(), dtype=np.float64)
        if not np.isfinite(target_values).all():
            raise ValueError("target must contain only finite values")
        unique_targets = set(np.unique(target_values).tolist())
        if not unique_targets.issubset({0.0, 1.0}) or len(unique_targets) != 2:
            raise ValueError("target must contain both binary classes 0 and 1")
        normalized[_TARGET] = target_numeric.astype("int8")

    return normalized


def summarize_base(frame: pd.DataFrame, *, split: str) -> BaseSummary:
    """Summarize one normalized base table."""
    weeks = np.asarray(frame["WEEK_NUM"].to_numpy(), dtype=np.int64)
    dates = np.asarray(frame["date_decision"].to_numpy(), dtype="datetime64[ns]")

    positives: int | None = None
    negatives: int | None = None
    positive_rate: float | None = None
    if _TARGET in frame.columns:
        target = np.asarray(frame[_TARGET].to_numpy(), dtype=np.int8)
        positives = int(target.sum())
        negatives = int(target.size - positives)
        positive_rate = float(positives / target.size)

    return BaseSummary(
        split=split,
        rows=len(frame),
        unique_cases=int(frame["case_id"].nunique(dropna=False)),
        weeks=int(np.unique(weeks).size),
        week_min=int(weeks.min()),
        week_max=int(weeks.max()),
        date_min=str(np.datetime_as_string(dates.min(), unit="D")),
        date_max=str(np.datetime_as_string(dates.max(), unit="D")),
        positives=positives,
        negatives=negatives,
        positive_rate=positive_rate,
    )


def build_week_profile(frame: pd.DataFrame) -> list[WeekProfile]:
    """Build a deterministic weekly target profile in O(n log n) time."""
    if _TARGET not in frame.columns:
        raise ValueError("weekly target profile requires target")

    weeks = np.asarray(frame["WEEK_NUM"].to_numpy(), dtype=np.int64)
    target = np.asarray(frame[_TARGET].to_numpy(), dtype=np.int8)
    dates = np.asarray(frame["date_decision"].to_numpy(), dtype="datetime64[ns]")

    order = np.argsort(weeks, kind="stable")
    sorted_weeks = weeks[order]
    sorted_target = target[order]
    sorted_dates = dates[order]

    boundaries = np.flatnonzero(np.diff(sorted_weeks)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.array([len(sorted_weeks)], dtype=np.int64)))

    rows: list[WeekProfile] = []
    for start, stop in zip(starts, stops, strict=True):
        week_target = sorted_target[start:stop]
        week_dates = sorted_dates[start:stop]
        count = int(stop - start)
        positives = int(week_target.sum())
        negatives = count - positives
        rows.append(
            WeekProfile(
                week_num=int(sorted_weeks[start]),
                rows=count,
                positives=positives,
                negatives=negatives,
                positive_rate=float(positives / count),
                date_min=str(np.datetime_as_string(week_dates.min(), unit="D")),
                date_max=str(np.datetime_as_string(week_dates.max(), unit="D")),
                metric_eligible=positives > 0 and negatives > 0,
            )
        )
    return rows


def _count_date_reversals(profile: list[WeekProfile], *, field: str) -> int:
    if field not in {"date_min", "date_max"}:
        raise ValueError(f"unsupported date field: {field}")
    values = np.asarray(
        [np.datetime64(getattr(row, field)) for row in profile],
        dtype="datetime64[D]",
    )
    if values.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(values).astype("timedelta64[D]") < np.timedelta64(0, "D")))


def temporal_integrity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    profile: list[WeekProfile],
) -> TemporalIntegrity:
    """Measure temporal ordering and train/test identifier isolation."""
    train_cases = np.asarray(train["case_id"].to_numpy(), dtype=np.int64)
    test_cases = np.asarray(test["case_id"].to_numpy(), dtype=np.int64)
    overlap = int(np.intersect1d(train_cases, test_cases, assume_unique=True).size)

    train_weeks = np.asarray(train["WEEK_NUM"].to_numpy(), dtype=np.int64)
    test_weeks = np.asarray(test["WEEK_NUM"].to_numpy(), dtype=np.int64)
    train_dates = np.asarray(train["date_decision"].to_numpy(), dtype="datetime64[ns]")
    test_dates = np.asarray(test["date_decision"].to_numpy(), dtype="datetime64[ns]")

    return TemporalIntegrity(
        case_id_overlap=overlap,
        week_start_reversals=_count_date_reversals(profile, field="date_min"),
        week_end_reversals=_count_date_reversals(profile, field="date_max"),
        test_starts_after_train_week=bool(test_weeks.min() > train_weeks.max()),
        test_starts_after_train_date=bool(test_dates.min() > train_dates.max()),
    )


def build_holdout_candidates(
    profile: list[WeekProfile],
    *,
    validation_week_fractions: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30),
    min_train_weeks: int = 20,
    min_metric_weeks: int = 8,
) -> list[HoldoutCandidate]:
    """Create trailing contiguous-week candidates without choosing one prematurely."""
    if len(profile) < min_train_weeks + min_metric_weeks:
        raise ValueError("not enough weeks to construct a robust out-of-time holdout")
    if any(not 0.0 < fraction < 1.0 for fraction in validation_week_fractions):
        raise ValueError("validation week fractions must be between 0 and 1")

    total_rows = sum(row.rows for row in profile)
    candidates: list[HoldoutCandidate] = []
    seen_validation_weeks: set[int] = set()

    for fraction in validation_week_fractions:
        validation_weeks = max(min_metric_weeks, math.ceil(len(profile) * fraction))
        if validation_weeks in seen_validation_weeks:
            continue
        seen_validation_weeks.add(validation_weeks)

        train_weeks = len(profile) - validation_weeks
        if train_weeks < min_train_weeks:
            continue

        train_profile = profile[:train_weeks]
        validation_profile = profile[train_weeks:]

        train_rows = sum(row.rows for row in train_profile)
        validation_rows = sum(row.rows for row in validation_profile)
        train_positives = sum(row.positives for row in train_profile)
        validation_positives = sum(row.positives for row in validation_profile)
        train_rate = train_positives / train_rows
        validation_rate = validation_positives / validation_rows
        metric_weeks = sum(row.metric_eligible for row in validation_profile)
        min_positives = min(row.positives for row in validation_profile)
        min_negatives = min(row.negatives for row in validation_profile)
        all_metric_eligible = metric_weeks == len(validation_profile)
        eligible = (
            metric_weeks >= min_metric_weeks
            and train_positives > 0
            and train_positives < train_rows
            and validation_positives > 0
            and validation_positives < validation_rows
        )

        candidates.append(
            HoldoutCandidate(
                name=f"tail_{round(fraction * 100):02d}pct_weeks",
                requested_validation_week_fraction=fraction,
                train_week_min=train_profile[0].week_num,
                train_week_max=train_profile[-1].week_num,
                validation_week_min=validation_profile[0].week_num,
                validation_week_max=validation_profile[-1].week_num,
                train_weeks=len(train_profile),
                validation_weeks=len(validation_profile),
                train_rows=train_rows,
                validation_rows=validation_rows,
                validation_row_fraction=float(validation_rows / total_rows),
                train_positive_rate=float(train_rate),
                validation_positive_rate=float(validation_rate),
                target_rate_shift=float(validation_rate - train_rate),
                validation_metric_eligible_weeks=metric_weeks,
                min_validation_positives_per_week=min_positives,
                min_validation_negatives_per_week=min_negatives,
                all_validation_weeks_metric_eligible=all_metric_eligible,
                eligible=eligible,
            )
        )

    if not candidates:
        raise ValueError("no out-of-time holdout candidates could be constructed")
    return candidates


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(payload: object, output: Path) -> None:
    """Atomically write deterministic JSON."""
    _atomic_write(output, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_week_profile(profile: list[WeekProfile], output: Path) -> None:
    """Atomically write the deterministic weekly profile as JSONL."""
    payload = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in profile)
    _atomic_write(output, payload)


def as_serializable(value: Any) -> dict[str, object]:
    """Convert one dataclass-like audit record to a typed dictionary."""
    return cast(dict[str, object], asdict(value))
