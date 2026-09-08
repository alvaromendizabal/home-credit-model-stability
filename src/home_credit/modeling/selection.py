"""Bounded development-only ensembles from aligned, previously saved predictions.

No model is loaded or fitted here. The fixed candidate grid is a development
selection exercise, not an unbiased estimate of future generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_integer_dtype
from sklearn.metrics import roc_auc_score

from home_credit.metrics.classification import evaluate_probabilities


@dataclass(frozen=True)
class AlignedPredictions:
    """One common, explicitly verified evaluation population."""

    case_id: NDArray[np.int64]
    target: NDArray[np.int64]
    week: NDArray[np.int64]
    fold: NDArray[np.int64]
    predictions: dict[str, NDArray[np.float64]]


def require(condition: bool, message: str) -> None:
    """Keep input guards active even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def validate_windows(windows: list[dict[str, int]]) -> dict[int, int]:
    """Map each contiguous development week to exactly one expanding fold."""
    require(bool(windows), "no temporal folds")
    mapping: dict[int, int] = {}
    previous_train = -1
    previous_end: int | None = None
    for index, window in enumerate(windows, 1):
        require(window["fold"] == index, "fold numbers must be contiguous")
        start, end = window["validation_week_min"], window["validation_week_max"]
        require(0 <= start < end < 73, "holdout overlap or invalid validation window")
        require(window["train_week_min"] == 0, "training must expand from week zero")
        require(previous_train < window["train_week_max"] < start, "training overlap")
        require(window["train_week_max"] == start - 1, "unexpected temporal gap")
        require(previous_end is None or start == previous_end + 1, "validation gap or overlap")
        for week in range(start, end + 1):
            require(week not in mapping, "overlapping validation folds")
            mapping[week] = index
        previous_train, previous_end = window["train_week_max"], end
    return mapping


def align_predictions(
    frames: dict[str, pd.DataFrame],
    windows: list[dict[str, int]],
    *,
    expected_rows: int,
) -> AlignedPredictions:
    """Reject silent joins, duplicated cases, changed labels, and holdout rows."""
    require(bool(frames) and expected_rows > 0, "empty evaluation population")
    mapping = validate_windows(windows)
    metadata = ["case_id", "target", "WEEK_NUM", "fold"]
    reference: pd.DataFrame | None = None
    predictions: dict[str, NDArray[np.float64]] = {}
    for name, raw in frames.items():
        require(set([*metadata, "prediction"]) <= set(raw.columns), f"missing columns: {name}")
        require(raw.columns.is_unique, f"duplicate columns: {name}")
        frame = raw.sort_values("case_id", kind="stable").reset_index(drop=True)
        require(len(frame) == expected_rows, f"row coverage mismatch: {name}")
        require(not bool(frame[[*metadata, "prediction"]].isna().any().any()), f"nulls: {name}")
        for column in metadata:
            require(is_integer_dtype(frame[column].dtype), f"noninteger {column}: {name}")
        require(frame["case_id"].is_unique, f"duplicate case_id: {name}")
        require(bool((frame["case_id"] >= 0).all()), f"negative case_id: {name}")
        require(set(frame["WEEK_NUM"].unique()) == set(mapping), f"week coverage mismatch: {name}")
        require(
            bool((frame["fold"] == frame["WEEK_NUM"].map(mapping)).all()),
            f"fold/week disagreement: {name}",
        )
        require(set(frame["target"].unique()) == {0, 1}, f"nonbinary target: {name}")
        require(
            bool((frame.groupby("WEEK_NUM")["target"].nunique() == 2).all()),
            f"single-class week: {name}",
        )
        current = frame[metadata].astype("int64")
        if reference is None:
            reference = current
        else:
            require(reference.equals(current), f"case/target/week alignment mismatch: {name}")
        probability = frame["prediction"].to_numpy(dtype=np.float64)
        require(bool(np.isfinite(probability).all()), f"nonfinite probability: {name}")
        require(
            bool(((probability >= 0) & (probability <= 1)).all()), f"invalid probability: {name}"
        )
        predictions[name] = probability
    assert reference is not None  # Established by the nonempty input guard.
    return AlignedPredictions(
        case_id=reference["case_id"].to_numpy(dtype=np.int64),
        target=reference["target"].to_numpy(dtype=np.int64),
        week=reference["WEEK_NUM"].to_numpy(dtype=np.int64),
        fold=reference["fold"].to_numpy(dtype=np.int64),
        predictions=predictions,
    )


def fixed_candidates(names: list[str], leader: str) -> dict[str, dict[str, float]]:
    """Four single models, nine two-model blends, and two equal-weight blends."""
    require(len(names) == len(set(names)) == 4 and leader in names, "expected four unique models")
    require(all(name and name.replace("_", "").isalnum() for name in names), "invalid model name")
    candidates = {name: {name: 1.0} for name in names}
    for other in names:
        if other == leader:
            continue
        for weight in (0.5, 0.75, 0.9):
            name = f"{leader}_{int(weight * 100):02d}_{other}"
            candidates[name] = {leader: weight, other: 1.0 - weight}
    complementary = [name for name in names if name != "lightgbm"]
    require(len(complementary) == 3, "the original lightgbm control is required")
    candidates["equal_three_families"] = {name: 1.0 / 3.0 for name in complementary}
    candidates["equal_all_four"] = {name: 0.25 for name in names}
    require(len(candidates) == 15, "candidate names collided")
    return candidates


def blend(data: AlignedPredictions, weights: dict[str, float]) -> NDArray[np.float64]:
    """Average probabilities without silently normalizing an invalid weight plan."""
    require(bool(weights) and set(weights) <= set(data.predictions), "unknown ensemble member")
    require(all(math.isfinite(w) and 0 < w <= 1 for w in weights.values()), "invalid weight")
    require(
        math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0),
        "weights must sum to one",
    )
    result = np.zeros(len(data.target), dtype=np.float64)
    for name, weight in weights.items():
        result += weight * data.predictions[name]
    return result


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, float]:
    """Use the locked mean-fold objective and all declared secondary metrics."""
    require(bool(folds), "empty fold summary")
    return {
        "mean_fold_stability": float(np.mean([r["stability_score"] for r in folds])),
        "worst_fold_stability": min(float(r["stability_score"]) for r in folds),
        "mean_weekly_gini": float(np.mean([r["mean_gini"] for r in folds])),
        "mean_temporal_slope": float(np.mean([r["temporal_slope"] for r in folds])),
        "mean_residual_std": float(np.mean([r["residual_std"] for r in folds])),
        "mean_brier_score": float(np.mean([r["brier_score"] for r in folds])),
    }


def evaluate_candidate(
    data: AlignedPredictions, name: str, weights: dict[str, float]
) -> dict[str, Any]:
    """Recompute every metric from raw saved prediction ranks."""
    probability = blend(data, weights)
    folds: list[dict[str, Any]] = []
    for number in sorted(set(data.fold.tolist())):
        mask = data.fold == number
        metrics = evaluate_probabilities(data.target[mask], probability[mask], data.week[mask])
        folds.append({"fold": number, **metrics})
    pooled = evaluate_probabilities(data.target, probability, data.week)
    return {
        "name": name,
        "weights": weights,
        "state": "complete",
        "folds": folds,
        "metrics": {
            **summarize_folds(folds),
            **{f"oof_{key}": pooled[key] for key in ("auc", "pr_auc", "brier_score", "log_loss")},
        },
    }


def rank_records(records: list[dict[str, Any]], leader: str) -> list[dict[str, Any]]:
    """Retain the single-model incumbent on an exact tie after protocol tie-breaks."""

    def key(record: dict[str, Any]) -> tuple[float, float, float, float, float, float, bool, str]:
        metrics = record["metrics"]
        return (
            -float(metrics["mean_fold_stability"]),
            -float(metrics["worst_fold_stability"]),
            -float(metrics["mean_weekly_gini"]),
            -float(metrics["mean_temporal_slope"]),
            float(metrics["mean_residual_std"]),
            float(metrics["mean_brier_score"]),
            record["name"] != leader,
            str(record["name"]),
        )

    return sorted(records, key=key)


def validate_record(
    record: dict[str, Any], name: str, weights: dict[str, float], folds: int
) -> None:
    """Validate a restored candidate rather than trusting a completion flag."""
    require(record["name"] == name and record["weights"] == weights, "cached candidate changed")
    require(record["state"] == "complete", "incomplete cached candidate")
    rows = record["folds"]
    require([row["fold"] for row in rows] == list(range(1, folds + 1)), "cached fold coverage")
    for row in rows:
        require(all(math.isfinite(float(v)) for v in row.values()), "nonfinite cached fold")
    require(
        all(math.isfinite(float(value)) for value in record["metrics"].values()),
        "nonfinite cached metrics",
    )
    for metric, expected in summarize_folds(rows).items():
        require(
            math.isclose(float(record["metrics"][metric]), expected, rel_tol=1e-10, abs_tol=1e-12),
            f"cached summary mismatch: {metric}",
        )


def prequential_choices(records: list[dict[str, Any]], leader: str) -> list[dict[str, Any]]:
    """Select weights using earlier folds only; base-model tuning is NOT nested."""
    require(bool(records), "no candidates for prequential diagnostics")
    numbers = [row["fold"] for row in records[0]["folds"]]
    choices: list[dict[str, Any]] = []
    for number in numbers[1:]:
        historical = []
        for record in records:
            earlier = [row for row in record["folds"] if row["fold"] < number]
            historical.append({"name": record["name"], "metrics": summarize_folds(earlier)})
        chosen = rank_records(historical, leader)[0]["name"]
        record = next(row for row in records if row["name"] == chosen)
        observed = next(row for row in record["folds"] if row["fold"] == number)
        choices.append(
            {
                "fold": number,
                "selected_using_folds": numbers[: number - 1],
                "candidate": chosen,
                **observed,
            }
        )
    return choices


def diagnostics(data: AlignedPredictions, weights: dict[str, float]) -> dict[str, Any]:
    """Aggregate-only weekly discrimination, reliability, and model diversity."""
    probability = blend(data, weights)
    weekly: list[dict[str, Any]] = []
    for week in sorted(set(data.week.tolist())):
        mask = data.week == week
        weekly.append(
            {
                "week": week,
                "fold": int(data.fold[mask][0]),
                "rows": int(mask.sum()),
                "positive_rate": float(data.target[mask].mean()),
                "mean_prediction": float(probability[mask].mean()),
                "gini": float(2 * roc_auc_score(data.target[mask], probability[mask]) - 1),
            }
        )
    # Equal values stay together; repeated quantiles do not produce empty fake bins.
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, 11)))
    assignments = np.searchsorted(edges[1:-1], probability, side="right")
    reliability = []
    for number in sorted(set(assignments.tolist())):
        mask = assignments == number
        reliability.append(
            {
                "bin": number + 1,
                "rows": int(mask.sum()),
                "mean_prediction": float(probability[mask].mean()),
                "observed_default_rate": float(data.target[mask].mean()),
            }
        )
    names = list(data.predictions)
    pairs = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            a, b = data.predictions[first], data.predictions[second]
            # A constant predictor has undefined correlation; report null, never NaN.
            correlation = (
                None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
            )
            pairs.append(
                {"first": first, "second": second, "pearson_prediction_correlation": correlation}
            )
    return {"weekly": weekly, "reliability": reliability, "prediction_correlations": pairs}
