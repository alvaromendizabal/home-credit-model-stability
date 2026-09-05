"""Home Credit competition stability metric and supporting weekly Gini statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Components of the Home Credit stability score."""

    score: float
    mean_gini: float
    slope: float
    residual_std: float


def _as_1d_array(name: str, values: ArrayLike) -> NDArray[np.generic]:
    """Convert an array-like input to one dimension and reject ambiguous shapes."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def normalized_gini(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Return normalized Gini, equal to ``2 * AUC - 1``."""
    target = _as_1d_array("y_true", y_true)
    score = np.asarray(_as_1d_array("y_score", y_score), dtype=np.float64)
    if target.size != score.size:
        raise ValueError("y_true and y_score must have equal lengths")
    if not np.isfinite(score).all():
        raise ValueError("y_score must contain only finite values")
    if np.unique(target).size < 2:
        raise ValueError("normalized_gini requires both target classes")
    return float(2.0 * roc_auc_score(target, score) - 1.0)


def stability_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    week_num: ArrayLike,
    *,
    slope_penalty: float = 88.0,
    residual_penalty: float = 0.5,
) -> StabilityResult:
    """Compute the official-style weekly Gini stability objective.

    The score is mean weekly Gini plus a penalty for a negative temporal slope
    minus a penalty for residual variability around the linear trend.
    """
    target = _as_1d_array("y_true", y_true)
    prediction: NDArray[np.float64] = np.asarray(_as_1d_array("y_score", y_score), dtype=np.float64)
    weeks: NDArray[np.float64] = np.asarray(_as_1d_array("week_num", week_num), dtype=np.float64)

    if not (target.size == prediction.size == weeks.size):
        raise ValueError("y_true, y_score, and week_num must have equal lengths")
    if not np.isfinite(prediction).all():
        raise ValueError("y_score must contain only finite values")
    if not np.isfinite(weeks).all():
        raise ValueError("week_num must contain only finite numeric values")

    rows: list[tuple[float, float]] = []
    for week in np.unique(weeks):
        mask = weeks == week
        week_target = target[mask]
        if np.unique(week_target).size < 2:
            continue
        week_prediction = prediction[mask]
        rows.append((float(week), normalized_gini(week_target, week_prediction)))

    if len(rows) < 2:
        raise ValueError("stability_score requires at least two valid weeks")

    valid_weeks = np.asarray([row[0] for row in rows], dtype=np.float64)
    ginis = np.asarray([row[1] for row in rows], dtype=np.float64)
    slope, intercept = np.polyfit(valid_weeks, ginis, deg=1)
    trend = slope * valid_weeks + intercept
    residual_std = float(np.std(ginis - trend, ddof=0))
    mean_gini = float(np.mean(ginis))
    score = mean_gini + slope_penalty * min(0.0, float(slope)) - residual_penalty * residual_std
    return StabilityResult(
        score=float(score),
        mean_gini=mean_gini,
        slope=float(slope),
        residual_std=residual_std,
    )
