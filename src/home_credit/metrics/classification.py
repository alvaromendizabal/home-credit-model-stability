"""Probability evaluation that preserves ranking at extreme predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from home_credit.metrics.stability import stability_score


def evaluate_probabilities(
    target: NDArray[np.generic], prediction: NDArray[np.generic], weeks: NDArray[np.generic]
) -> dict[str, float]:
    """Use raw ranks for Gini/AUC/AP and raw probabilities for Brier.

    Only log loss clips probabilities (to [1e-7, 1-1e-7]), retaining the
    benchmark's explicit numerical convention for logarithms.
    """
    if not (target.ndim == prediction.ndim == weeks.ndim == 1):
        raise ValueError("metric arrays must be one-dimensional")
    if not (len(target) == len(prediction) == len(weeks)):
        raise ValueError("metric arrays must have equal length")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("both binary target classes are required")
    probability = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("predictions must be finite")
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("predictions must be probabilities in [0, 1]")
    result = stability_score(target, probability, weeks)
    return {
        "stability_score": result.score,
        "mean_gini": result.mean_gini,
        "temporal_slope": result.slope,
        "residual_std": result.residual_std,
        "auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
