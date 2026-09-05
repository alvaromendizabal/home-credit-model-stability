from __future__ import annotations

import numpy as np
import pytest

from home_credit.metrics.stability import normalized_gini, stability_score


def test_normalized_gini_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert normalized_gini(y_true, y_score) == pytest.approx(1.0)


def test_normalized_gini_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        normalized_gini(np.zeros(4), np.arange(4))


def test_normalized_gini_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        normalized_gini(np.array([0, 1]), np.array([0.1]))


def test_stability_score_is_deterministic() -> None:
    target = np.tile(np.array([0, 1, 0, 1]), 4)
    prediction = np.tile(np.array([0.1, 0.9, 0.2, 0.8]), 4)
    weeks = np.repeat(np.arange(4), 4)
    first = stability_score(target, prediction, weeks)
    second = stability_score(target, prediction, weeks)
    assert first == second
    assert first.score == pytest.approx(1.0)


def test_stability_score_accepts_integer_week_numbers() -> None:
    target = np.tile(np.array([0, 1, 0, 1]), 3)
    prediction = np.tile(np.array([0.1, 0.9, 0.2, 0.8]), 3)
    weeks = np.repeat(np.arange(3, dtype=np.int64), 4)
    result = stability_score(target, prediction, weeks)
    assert result.score == pytest.approx(1.0)


def test_stability_score_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        stability_score(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0, 0, 1]),
        )


def test_stability_penalizes_negative_slope() -> None:
    weeks = np.repeat(np.arange(4), 100)
    target = np.tile(np.r_[np.zeros(50), np.ones(50)], 4).astype(int)
    prediction_parts = []
    for degradation in (0.0, 0.15, 0.3, 0.45):
        base = np.r_[
            np.linspace(0.0, 0.4 + degradation, 50),
            np.linspace(0.6 - degradation, 1.0, 50),
        ]
        prediction_parts.append(base)
    result = stability_score(target, np.concatenate(prediction_parts), weeks)
    assert result.slope <= 0.0
