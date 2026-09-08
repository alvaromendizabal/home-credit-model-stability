"""Prove that current/future outcomes cannot enter calibration fitting."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from home_credit.modeling.calibration_research import (
    apply_calibrator,
    fit_calibrator,
    fit_prior_folds,
    reliability,
    validate_policy,
)
from home_credit.modeling.selection import AlignedPredictions

ROOT = Path(__file__).resolve().parents[2]


def population() -> AlignedPredictions:
    rng = np.random.default_rng(21)
    weeks = np.repeat(np.arange(33, 73), 100)
    p = rng.uniform(0.02, 0.8, len(weeks))
    return AlignedPredictions(
        np.arange(len(weeks)),
        rng.binomial(1, p),
        weeks,
        (weeks - 33) // 8 + 1,
        {"tuned": p, "original": p * 0.9},
    )


@pytest.mark.parametrize("method", ["uncalibrated", "sigmoid", "isotonic"])
def test_calibration_ignores_current_and_future_labels(method: str) -> None:
    data = population()
    target = data.target.copy()
    target[data.fold >= 3] = 1 - target[data.fold >= 3]
    before, p = fit_prior_folds(data, 3, method)
    after, q = fit_prior_folds(replace(data, target=target), 3, method)
    assert before["model"] == after["model"]
    assert np.array_equal(p, q)
    assert before["fit_week_max"] < before["evaluation_week_min"]
    assert before["metrics"] != after["metrics"]


@pytest.mark.parametrize("method", ["uncalibrated", "sigmoid", "isotonic"])
def test_portable_calibrator_parameters_preserve_predictions(method: str) -> None:
    data = population()
    model = fit_calibrator(data.predictions["tuned"], data.target, method)
    probe = np.array([0.0, 0.01, 0.02, 0.3, 0.6, 0.8, 0.99, 1.0])
    expected = apply_calibrator(probe, model)
    actual = apply_calibrator(probe, json.loads(json.dumps(model, allow_nan=False)))
    assert np.array_equal(expected, actual)
    assert (np.diff(actual) >= 0).all()
    assert np.isfinite(actual).all()


def test_holdout_or_first_fold_is_rejected() -> None:
    data = population()
    with pytest.raises(ValueError):
        fit_prior_folds(data, 1, "sigmoid")
    with pytest.raises(ValueError):
        fit_prior_folds(replace(data, week=data.week + 20), 3, "sigmoid")


@pytest.mark.parametrize(
    "parameters",
    [
        {"method": "sigmoid", "coefficient": -1, "intercept": 0},
        {"method": "isotonic", "x": [0.1, 0.1], "y": [0.2, 0.4]},
        {"method": "isotonic", "x": [0.1, 0.2], "y": [0.4, 0.2]},
    ],
)
def test_invalid_saved_models_are_rejected(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        apply_calibrator([0.1, 0.2], parameters)


def test_reliability_includes_zero_one_and_exact_edges() -> None:
    frame = pl.DataFrame({"prediction": [0.0, 0.1, 0.2, 0.9, 1.0], "target": [0, 0, 1, 1, 1]})
    bins = reliability(frame, [0.0, 0.1, 0.2, 1.0])
    assert [b["rows"] for b in bins] == [1, 1, 3]
    assert sum(b["rows"] for b in bins) == len(frame)


def test_only_predeclared_methods_and_temporal_scope_are_allowed() -> None:
    policy = json.loads((ROOT / "configs/calibration_research.json").read_text())
    validate_policy(policy)
    policy["holdout_evaluation_allowed"] = True
    with pytest.raises(ValueError):
        validate_policy(policy)
