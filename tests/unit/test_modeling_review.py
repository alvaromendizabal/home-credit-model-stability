from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from home_credit.modeling.acceptance import read_json
from home_credit.modeling.review import (
    load_review,
    rescore_predictions,
    review_evidence,
    stability_components,
)
from home_credit.modeling.runner import _evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]


def test_extreme_probabilities_preserve_rank_in_training_evaluator() -> None:
    target = np.tile([0, 0, 1, 1], 2)
    prediction = np.tile([1e-12, 2e-12, 3e-12, 4e-12], 2)
    metrics = _evaluate_predictions(target, prediction, np.repeat([33, 34], 4))
    assert metrics["auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["stability_score"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(np.mean((prediction - target) ** 2), abs=1e-12)
    assert np.isfinite(metrics["log_loss"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_probability_validation_rejects_invalid_predictions(value: float) -> None:
    with pytest.raises(ValueError):
        _evaluate_predictions(
            np.array([0, 1, 0, 1]), np.array([value, 0.5, 0.2, 0.8]), np.array([1, 1, 2, 2])
        )


def test_stability_components_match_analytic_decline_and_improvement() -> None:
    decline = stability_components([10, 11, 12], [1, 0.5, 0])
    assert decline["stability_score"] == pytest.approx(-43.5)
    assert decline["slope_penalty"] == pytest.approx(-44)
    improve = stability_components([10, 11, 12], [0, 0.5, 1])
    assert improve["stability_score"] == pytest.approx(0.5)
    assert improve["slope_penalty"] == 0


@pytest.mark.parametrize(
    ("weeks", "ginis"),
    [([1, 1], [0.5, 0.6]), ([1], [0.5]), ([1, 2], [np.nan, 0.5]), ([1, 2], [2, 0.5])],
)
def test_invalid_weekly_components_are_rejected(weeks: list, ginis: list) -> None:
    with pytest.raises(ValueError):
        stability_components(weeks, ginis)


def test_real_review_reconciles_raw_metrics_and_archived_scores() -> None:
    _, review = load_review(ROOT)
    assert len(review["folds"]) == 20
    assert len(review["weekly_population"]) == 40
    assert review["leader"] == "lightgbm"
    leader = review["models"][0]
    assert leader["mean_fold_stability"] == pytest.approx(0.5851883723923733)
    assert leader["mean_fold_stability"] == pytest.approx(
        leader["mean_gini"] + leader["mean_slope_penalty"] + leader["mean_residual_penalty"]
    )
    baseline = next(r for r in review["models"] if r["model"] == "linear_logistic")
    assert baseline["mean_fold_stability"] == pytest.approx(-0.12098976074743428)
    assert baseline["mean_fold_stability"] != baseline["archived_mean_fold_stability"]


@pytest.mark.parametrize(
    "change", ["weekly_gini", "duplicate_week", "fold_metric", "missing_fold", "holdout"]
)
def test_review_rejects_inconsistent_aggregate_evidence(change: str) -> None:
    evidence = copy.deepcopy(read_json(ROOT / "reports/benchmark/acceptance.json"))
    protocol = read_json(ROOT / "configs/validation_protocol.json")
    metrics = read_json(ROOT / "reports/benchmark/metrics.json")
    if change == "weekly_gini":
        evidence["weekly_metrics"][0]["gini"] += 0.01
    elif change == "duplicate_week":
        evidence["weekly_metrics"][1] = evidence["weekly_metrics"][0]
    elif change == "fold_metric":
        evidence["fold_metrics"][0]["stability_score"] += 0.01
    elif change == "missing_fold":
        evidence["fold_metrics"].pop()
    else:
        evidence["holdout_predictions_present"] = True
    with pytest.raises(ValueError):
        review_evidence(evidence, protocol, metrics)


def test_rescoring_rejects_prediction_path_traversal(tmp_path: Path) -> None:
    evidence = read_json(ROOT / "reports/benchmark/acceptance.json")
    evidence["models"][0]["oof_path"] = "../outside.parquet"
    with pytest.raises(ValueError, match="escaped"):
        rescore_predictions(
            evidence, read_json(ROOT / "configs/validation_protocol.json"), tmp_path
        )
