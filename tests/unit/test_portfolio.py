"""Real-evidence accounting and deliberate corruption of portfolio release claims."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from home_credit.modeling.portfolio import feature_accounting, load_portfolio, validate_release
from home_credit.modeling.portfolio_report import chart_pairs
from home_credit.modeling.release import read_object

ROOT = Path(__file__).resolve().parents[2]


def test_real_feature_accounting_closes_every_candidate():
    result = load_portfolio(ROOT)
    features = result["features"]
    assert features["candidate_count"] == 2508
    assert features["eligible_count"] == 2034
    assert features["retained_count"] == 700
    assert features["rejected_count"] == 1808
    assert features["decisions"] == {
        "missingness": 343,
        "constant": 129,
        "categorical_cardinality": 2,
        "ranking_budget": 1334,
        "retained": 700,
    }
    assert sum(r["candidates"] for r in features["families"]) == 2508


@pytest.mark.parametrize("fault", ["duplicate", "selected", "missingness", "eligible", "window"])
def test_feature_accounting_rejects_changed_evidence(fault):
    screen = read_object(ROOT / "reports/feature_ablation/feature_screen.json")
    plan = read_object(ROOT / "configs/benchmark_features.json")
    if fault == "duplicate":
        screen["result"]["scores"][1] = copy.deepcopy(screen["result"]["scores"][0])
    elif fault == "selected":
        screen["result"]["scores"][0]["selected"] = False
    elif fault == "missingness":
        screen["result"]["scores"][0]["missing_fraction"] = 1.0
    elif fault == "eligible":
        screen["result"]["eligible_candidates"] += 1
    else:
        screen["screening_config"]["validation_week_max"] = 33
        plan["screening_config"]["validation_week_max"] = 33
    with pytest.raises(ValueError):
        feature_accounting(screen, plan)


@pytest.mark.parametrize("fault", ["metric", "refit", "rows", "week", "intent", "early_stop"])
def test_release_rejects_misleading_or_inconsistent_results(fault):
    evidence = load_portfolio(ROOT)
    state, evaluation = evidence["state"], evidence["evaluation"]
    intent = copy.deepcopy(evaluation["intent"])
    if fault == "metric":
        evaluation["metrics"]["stability_score"] += 0.001
    elif fault == "refit":
        evaluation["all_label_model_evaluated"] = True
    elif fault == "rows":
        evaluation["rows"] -= 1
    elif fault == "week":
        evaluation["weekly"][0]["week"] = 72
    elif fault == "intent":
        intent["weights"]["lightgbm"] = 0.2
    else:
        state["stages"]["development/lightgbm"]["early_stopping_used"] = True
    with pytest.raises(ValueError):
        validate_release(ROOT, state, evaluation, intent)


def test_chart_values_match_feature_counts_ablations_and_holdout():
    evidence = load_portfolio(ROOT)
    for section in ("features", "release"):
        pairs = chart_pairs(evidence, section)
        try:
            assert len(pairs) == 2
            for _, interactive in pairs.values():
                assert interactive.to_json()
            if section == "release":
                figure = pairs["Holdout discrimination"][1]
                assert list(figure.data[0].y) == [
                    r["gini"] for r in evidence["evaluation"]["weekly"]
                ]
        finally:
            for static, _ in pairs.values():
                plt.close(static)
