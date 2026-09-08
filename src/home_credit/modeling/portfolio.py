"""Read-only portfolio evidence: feature accounting and the frozen release result.

These summaries never fit a model, choose features, or read borrower-level data.
Every published input is pinned to the experiment that originally produced it.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from home_credit.modeling.release import checked_member, read_object, verify_file
from home_credit.modeling.selection import require


def feature_accounting(screen: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the historical decision rules from stored training-window statistics."""
    require(screen["schema_version"] == 1, "unsupported feature screen")
    require(screen["identity"] == plan["identity"], "feature screening lineage changed")
    config, result = screen["screening_config"], screen["result"]
    require(config == plan["screening_config"], "screening policy changed")
    require(config["validation_week_max"] < 33, "screening reaches model validation")
    scores = result["scores"]
    names = [row["name"] for row in scores]
    require(len(names) == len(set(names)) == result["structural_candidates"], "candidate mismatch")
    require(result["selected_features"] == plan["selected_features"], "selected features changed")
    selected = {row["name"] for row in plan["selected_features"]}
    decisions: Counter[str] = Counter()
    families: dict[str, Counter[str]] = {}
    catalog = []
    for row in scores:
        missing, unique = row["missing_fraction"], row["unique_values"]
        require(math.isfinite(missing) and 0 <= missing <= 1, "invalid feature missingness")
        require(isinstance(unique, int) and unique >= 1, "invalid feature cardinality")
        require(row["selected"] == (row["name"] in selected), "selection flag mismatch")
        if missing > config["max_missing_fraction"]:
            reason = "missingness"
        elif unique <= 1:
            reason = "constant"
        elif row["categorical"] and unique > config["max_categorical_cardinality"]:
            reason = "categorical_cardinality"
        else:
            reason = "retained" if row["selected"] else "ranking_budget"
        require(not row["selected"] or reason == "retained", "structurally rejected selection")
        decisions[reason] += 1
        family = families.setdefault(row["family"], Counter())
        family["candidates"] += 1
        family[reason] += 1
        catalog.append({**row, "decision": reason})
    require(decisions["retained"] == len(selected), "retained count mismatch")
    require(
        decisions["retained"] + decisions["ranking_budget"] == result["eligible_candidates"],
        "eligible count mismatch",
    )
    return {
        "candidate_count": len(scores),
        "eligible_count": result["eligible_candidates"],
        "retained_count": len(selected),
        "rejected_count": len(scores) - len(selected),
        "decisions": dict(sorted(decisions.items())),
        "families": [{"family": name, **counts} for name, counts in sorted(families.items())],
        "catalog": catalog,
        "screening_weeks": [config["train_week_min"], config["validation_week_max"]],
        "drift_validation_auc": result["drift_validation_auc"],
    }


def validate_release(
    root: Path, state: dict[str, Any], evaluation: dict[str, Any], intent: dict[str, Any]
) -> None:
    """Bind the public holdout result to predeclared development models and a distinct refit."""
    plan = read_object(root / "configs/model_release.json")
    require(state["schema_version"] == 1 and state["complete"] is True, "incomplete release")
    require(evaluation["intent"] == intent, "holdout intent mismatch")
    require(evaluation["release_identity"] == state["identity"], "evaluation lineage mismatch")
    for name, digest in state["identity"]["inputs"].items():
        verify_file(checked_member(root, name), digest)
    require(evaluation["evaluated_model_phase"] == "development", "wrong evaluated model")
    require(evaluation["all_label_model_evaluated"] is False, "all-label evaluation claim")
    require(evaluation["calibration"] == plan["calibration"] == "none", "calibration changed")
    require(evaluation["fit_weeks"] == plan["holdout_fit_weeks"] == [0, 72], "fit scope changed")
    require(evaluation["evaluation_weeks"] == [73, 91], "holdout scope changed")
    require(intent["weights"] == plan["selection"]["weights"], "frozen weights changed")
    for name in plan["components"]:
        dev = state["stages"][f"development/{name}"]
        final = state["stages"][f"all_labels/{name}"]
        require(dev["model"]["sha256"] == intent["models"][name], "evaluated model changed")
        require(dev["encoder"]["sha256"] == intent["encoders"][name], "evaluated encoder changed")
        require(dev["fit_weeks"] == [0, 72] and final["fit_weeks"] == [0, 91], "phase mismatch")
        require(dev["fit_rows"] == 1323314 and final["fit_rows"] == 1526659, "fit count changed")
        require(dev["model"]["sha256"] != final["model"]["sha256"], "refit model conflation")
        for record in (dev, final):
            require(record["early_stopping_used"] is False, "holdout early stopping")
            require(record["features"] == plan["feature_count"] == 700, "feature count changed")
            require(record["reload_max_absolute_error"] <= 1e-12, "native reload mismatch")
            require(
                record["requested_rounds"] == plan["components"][name]["num_boost_round"],
                "iteration policy changed",
            )
    weekly = evaluation["weekly"]
    require([r["week"] for r in weekly] == list(range(73, 92)), "weekly coverage changed")
    require(sum(r["rows"] for r in weekly) == evaluation["rows"] == 203345, "holdout count")
    require(all(0 < r["positives"] < r["rows"] for r in weekly), "single-class holdout week")
    gini = np.array([r["gini"] for r in weekly], dtype=float)
    require(bool(np.isfinite(gini).all()) and bool((np.abs(gini) <= 1).all()), "invalid Gini")
    slope, intercept = np.polyfit(np.arange(len(gini)), gini, 1)
    residual = float(np.std(gini - (slope * np.arange(len(gini)) + intercept)))
    expected = {
        "mean_gini": float(gini.mean()),
        "temporal_slope": float(slope),
        "residual_std": residual,
        "stability_score": float(gini.mean() + 88 * min(slope, 0) - 0.5 * residual),
    }
    for name, value in expected.items():
        require(math.isclose(value, evaluation["metrics"][name], abs_tol=1e-12), "metric mismatch")
    for name in ("auc", "pr_auc", "brier_score", "log_loss"):
        value = evaluation["metrics"][name]
        require(math.isfinite(value) and value >= 0, "invalid probability metric")
        require(name == "log_loss" or value <= 1, "invalid bounded metric")
    bins = evaluation["reliability"]
    require(sum(r["rows"] for r in bins) == evaluation["rows"], "reliability support mismatch")
    for row in bins:
        require(row["rows"] > 0, "empty reliability bin")
        require(0 <= row["mean_prediction"] <= 1, "invalid reliability prediction")
        require(0 <= row["observed_default_rate"] <= 1, "invalid reliability rate")


def load_portfolio(root: Path) -> dict[str, Any]:
    """Load only immutable aggregate evidence; a changed digest fails closed."""
    policy = read_object(root / "configs/portfolio_review.json")
    require(policy["schema_version"] == 1, "unsupported publication policy")
    for entry in policy["artifacts"]:
        verify_file(checked_member(root, entry["path"]), entry["sha256"])
    state = read_object(root / "reports/model_release/state.json")
    evaluation = read_object(root / "reports/model_release/evaluation.json")
    intent = read_object(root / "reports/model_release/holdout_intent.json")
    validate_release(root, state, evaluation, intent)
    screen = read_object(root / "reports/feature_ablation/feature_screen.json")
    features = feature_accounting(screen, read_object(root / "configs/benchmark_features.json"))
    return {
        "policy": policy,
        "state": state,
        "evaluation": evaluation,
        "features": features,
        "ablation": read_object(root / "reports/feature_ablation/comparison.json"),
    }
