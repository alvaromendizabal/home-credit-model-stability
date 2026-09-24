#!/usr/bin/env python3
"""Verify and summarize the committed post-release frontier evidence."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/applprev_histogram"
SUMMARY = ROOT / "reports/post_release_frontier/summary.json"
DENSE = ROOT / "reports/denselight_frontier/contract.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def build_summary() -> dict[str, Any]:
    rows: list[dict[str, str]]
    with (REPORT / "fold_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30:
        raise ValueError(f"expected 30 fold-metric rows, found {len(rows)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    for name in ("saved_champion", "hist_augmented", "blend_hist_25"):
        if len(grouped[name]) != 5:
            raise ValueError(f"expected five folds for {name}")

    def mean(model: str, key: str) -> float:
        return sum(float(row[key]) for row in grouped[model]) / len(grouped[model])

    champion = grouped["saved_champion"]
    probe = grouped["blend_hist_25"]
    wins = sum(
        float(new["stability"]) > float(old["stability"])
        for new, old in zip(probe, champion, strict=True)
    )

    confirmation = _load_json(REPORT / "confirmation_decision.json")
    external = _load_json(REPORT / "external_probe_decision.json")
    dense = _load_json(DENSE)

    if confirmation["passed"] is not False:
        raise ValueError("standalone histogram confirmation must remain rejected")
    if external["internal_promotion_passed"] is not False or external["post_selection"] is not True:
        raise ValueError("external probe disclosure changed")
    if external["selected"] != "blend_hist_25" or external["new_model_weight"] != 0.25:
        raise ValueError("external probe identity changed")
    if dense["status"] != "registered_not_executed":
        raise ValueError("DenseLight contract no longer represents preregistration evidence")
    if dense["fit_budget"]["maximum_substantive_fits"] != 6:
        raise ValueError("DenseLight fit budget changed")
    if max(end for _, end in dense["data"]["confirmation_folds"]) > 72:
        raise ValueError("DenseLight selection touched the observed holdout")

    return {
        "schema_version": 1,
        "frozen_release": {
            "development_mean_stability": 0.601898890294407,
            "local_holdout_stability": 0.729674,
            "kaggle_public": 0.56062,
            "kaggle_private": 0.47429,
        },
        "categorical_identity": {
            "status": "rejected_on_confirmation",
            "mean_confirmation_stability_delta": 0.00006229249958655814,
        },
        "applprev_histogram": {
            "registered_standalone_status": "rejected_on_confirmation",
            "standalone_confirmation_mean_stability_delta": confirmation["comparisons"][
                "saved_champion"
            ]["mean_stability_delta"],
            "external_probe_candidate": "blend_hist_25",
            "external_probe_internal_promotion_passed": False,
            "five_fold_stability_wins": wins,
            "mean_five_fold_stability_delta": mean("blend_hist_25", "stability")
            - mean("saved_champion", "stability"),
            "mean_five_fold_auc_delta": mean("blend_hist_25", "auc")
            - mean("saved_champion", "auc"),
            "mean_five_fold_gini_delta": mean("blend_hist_25", "mean_gini")
            - mean("saved_champion", "mean_gini"),
            "all_label_fit_completed": True,
            "hidden_test_status": "not_yet_scored_in_published_evidence",
        },
        "zero_fit_ensemble_audit": {
            "grid_candidates": 47,
            "five_fold_winner_count": 2,
            "top_descriptive_weights": {
                "saved_champion": 0.75,
                "histogram": 0.25,
                "xgboost": 0.0,
                "catboost": 0.0,
            },
            "prequential_stability_delta": 0.004193182082020144,
            "prequential_auc_delta": 0.0004469927536138729,
            "prequential_gini_delta": 0.0009558351246294339,
            "new_model_fits": 0,
        },
        "denselight": {
            "status": "registered_not_executed",
            "maximum_substantive_fits": 6,
            "holdout_weeks_73_91": "prohibited_for_selection",
        },
    }


def main() -> None:
    observed = build_summary()
    expected = _load_json(SUMMARY)
    if observed != expected:
        raise ValueError("published post-release frontier summary does not reproduce")
    print(json.dumps(observed, indent=2, sort_keys=True))
    print("POST_RELEASE_FRONTIER_VERIFIED")


if __name__ == "__main__":
    main()
