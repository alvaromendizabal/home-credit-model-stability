"""Reject coherent but misleading rewrites of the actual completed study."""

import copy
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.feature_research_report import (
    FAMILIES,
    LABELS,
    chart_pairs,
    importance_summary,
    load_evidence,
    permutation_summary,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_feature_research_preserves_stability_tradeoff() -> None:
    evidence = load_evidence(ROOT)
    rows = {r["experiment"]: r for r in evidence["result"]["rows"]}
    original, added = rows["control"], rows["engineered"]
    assert added["mean_fold_stability"] < original["mean_fold_stability"]
    assert added["oof_auc"] > original["oof_auc"]
    assert added["oof_brier_score"] < original["oof_brier_score"]
    assert rows["wider_original"]["worst_fold_stability"] < original["worst_fold_stability"]
    assert sum(d["replayed_predictions"] for d in evidence["diagnostics"]) == 727187
    assert len(evidence["importance"]) == 4780


@pytest.mark.parametrize(
    "fault",
    [
        "holdout",
        "budget",
        "duplicate",
        "mean",
        "rejection",
        "recipe",
        "sampling",
        "sample_week",
        "additivity",
        "peer_refit",
        "verification",
        "permutation",
    ],
)
def test_semantically_invalid_republished_feature_evidence_is_rejected(tmp_path, fault) -> None:
    names = [
        "configs/feature_research_review.json",
        "configs/feature_research.json",
        "configs/feature_interpretation.json",
        "scripts/verify_feature_research.py",
        "uv.lock",
    ]
    policy = json.loads((ROOT / names[0]).read_text())
    names += [f"reports/feature_research/{name}" for name in policy["files"]]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    directory = tmp_path / "reports/feature_research"
    data = {name: json.loads((directory / name).read_text()) for name in policy["files"]}
    result, diag = data["comparison.json"], data["diagnostics_fold_1.json"]
    if fault == "holdout":
        result["outer_holdout_touched"] = True
    elif fault == "budget":
        result["new_model_fits"] = 19
    elif fault == "duplicate":
        result["rows"][1] = copy.deepcopy(result["rows"][0])
    elif fault == "mean":
        result["rows"][0]["mean_fold_stability"] += 0.001
    elif fault == "rejection":
        data["screen.json"]["rejection_counts"]["exact_duplicate"] -= 1
    elif fault == "recipe":
        result["fit_records"][0]["artifacts"]["features"]["sha256"] = "0" * 64
    elif fault == "sampling":
        diag["shap_sampling"] = "sorted_prefix"
    elif fault == "sample_week":
        diag["shap_week_counts"][0]["WEEK_NUM"] = 73
    elif fault == "additivity":
        diag["maximum_additivity_absolute_error"] = 0.01
    elif fault == "peer_refit":
        diag["peer_references_refitted"] = True
    elif fault == "verification":
        data["verification.json"]["predictions_verified"] -= 1
    else:
        diag["permutation"][1] = copy.deepcopy(diag["permutation"][0])

    def save(name):
        path = directory / name
        path.write_text(json.dumps(data[name]))
        return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    # Repin every dependent identity: failures must come from semantics, not a stale hash.
    screen_pin = save("screen.json")
    result["screen"]["screen"].update(screen_pin)
    report_pin = save("comparison.json")
    data["study.json"]["trials"] = result["fit_records"]
    data["study.json"]["stages"]["screen"] = result["screen"]
    data["study.json"]["stages"]["report"].update(report_pin)
    ledger_pin = save("study.json")
    data["verification.json"]["report_sha256"] = report_pin["sha256"]
    save("verification.json")
    interpretation = data["interpretation.json"]
    interpretation["identity"]["training_ledger_sha256"] = ledger_pin["sha256"]
    for row in interpretation["folds"]:
        row["diagnostics"].update(save(row["diagnostics"]["path"]))
    save("interpretation.json")
    policy["files"] = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)}
        for name in policy["files"]
    }
    (tmp_path / "configs/feature_research_review.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_evidence(tmp_path)


def test_feature_chart_values_match_static_interactive_and_evidence() -> None:
    evidence = load_evidence(ROOT)
    pairs = chart_pairs(evidence)
    try:
        family, stability, delta, permutation, shap = list(pairs.values())
        for i, key in enumerate(("candidate_families", "retained_families")):
            expected = [evidence["result"][key][f] for f in FAMILIES]
            assert list(family[1].data[i].x) == expected
            np.testing.assert_array_equal(family[0].axes[0].containers[i].datavalues, expected)
        rows = {r["experiment"]: r for r in evidence["result"]["rows"]}
        for i, key in enumerate(("mean_fold_stability", "worst_fold_stability")):
            expected = [rows[e][key] for e in LABELS]
            assert list(stability[1].data[i].x) == expected
            np.testing.assert_array_equal(stability[0].axes[0].lines[i].get_xdata(), expected)
        folds = {(r["experiment"], r["fold"]): r for r in evidence["result"]["folds"]}
        expected = [
            [
                folds[e, f]["stability_score"] - folds["control", f]["stability_score"]
                for f in range(1, 6)
            ]
            for e in LABELS
            if e != "control"
        ]
        np.testing.assert_array_equal(delta[1].data[0].z, expected)
        np.testing.assert_array_equal(delta[0].axes[0].images[0].get_array(), expected)
        summary = permutation_summary(evidence)
        np.testing.assert_array_equal(permutation[1].data[0].x, summary["mean"])
        np.testing.assert_array_equal(permutation[0].axes[0].lines[0].get_xdata(), summary["mean"])
        bounds = permutation[0].axes[0].collections[0].get_segments()
        np.testing.assert_allclose([b[0, 0] for b in bounds], summary["minimum"], atol=1e-16)
        np.testing.assert_allclose([b[1, 0] for b in bounds], summary["maximum"], atol=1e-16)
        np.testing.assert_array_equal(
            permutation[1].data[0].error_x.array, summary["maximum"] - summary["mean"]
        )
        expected = importance_summary(evidence).head(12)["mean_abs_shap"]
        np.testing.assert_array_equal(shap[1].data[0].x, expected)
        np.testing.assert_array_equal(shap[0].axes[0].containers[0].datavalues, expected)
    finally:
        for static, _ in pairs.values():
            plt.close(static)
