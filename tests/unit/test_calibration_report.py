"""Reconcile real research evidence and reject misleading published results."""

import copy
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from home_credit.modeling.calibration_report import LABELS, chart_pairs, load_evidence
from home_credit.modeling.checkpoints import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_real_calibration_evidence_preserves_negative_result() -> None:
    evidence = load_evidence(ROOT)
    rows = {r["method"]: r for r in evidence["rows"]}
    reference = rows["uncalibrated"]
    assert reference["evaluation_rows"] == 544611
    for method in ("sigmoid", "isotonic"):
        assert rows[method]["pooled_brier_score"] > reference["pooled_brier_score"]
        assert rows[method]["pooled_log_loss"] > reference["pooled_log_loss"]
    assert rows["isotonic"]["mean_fold_stability"] < reference["mean_fold_stability"]
    assert rows["sigmoid"]["mean_fold_stability"] == reference["mean_fold_stability"]


@pytest.mark.parametrize(
    "fault", ["holdout", "budget", "window", "mean", "support", "duplicate", "parameters"]
)
def test_semantically_invalid_republished_calibration_is_rejected(tmp_path, fault) -> None:
    for name in (
        "configs/calibration_review.json",
        "configs/calibration_research.json",
        "reports/calibration/comparison.json",
        "reports/calibration/verification.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    path = tmp_path / "reports/calibration/comparison.json"
    report = json.loads(path.read_text())
    if fault == "holdout":
        report["holdout_used"] = True
    elif fault == "budget":
        report["new_base_model_fits"] = 1
    elif fault == "window":
        report["folds"][0]["fit_week_max"] = report["folds"][0]["evaluation_week_min"]
    elif fault == "mean":
        report["rows"][0]["mean_fold_stability"] += 0.001
    elif fault == "support":
        report["reliability"]["uncalibrated"][0]["rows"] += 1
    elif fault == "duplicate":
        report["rows"][1] = copy.deepcopy(report["rows"][0])
    else:
        next(r for r in report["folds"] if r["method"] == "sigmoid")["model"]["coefficient"] = -1
    path.write_text(json.dumps(report))
    policy_path = tmp_path / "configs/calibration_review.json"
    policy = json.loads(policy_path.read_text())
    policy.update(sha256=sha256_file(path), bytes=path.stat().st_size)
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_evidence(tmp_path)


def test_calibration_plotly_and_static_values_match_exactly() -> None:
    evidence = load_evidence(ROOT)
    pairs = chart_pairs(evidence)
    try:
        for metric, title in (("brier_score", "Brier score"), ("log_loss", "Log loss")):
            static, interactive = pairs[title]
            reference = {
                r["fold"]: r["metrics"][metric]
                for r in evidence["folds"]
                if r["method"] == "uncalibrated"
            }
            for index, method in enumerate(LABELS):
                rows = sorted(
                    (r for r in evidence["folds"] if r["method"] == method), key=lambda r: r["fold"]
                )
                expected = [r["metrics"][metric] - reference[r["fold"]] for r in rows]
                assert list(interactive.data[index].y) == expected
                np.testing.assert_array_equal(static.axes[0].lines[index].get_ydata(), expected)
        static, interactive = pairs["Reliability"]
        for index, method in enumerate(LABELS, 1):
            expected = [b["observed_default_rate"] for b in evidence["reliability"][method]]
            assert list(interactive.data[index].y) == expected
            np.testing.assert_array_equal(static.axes[0].lines[index].get_ydata(), expected)
    finally:
        for static, _ in pairs.values():
            plt.close(static)
