from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/review_public_evidence.py"
SPEC = importlib.util.spec_from_file_location("public_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def documents() -> dict[str, dict[str, Any]]:
    return {name: json.loads((ROOT / name).read_bytes()) for name in MODULE.SOURCE_HASHES}


def test_public_review_reproduces_committed_outputs() -> None:
    payload = MODULE.serialize(MODULE.build_review(ROOT))
    assert payload == (ROOT / "reports/public_review/review.json").read_bytes()
    assert (
        MODULE.render_html(json.loads(payload)).encode()
        == (ROOT / "reports/public_review/index.html").read_bytes()
    )
    review = json.loads(payload)
    assert review["verification"]["development_rows"] == 727187
    assert review["verification"]["release_rows"] == 203345
    assert review["verification"]["weekly_stability_recomputations"] == 25
    assert review["benchmark"][0]["mean_fold_stability"] == pytest.approx(0.585188372392)
    assert review["historical_release"]["stability_components"]["stability_score"] == pytest.approx(
        0.729673794459
    )


def test_offline_cli_is_deterministic_in_isolated_python(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    command = [sys.executable, "-I", str(SCRIPT), "--output", str(output)]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    json_bytes = (output / "review.json").read_bytes()
    html_bytes = (output / "index.html").read_bytes()
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert "PUBLIC_EVIDENCE_VERIFIED" in first.stdout
    assert (output / "review.json").read_bytes() == json_bytes
    assert (output / "index.html").read_bytes() == html_bytes
    assert b'<html lang="en">' in html_bytes
    assert b"https://" not in html_bytes
    assert b"<script" not in html_bytes


def test_source_byte_change_fails_closed(tmp_path: Path) -> None:
    for relative in MODULE.SOURCE_HASHES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    modified = tmp_path / "reports/benchmark/metrics.json"
    modified.write_bytes(modified.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MODULE.build_review(tmp_path)


def test_source_symlink_cannot_escape_evidence_root(tmp_path: Path) -> None:
    relative = next(iter(MODULE.SOURCE_HASHES))
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.symlink_to(ROOT / relative)
    with pytest.raises(ValueError, match="evidence path escaped"):
        MODULE.build_review(tmp_path)


def test_official_formula_penalizes_only_negative_slope() -> None:
    falling = MODULE.stability_components([10, 11, 12], [0.7, 0.6, 0.5])
    rising = MODULE.stability_components([10, 11, 12], [0.5, 0.6, 0.7])
    assert falling["mean_gini"] == pytest.approx(0.6)
    assert falling["temporal_slope"] == pytest.approx(-0.1)
    assert falling["stability_score"] == pytest.approx(-8.2)
    assert rising["stability_score"] == pytest.approx(0.6)
    assert rising["residual_std"] == pytest.approx(0.0, abs=1e-15)


def test_formula_uses_population_residual_standard_deviation() -> None:
    result = MODULE.stability_components([0, 1, 2], [0.4, 0.7, 0.4])
    assert result["temporal_slope"] == pytest.approx(0.0)
    assert result["residual_std"] == pytest.approx(math.sqrt(0.02))
    assert result["stability_score"] == pytest.approx(0.5 - 0.5 * math.sqrt(0.02))


@pytest.mark.parametrize(
    ("weeks", "ginis"),
    [
        ([0, 0], [0.5, 0.6]),
        ([1, 0], [0.5, 0.6]),
        ([0, 1], [0.5, math.nan]),
        ([0, 1], [0.5, 1.1]),
        ([0], [0.5]),
        ([0, 1], [0.5]),
    ],
)
def test_invalid_weekly_aggregates_fail(weeks: list[int], ginis: list[float]) -> None:
    with pytest.raises(ValueError):
        MODULE.stability_components(weeks, ginis)


def test_prediction_lineage_mismatch_fails_even_with_valid_metrics() -> None:
    evidence = documents()
    evidence["reports/benchmark/metrics.json"]["models"][0]["oof_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="OOF lineage"):
        MODULE.review_documents(evidence)


def test_duplicate_week_with_unchanged_record_count_fails() -> None:
    evidence = documents()
    weekly = evidence["reports/benchmark/acceptance.json"]["weekly_metrics"]
    weekly[1] = dict(weekly[0])
    with pytest.raises(ValueError, match="weekly coverage"):
        MODULE.review_documents(evidence)


def test_population_mismatch_fails_before_publication() -> None:
    evidence = documents()
    weekly = evidence["reports/benchmark/acceptance.json"]["weekly_metrics"]
    weekly[0]["rows"] += 1
    weekly[0]["positive_rate"] = weekly[0]["positives"] / weekly[0]["rows"]
    with pytest.raises(ValueError, match="weekly row total mismatch"):
        MODULE.review_documents(evidence)


def test_missing_fold_replaced_by_duplicate_fails() -> None:
    evidence = documents()
    folds = evidence["reports/benchmark/metrics.json"]["folds"]
    folds[1] = dict(folds[0])
    with pytest.raises(ValueError, match="fold coverage"):
        MODULE.review_documents(evidence)


def test_changed_weekly_score_fails_against_recorded_components() -> None:
    evidence = documents()
    evidence["reports/benchmark/acceptance.json"]["weekly_metrics"][0]["gini"] += 0.01
    with pytest.raises(ValueError, match="aggregate mismatch"):
        MODULE.review_documents(evidence)


def test_ablation_control_must_match_the_same_benchmark() -> None:
    evidence = documents()
    evidence["reports/feature_ablation/comparison.json"]["folds"][0]["auc"] += 0.01
    with pytest.raises(ValueError, match="ablation matched control"):
        MODULE.review_documents(evidence)


def test_ablation_cannot_silently_include_the_observed_holdout() -> None:
    evidence = documents()
    evidence["reports/feature_ablation/comparison.json"]["outer_holdout_touched"] = True
    with pytest.raises(ValueError, match="ablation includes holdout"):
        MODULE.review_documents(evidence)


def test_release_requires_matching_frozen_intent() -> None:
    evidence = documents()
    evidence["reports/model_release/evaluation.json"]["intent"]["evaluation_weeks"] = [72, 91]
    with pytest.raises(ValueError, match="frozen release intent mismatch"):
        MODULE.review_documents(evidence)


def test_release_rejects_all_label_fit_as_independent_evaluation() -> None:
    evidence = documents()
    evidence["reports/model_release/evaluation.json"]["all_label_model_evaluated"] = True
    with pytest.raises(ValueError, match="all-label fit is not a holdout test"):
        MODULE.review_documents(evidence)


def test_unknown_schema_fails() -> None:
    evidence = documents()
    evidence["reports/model_release/evaluation.json"]["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported schema"):
        MODULE.review_documents(evidence)


def test_nonfinite_recorded_metric_fails() -> None:
    evidence = documents()
    evidence["reports/benchmark/metrics.json"]["models"][0]["auc"] = math.inf
    with pytest.raises(ValueError, match="finite value required"):
        MODULE.review_documents(evidence)


def test_presentation_does_not_export_private_source_metadata() -> None:
    evidence = documents()
    evidence["reports/benchmark/acceptance.json"]["private_path"] = "s3://private-training/example"
    evidence["reports/model_release/evaluation.json"]["private_config"] = {"secret_recipe": 1}
    payload = MODULE.serialize(MODULE.review_documents(evidence))
    assert b"s3://" not in payload
    assert b"private_config" not in payload
    assert b"secret_recipe" not in payload
    assert b"selection_sha256" not in payload
    assert b"encoders" not in payload


def test_json_with_duplicate_keys_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads(
            '{"schema_version": 1, "schema_version": 2}',
            object_pairs_hook=MODULE.reject_duplicate_keys,
        )
