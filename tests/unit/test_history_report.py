"""Publication contracts and chart fidelity for the actual raw-history study."""

import copy
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.history_report import LABELS, chart_pairs, load_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_actual_history_study_has_complete_aligned_evidence():
    evidence = load_evidence(ROOT)
    result = evidence["result"]
    assert result["additional_candidates"] == 524
    assert result["additional_retained"] <= 96
    assert len(result["fit_records"]) == 10
    assert result["aligned_oof_cases"] == 727187
    assert not result["release_changed"] and not result["outer_holdout_touched"]
    audits = evidence["manifest"]["source_audits"]
    assert len(audits) == 14 and sum(len(a["date_fields"]) for a in audits) == 33
    assert sum(not a["numeric_columns"] for a in audits) == 2
    assert evidence["verification"]["predictions_replayed"] == 1454374


@pytest.mark.parametrize(
    "fault",
    [
        "holdout",
        "promotion",
        "mean",
        "rejection",
        "screen",
        "duplicate",
        "date",
        "verification",
        "width",
    ],
)
def test_rehashed_but_invalid_history_evidence_is_rejected(tmp_path, fault):
    policy = json.loads((ROOT / "configs/history_research_review.json").read_text())
    paths = [
        "configs/history_research_review.json",
        "configs/history_research.json",
        "configs/model_benchmark.json",
        "configs/validation_protocol.json",
        "scripts/verify_history_research.py",
        "uv.lock",
    ]
    paths += [f"reports/history_research/{n}" for n in policy["files"]]
    for name in paths:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    directory = tmp_path / "reports/history_research"
    data = {name: json.loads((directory / name).read_text()) for name in policy["files"]}
    result = data["comparison.json"]
    if fault == "holdout":
        result["outer_holdout_touched"] = True
    elif fault == "promotion":
        result["release_changed"] = True
    elif fault == "mean":
        result["rows"][0]["mean_fold_stability"] += 0.01
    elif fault == "rejection":
        data["screen.json"]["rejected"] -= 1
    elif fault == "screen":
        data["screen.json"]["screen_weeks"]["validation"][1] = 40
    elif fault == "duplicate":
        result["fit_records"][1] = copy.deepcopy(result["fit_records"][0])
    elif fault == "date":
        data["history_manifest.json"]["source_audits"][0]["chronology_admitted"] = True
    elif fault == "verification":
        data["verification.json"]["predictions_replayed"] -= 1
    else:
        result["fit_records"][0]["features"] -= 1

    def save(name):
        path = directory / name
        path.write_text(json.dumps(data[name]))
        return {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    for name, member in (
        ("screen.json", result["screen"]),
        ("history_manifest.json", result["history_manifest"]),
    ):
        member.update(save(name))
    for trial in result["fit_records"]:
        trial["history_view_sha256"] = result["history_manifest"]["sha256"]
    result_pin = save("comparison.json")
    data["verification.json"]["comparison_sha256"] = result_pin["sha256"]
    data["study.json"]["trials"] = result["fit_records"]
    data["study.json"]["stages"]["report"].update(result_pin)
    data["study.json"]["stages"]["screen"] = result["screen"]
    data["study.json"]["stages"]["history_manifest"] = result["history_manifest"]
    policy["files"] = {name: save(name) for name in policy["files"]}
    (tmp_path / "configs/history_research_review.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_evidence(tmp_path)


def test_history_static_and_plotly_views_use_exact_evidence():
    evidence = load_evidence(ROOT)
    pairs = chart_pairs(evidence)
    try:
        rows = {r["experiment"]: r for r in evidence["result"]["rows"]}
        static, chart = next(iter(pairs.values()))
        for i, key in enumerate(("mean_fold_stability", "worst_fold_stability")):
            expected = [rows[n][key] for n in LABELS]
            np.testing.assert_allclose(static.axes[0].collections[i].get_offsets()[:, 0], expected)
            np.testing.assert_allclose(chart.data[i].x, expected)
        static, chart = list(pairs.values())[1]
        np.testing.assert_array_equal(static.axes[0].images[0].get_array(), chart.data[0].z)
    finally:
        for static, _ in pairs.values():
            plt.close(static)
