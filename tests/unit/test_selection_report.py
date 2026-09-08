"""Publication guards, deterministic HTML/SVG, and interactive/static chart parity."""

from __future__ import annotations

import copy
import json
import runpy
import shutil
from pathlib import Path

import matplotlib
import pytest
from test_modeling_selection import make_selection_result

from home_credit.modeling.selection_report import (
    candidate_label,
    selection_plotly_figures,
    validate_selection_evidence,
    write_selection_report,
)


@pytest.mark.parametrize(
    "fault",
    ["holdout", "synthetic_as_real", "weights", "winner", "fold", "delta", "rows", "week"],
)
def test_publication_rejects_inconsistent_evidence(fault):
    result = make_selection_result()
    if fault == "holdout":
        result["outer_holdout_touched"] = True
    elif fault == "synthetic_as_real":
        result["smoke"] = False
    elif fault == "weights":
        result["selected_weights"] = {"tuned_lightgbm": 0.3}
    elif fault == "winner":
        result["selected_candidate"] = "not-a-candidate"
    elif fault == "fold":
        result["folds"][0]["stability_score"] += 0.1
    elif fault == "delta":
        result["rows"][0]["delta_vs_tuned_lightgbm"] += 0.1
    elif fault == "rows":
        result["diagnostics"]["reliability"][0]["rows"] += 1
    else:
        result["diagnostics"]["weekly"][0]["week"] = 73
    with pytest.raises(ValueError):
        validate_selection_evidence(result)


def test_plotly_uses_same_evidence_and_hover_support():
    result = make_selection_result()
    original = copy.deepcopy(result)
    figures = selection_plotly_figures(result)
    assert set(figures) == {"candidates", "weekly_gini", "reliability"}
    assert list(figures["candidates"].data[0].x) == [
        row["mean_fold_stability"] for row in reversed(result["rows"])
    ]
    weekly = result["diagnostics"]["weekly"]
    assert list(figures["weekly_gini"].data[0].y) == [row["gini"] for row in weekly]
    assert [row[1] for row in figures["weekly_gini"].data[0].customdata] == [
        row["rows"] for row in weekly
    ]
    assert result == original
    for figure in figures.values():
        json.loads(figure.to_json())
    assert candidate_label("tuned_lightgbm_90_xgboost") == "Tuned LGBM 90% + XGBoost"
    assert candidate_label("unrecognized") == "unrecognized"


def test_offline_report_is_deterministic_and_has_static_fallbacks(tmp_path):
    result = make_selection_result()
    before = matplotlib.rcParams["svg.hashsalt"]
    paths = write_selection_report(result, tmp_path)
    first = {path.name: path.read_bytes() for path in paths}
    write_selection_report(result, tmp_path)
    assert {path.name: path.read_bytes() for path in paths} == first
    assert matplotlib.rcParams["svg.hashsalt"] == before
    page = (tmp_path / "report.html").read_text()
    assert "<script src=" not in page
    assert "<noscript>" in page and "<svg" in page
    for name in ("candidates", "weekly_gini", "reliability"):
        assert f'id="selection-{name}"' in page
    assert "case_id" not in json.dumps(result)


def test_real_publication_matches_pinned_scoring_lineage():
    root = Path(__file__).resolve().parents[2]
    api = runpy.run_path(str(root / "scripts/review_model_selection.py"))
    source, result = api["load_evidence"](root)
    assert source.stat().st_size == 40389
    assert result["rows_evaluated"] == 727187
    assert len(result["folds"]) == 75
    assert result["selected_candidate"] == "tuned_lightgbm_90_lightgbm"


@pytest.mark.parametrize("fault", ["evidence_bytes", "scoring_lineage"])
def test_real_publication_rejects_changed_bytes_or_lineage(tmp_path, fault):
    root = Path(__file__).resolve().parents[2]
    api = runpy.run_path(str(root / "scripts/review_model_selection.py"))
    source, result = api["load_evidence"](root)
    names = {
        *result["identity"]["inputs"],
        "configs/model_selection_review.json",
        source.relative_to(root).as_posix(),
    }
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    relative = source.relative_to(root) if fault == "evidence_bytes" else Path("uv.lock")
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n")
    expected = "evidence digest changed" if fault == "evidence_bytes" else "lineage changed"
    with pytest.raises(ValueError, match=expected):
        api["load_evidence"](tmp_path)
