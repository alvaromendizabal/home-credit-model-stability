"""The published tuning excerpt preserves the real completed study's conclusions."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pytest


def test_tuning_excerpt_agrees_with_the_historical_control():
    result = json.loads(Path("reports/model_tuning/metrics.json").read_text())
    assert result["complete"] and not result["smoke"]
    assert result["outer_holdout_touched"] is False
    assert result["completed_new_model_folds"] == 40
    assert result["source_study_sha256"] == (
        "0de34e51d0a97bd8cd4064931f81135ba3a5299c30e76b3ca1f178f0cb3569dd"
    )
    leader = max(result["rows"], key=lambda row: row["mean_fold_stability"])
    assert leader["candidate"] == result["selected_trial"] == "trial_006"
    for name in ("control", "trial_006"):
        row = next(row for row in result["rows"] if row["candidate"] == name)
        assert row["mean_fold_stability"] == pytest.approx(
            np.mean([fold[name] for fold in result["folds"]]), abs=1e-12
        )
        assert row["worst_fold_stability"] == min(fold[name] for fold in result["folds"])
    benchmark = json.loads(Path("reports/benchmark/metrics.json").read_text())
    old = [fold for fold in benchmark["folds"] if fold["model"] == "lightgbm"]
    assert [f["control"] for f in result["folds"]] == [f["stability_score"] for f in old]
    assert sum(f["delta"] > 0 for f in result["folds"]) == 3


def test_tuning_notebook_contains_executed_aggregate_evidence():
    notebook = nbformat.read("notebooks/07_model_tuning.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in cells] == [1, 2, 3, 4]
    assert all(cell.outputs for cell in cells)
    assert sum("image/svg+xml" in out.get("data", {}) for c in cells for out in c.outputs) == 2
    assert all(out.output_type != "error" for c in cells for out in c.outputs)
