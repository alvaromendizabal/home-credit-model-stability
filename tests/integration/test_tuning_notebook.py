"""Execute the model-tuning notebook, including a real embedded PNG, in CI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
import pytest

from home_credit.modeling.tuning_report import write_notebook, write_report
from home_credit.observability.logging import RunLogger
from home_credit.runtime.notebooks import execute_notebook


def report_state():
    comparison = json.loads(Path("reports/feature_ablation/comparison.json").read_text())
    metrics = {
        **comparison["rows"][0],
        "mean_weekly_gini": 0.694,
        "mean_brier_score": 0.031,
        "mean_temporal_slope": 0.0008,
        "mean_residual_std": 0.024,
    }
    baseline = {
        "slot": 0,
        "name": "control",
        "state": "complete",
        "value": metrics["mean_fold_stability"],
        "params": {"num_leaves": 31},
        "metrics": metrics,
        "folds": [f for f in comparison["folds"] if f["experiment"] == "control"],
    }
    return {
        "identity": {"smoke": True, "git_commit": "synthetic-integration"},
        "complete": True,
        "trials": [baseline],
        "new_trial_budget": 1,
        "outer_holdout_touched": False,
    }


def test_report_exposes_scope_metrics_and_parameters(tmp_path):
    state = report_state()
    write_report(state, tmp_path / "report.html")
    page = (tmp_path / "report.html").read_text()
    for expected in [
        "SMOKE CHECK",
        "ROC AUC",
        "Average precision",
        "Brier",
        "Log loss",
        "weeks 73-91 remain locked",
        "num_leaves",
        "plotly.js",
    ]:
        assert expected in page
    assert "cdn.plot.ly" not in page.split("<script")[0]


@pytest.mark.skipif(
    os.environ.get("HOME_CREDIT_NOTEBOOK_INTEGRATION") != "1",
    reason="Jupyter socket integration runs explicitly in GitHub CI",
)
def test_tuning_notebook_executes_static_figures_and_resumes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    state = report_state()
    source = tmp_path / "study.json"
    source.write_text(json.dumps(state))
    path = tmp_path / "07_model_tuning.ipynb"
    write_notebook(state, path)
    kwargs = {
        "dependencies": [source],
        "receipt_path": tmp_path / "receipt.json",
        "execution_root": tmp_path,
    }
    logger = RunLogger("notebook-test", tmp_path / "logs")
    assert execute_notebook(root, path, logger, **kwargs) is False
    executed = nbformat.read(path, as_version=4)
    cells = [c for c in executed.cells if c.cell_type == "code"]
    assert [c.execution_count for c in cells] == [1, 2, 3]
    assert all(c.outputs for c in cells)
    assert any("image/png" in out.get("data", {}) for out in cells[1].outputs)
    original = path.read_bytes()
    write_notebook(state, path)
    assert path.read_bytes() == original
    assert execute_notebook(root, path, logger, **kwargs) is True


def test_published_ablation_notebook_preserves_completed_execution():
    from home_credit.modeling.checkpoints import sha256_file

    path = Path("reports/feature_ablation/06_feature_ablation.ipynb")
    assert sha256_file(path) == "883b30b6241699ee46fc1d61ff0744fdc6fa063855ec83131a5ad07da0b3cab3"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert [c.execution_count for c in cells] == [1, 2, 3]
    assert all(c.outputs for c in cells)
    assert all(out.output_type != "error" for cell in cells for out in cell.outputs)
