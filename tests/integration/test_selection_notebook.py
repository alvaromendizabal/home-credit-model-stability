"""Execute the new report with explicit synthetic data, then verify exact reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from home_credit.modeling.selection import align_predictions, evaluate_candidate, fixed_candidates
from home_credit.modeling.selection_report import write_selection_notebook, write_selection_report
from home_credit.modeling.selection_workflow import selection_result
from home_credit.observability.logging import RunLogger
from home_credit.runtime.notebooks import execute_notebook


def synthetic_result():
    names = ["tuned_lightgbm", "lightgbm", "xgboost", "catboost"]
    weeks = np.repeat(np.arange(33, 43), 20)
    target = np.tile([0] * 15 + [1] * 5, 10)
    rng = np.random.default_rng(71)
    frames = {
        name: pd.DataFrame(
            {
                "case_id": np.arange(200),
                "target": target,
                "WEEK_NUM": weeks,
                "fold": (weeks - 33) // 2 + 1,
                "prediction": np.clip(0.15 + 0.3 * target + rng.normal(0, 0.2, 200), 0, 1),
            }
        )
        for name in names
    }
    windows = [
        {
            "fold": f,
            "train_week_min": 0,
            "train_week_max": 30 + 2 * f,
            "validation_week_min": 31 + 2 * f,
            "validation_week_max": 32 + 2 * f,
        }
        for f in range(1, 6)
    ]
    data = align_predictions(frames, windows, expected_rows=200)
    records = [
        evaluate_candidate(data, name, weights)
        for name, weights in fixed_candidates(names, names[0]).items()
    ]
    plan = {
        "leader": names[0],
        "selection_scope": "SYNTHETIC INTEGRATION TEST",
        "prequential_scope": "Not unbiased nested validation",
        "study": {"sha256": "synthetic"},
    }
    return selection_result(data, {"identity": {"smoke": True}, "trials": records}, plan)


@pytest.mark.skipif(
    os.environ.get("HOME_CREDIT_NOTEBOOK_INTEGRATION") != "1",
    reason="Jupyter integration runs explicitly in GitHub CI",
)
def test_selection_notebook_executes_static_figures_and_resumes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = synthetic_result()
    source = tmp_path / "selection.json"
    source.write_text(json.dumps(result, allow_nan=False))
    write_selection_report(result, tmp_path)
    path = tmp_path / "08_model_selection.ipynb"
    write_selection_notebook(path)
    kwargs = {
        "dependencies": [source],
        "receipt_path": tmp_path / "receipt.json",
        "execution_root": tmp_path,
    }
    logger = RunLogger("selection-notebook-test", tmp_path / "logs")
    assert execute_notebook(root, path, logger, **kwargs) is False
    executed = nbformat.read(path, as_version=4)
    cells = [c for c in executed.cells if c.cell_type == "code"]
    assert [c.execution_count for c in cells] == [1, 2, 3, 4]
    assert all(c.outputs for c in cells)
    assert sum("image/png" in output.get("data", {}) for c in cells for output in c.outputs) == 3
    assert not any(output.output_type == "error" for c in cells for output in c.outputs)
    original = path.read_bytes()
    write_selection_notebook(path)
    assert path.read_bytes() == original
    assert execute_notebook(root, path, logger, **kwargs) is True
