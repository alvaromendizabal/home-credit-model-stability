"""Execute the new notebook through Jupyter on the CI runner."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import nbformat
import polars as pl
import pytest

from home_credit.modeling.ablation import compare_predictions
from home_credit.observability.logging import RunLogger
from home_credit.runtime.notebooks import execute_notebook


@pytest.mark.skipif(
    os.environ.get("HOME_CREDIT_NOTEBOOK_INTEGRATION") != "1",
    reason="Jupyter socket integration runs explicitly in GitHub CI",
)
def test_ablation_notebook_executes_and_resumes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ablation_cli", root / "scripts/run_feature_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pl.DataFrame(
        {
            "case_id": list(range(16)),
            "WEEK_NUM": [33] * 4 + [34] * 4 + [35] * 4 + [36] * 4,
            "target": [0, 1, 0, 1] * 4,
            "prediction": [0.1, 0.9, 0.2, 0.8] * 4,
        }
    )
    result = compare_predictions(
        {"control": frame, "same": frame.reverse()},
        [{"fold": 1, "validation_week_min": 33, "validation_week_max": 36}],
    )
    result["smoke"] = True
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(result))
    notebook = tmp_path / "06_feature_ablation.ipynb"
    module.write_notebook(result, notebook)
    options = dict(
        dependencies=[comparison], receipt_path=tmp_path / "receipt.json", execution_root=tmp_path
    )
    logger = RunLogger("ablation-notebook-test", tmp_path / "logs")
    assert execute_notebook(root, notebook, logger, **options) is False
    executed = nbformat.read(notebook, as_version=4)
    assert [c.execution_count for c in executed.cells if c.cell_type == "code"] == [1, 2, 3]
    assert all(c.outputs for c in executed.cells if c.cell_type == "code")
    original = notebook.read_bytes()
    module.write_notebook(result, notebook)
    assert notebook.read_bytes() == original
    assert execute_notebook(root, notebook, logger, **options) is True
