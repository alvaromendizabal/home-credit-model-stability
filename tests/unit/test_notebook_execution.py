from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
import pytest

from home_credit.observability.logging import RunLogger
from home_credit.runtime import notebooks

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_reuse_invalidation_and_failed_execution_preserve_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        "uv.lock",
        "configs/benchmark_review.json",
        "configs/validation_protocol.json",
        "reports/benchmark/acceptance.json",
        "reports/benchmark/metrics.json",
        "src/home_credit/modeling/review.py",
        "src/home_credit/modeling/report.py",
        "src/home_credit/metrics/classification.py",
        "src/home_credit/runtime/notebooks.py",
    ]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, target)
    path = tmp_path / "review.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(42)", id="example")]), path
    )
    calls = []

    class Client:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self, **kwargs):
            calls.append(1)
            self.notebook.cells[0].execution_count = 1
            self.notebook.cells[0].outputs = [
                nbformat.v4.new_output("stream", name="stdout", text="42\n")
            ]

    monkeypatch.setattr(notebooks, "NotebookClient", Client)
    monkeypatch.setattr(notebooks, "install", lambda **kwargs: None)
    monkeypatch.setattr(notebooks, "KernelManager", lambda **kwargs: None)
    logger = RunLogger("test-notebook", tmp_path / "logs")
    assert notebooks.execute_notebook(tmp_path, path, logger) is False
    assert notebooks.execute_notebook(tmp_path, path, logger) is True
    assert len(calls) == 1
    (tmp_path / "uv.lock").write_text("changed dependencies")
    assert notebooks.execute_notebook(tmp_path, path, logger) is False
    assert len(calls) == 2
    notebook_before = path.read_bytes()
    receipt = tmp_path / "artifacts/benchmark_review/notebook.json"
    receipt_before = receipt.read_bytes()

    def fail(self, **kwargs):
        raise RuntimeError("cell failure")

    monkeypatch.setattr(Client, "execute", fail)
    with pytest.raises(RuntimeError, match="cell failure"):
        notebooks.execute_notebook(tmp_path, path, logger, force=True)
    assert path.read_bytes() == notebook_before
    assert receipt.read_bytes() == receipt_before
    assert notebooks.execute_notebook(tmp_path, path, logger) is True
