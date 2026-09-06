"""Execute a review notebook with observable progress and verified whole-run reuse."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import nbformat
from ipykernel.kernelspec import install
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

from home_credit.modeling.acceptance import read_json
from home_credit.modeling.checkpoints import atomic_write, sha256_file
from home_credit.observability.logging import RunLogger


def notebook_identity(notebook: Any, dependencies: list[Path]) -> str:
    """Fingerprint source and dependencies; execution outputs never invalidate source."""
    payload = {
        "cells": [{k: cell[k] for k in ("cell_type", "id", "source")} for cell in notebook.cells],
        "dependencies": [sha256_file(path) for path in dependencies],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execute_notebook(
    root: Path,
    notebook_path: Path,
    logger: RunLogger,
    *,
    force: bool = False,
    dependencies: list[Path] | None = None,
    receipt_path: Path | None = None,
    execution_root: Path | None = None,
) -> bool:
    """Return True on verified reuse, otherwise atomically publish a successful run.

    Failed cells never replace a previously successful notebook or its receipt.
    A failed execution restarts this inexpensive notebook; expensive model fits
    and downloads are separate durable stages and are not invoked here.
    """
    notebook = nbformat.read(notebook_path, as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)
    dependencies = (
        dependencies
        if dependencies is not None
        else [
            root / p
            for p in (
                "uv.lock",
                "configs/benchmark_review.json",
                "configs/validation_protocol.json",
                "reports/benchmark/acceptance.json",
                "reports/benchmark/metrics.json",
                "src/home_credit/modeling/review.py",
                "src/home_credit/modeling/report.py",
                "src/home_credit/metrics/classification.py",
                "src/home_credit/runtime/notebooks.py",
            )
        ]
    )
    identity = notebook_identity(notebook, dependencies)
    receipt_path = receipt_path or root / "artifacts/benchmark_review/notebook.json"
    if receipt_path.is_file() and not force:
        try:
            receipt = read_json(receipt_path)
        except (ValueError, OSError):
            receipt = {}
        if receipt.get("identity") == identity and receipt.get("sha256") == sha256_file(
            notebook_path
        ):
            logger.event("notebook_reused", path=notebook_path, sha256=receipt["sha256"])
            return True
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop("execution", None)
    runtime = root / "artifacts/runtime/jupyter"
    runtime.mkdir(parents=True, exist_ok=True)
    install(prefix=str(runtime), kernel_name="home-credit", display_name="Python (Home Credit)")
    manager = KernelManager(
        kernel_name="home-credit",
        transport="ipc",
        ip=str(root / "artifacts/runtime" / f"k-{os.getpid()}"),
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime / "share/jupyter/kernels")]),
        connection_file=str(runtime / f"kernel-{os.getpid()}.json"),
    )
    starts: dict[int, float] = {}
    total = sum(c.cell_type == "code" for c in notebook.cells)
    completed = 0

    def on_execute(cell: Any, cell_index: int, **kwargs: Any) -> None:
        starts[cell_index] = time.monotonic()
        logger.event("notebook_cell_started", cell=cell_index, completed=completed, total=total)

    def on_executed(cell: Any, cell_index: int, execute_reply: Any, **kwargs: Any) -> None:
        nonlocal completed
        status = execute_reply["content"]["status"]
        completed += status == "ok"
        logger.event(
            "notebook_cell_completed" if status == "ok" else "notebook_cell_failed",
            cell=cell_index,
            completed=completed,
            total=total,
            elapsed_seconds=round(time.monotonic() - starts[cell_index], 3),
        )

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=120,
        startup_timeout=60,
        record_timing=False,
        allow_errors=False,
        store_widget_state=False,
        resources={"metadata": {"path": str(execution_root or root)}},
        on_cell_execute=on_execute,
        on_cell_executed=on_executed,
    )
    client.execute(cleanup_kc=True)
    nbformat.validate(notebook)
    atomic_write(notebook_path, nbformat.writes(notebook).encode())  # type: ignore[no-untyped-call]
    atomic_write(
        receipt_path,
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "sha256": sha256_file(notebook_path),
                    "completed_code_cells": completed,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    logger.event("notebook_published", path=notebook_path, completed=completed, total=total)
    return False
