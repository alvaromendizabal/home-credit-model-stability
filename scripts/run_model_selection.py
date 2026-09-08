#!/usr/bin/env python3
"""Compare 15 fixed blends of saved development predictions; train no new model."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.selection import require
from home_credit.modeling.selection_report import publish_selection_report
from home_credit.modeling.selection_workflow import execute_selection
from home_credit.observability.logging import RunLogger


class SelectionLogger(RunLogger):
    """Include invocation elapsed time in every event, including heartbeats."""

    def __init__(self, name: str, directory: Path) -> None:
        self.started = time.monotonic()
        super().__init__(name, directory)

    def event(self, event: str, **fields: Any) -> None:
        fields.setdefault("total_elapsed_seconds", round(time.monotonic() - self.started, 3))
        super().event(event, **fields)


def terminate(signum: int, frame: Any) -> None:
    """Unwind locks without discarding already committed candidate receipts."""
    raise InterruptedError(f"received signal {signum}; completed candidates remain resumable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(Path.cwd() == root, "run from the repository root")
    require(sys.version_info[:2] == (3, 12), "use the locked Python 3.12 environment")
    signal.signal(signal.SIGTERM, terminate)
    plan = json.loads((root / "configs/model_selection.json").read_text())
    # Notebook outputs may be locally changed. Training/selection sources may not.
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "src",
            "scripts",
            "configs",
            "uv.lock",
            "pyproject.toml",
        ],
        text=True,
    ).strip()
    require(not dirty, "commit source/configuration changes before model selection")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lock = tomllib.loads((root / "uv.lock").read_text())
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        versions[name] = importlib.metadata.version(name)
        require(
            {p["version"] for p in lock["package"] if p["name"] == name} == {versions[name]},
            f"active {name} differs from uv.lock",
        )
    dependencies = [
        "configs/model_selection.json",
        "configs/validation_protocol.json",
        "uv.lock",
        "src/home_credit/modeling/selection.py",
        "src/home_credit/modeling/selection_workflow.py",
        "src/home_credit/metrics/classification.py",
        "src/home_credit/metrics/stability.py",
        "src/home_credit/modeling/experiment_store.py",
        "src/home_credit/modeling/checkpoints.py",
        "scripts/run_model_selection.py",
    ]
    identity = {
        "inputs": {name: sha256_file(root / name) for name in dependencies},
        "versions": versions,
        "schema_version": 1,
    }
    run_key = sha256_bytes(canonical_json_bytes(identity))
    work = root / "artifacts/model_selection" / run_key
    work.mkdir(parents=True, exist_ok=True)
    logger = SelectionLogger("model-selection", work / "logs")
    client = boto3.client(
        "s3",
        region_name=plan["region"],
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 3},
            connect_timeout=5,
            read_timeout=30,
        ),
    )
    store = ExperimentStore(
        client, args.bucket, f"home-credit-model-stability/model-selection/{run_key}", work, logger
    )
    with (work / "run.lock").open("a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this selection study already has a local worker") from exc
        try:
            with WriterLease(store) as lease:
                logger.event(
                    "model_selection_started",
                    git_commit=commit,
                    candidates=15,
                    new_model_fits=0,
                    outer_holdout_touched=False,
                    run_key=run_key,
                )
                state = execute_selection(root, plan, identity, store, lease, logger)
                publish_selection_report(root, store, lease, logger)
                lease.check()
                state["complete"] = True
                state["revision"] += 1
                state["reporting_git_commit"] = commit
                store.commit(state)
                logger.event(
                    "model_selection_completed",
                    selected_candidate=state["selected_candidate"],
                    notebook=root / "notebooks/08_model_selection.ipynb",
                )
                store.publish(logger.jsonl_path, "logs")
        except Exception as exc:
            logger.event("model_selection_failed", error_type=type(exc).__name__, error=str(exc))
            try:
                store.publish(logger.jsonl_path, "logs")
            except Exception as upload_error:
                logger.event(
                    "log_upload_pending",
                    error_type=type(upload_error).__name__,
                    local_path=logger.jsonl_path,
                )
            raise
    print("MODEL_SELECTION_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
