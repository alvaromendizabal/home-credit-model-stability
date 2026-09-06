#!/usr/bin/env python3
"""Run the predeclared LightGBM ablation suite with durable model-fold resume."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import polars as pl

from home_credit.modeling.ablation import compare_predictions, write_report
from home_credit.modeling.acceptance import read_json, require
from home_credit.modeling.checkpoints import atomic_write, sha256_file, validate_benchmark_state
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="One capped fold per condition; separate outputs."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(Path.cwd() == root, "run from the repository root")
    started = time.monotonic()
    mode = "smoke" if args.smoke else "full"
    work = root / "artifacts/feature_ablation" / mode
    work.mkdir(parents=True, exist_ok=True)
    with (work / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a {mode} ablation suite is already running") from exc
        logger = RunLogger("feature-ablation", root / "logs")
        try:
            run_suite(root, work, args.bucket, args.smoke, logger)
        except Exception as exc:
            logger.event(
                "ablation_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                total_elapsed_seconds=round(time.monotonic() - started, 3),
            )
            raise
        logger.event(
            "ablation_completed",
            mode=mode,
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        print(
            "FEATURE_ABLATION_SMOKE_COMPLETED" if args.smoke else "FEATURE_ABLATION_COMPLETED",
            flush=True,
        )
    return 0


def run_suite(root: Path, work: Path, bucket: str, smoke: bool, logger: RunLogger) -> None:
    """Run each condition sequentially; persist checkpoints before advancing."""
    plan = read_json(root / "configs/feature_ablation.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "tracked source changes must be committed before training",
    )
    mode = "smoke" if smoke else "full"
    prefix = f"home-credit-model-stability/feature-ablation/{commit}/{mode}"
    identity_path = work / "source.json"
    identity = {
        "git_commit": commit,
        "plan_sha256": sha256_file(root / "configs/feature_ablation.json"),
        "smoke": smoke,
    }
    if identity_path.exists():
        require(
            read_json(identity_path) == identity,
            "ablation source changed; resume the original commit",
        )
    else:
        atomic_write(identity_path, (json.dumps(identity, sort_keys=True) + "\n").encode())
    names = plan["experiments"]
    logger.event(
        "ablation_started",
        conditions=len(names),
        total_model_folds=len(names) * (1 if smoke else 5),
        git_commit=commit,
        smoke=smoke,
    )
    for index, name in enumerate(names, 1):
        config = root / f"configs/ablations/{name}.json"
        command = [
            sys.executable,
            "scripts/run_model_benchmark_supervisor.py",
            "--feature-s3-uri",
            f"s3://{bucket}/home-credit-model-stability/feature-snapshots/{plan['feature_manifest_sha256']}",
            "--expected-feature-manifest-sha256",
            plan["feature_manifest_sha256"],
            "--expected-protocol-sha256",
            plan["protocol_sha256"],
            "--config",
            str(config),
            "--expected-config-sha256",
            sha256_file(config),
            "--checkpoint-s3-uri",
            f"s3://{bucket}/{prefix}/{name}/checkpoints",
            "--final-s3-uri",
            f"s3://{bucket}/{prefix}/{name}/results",
            "--feature-cache-dir",
            str(root / "artifacts/feature_cache"),
            "--work-dir",
            str(work / name),
            "--status-log",
            str(root / f"logs/ablation-{mode}-{name}.jsonl"),
            "--child-heartbeat-seconds",
            "15",
            "--max-processes",
            "6",
        ]
        if smoke:
            command.append("--smoke")
        logger.event("ablation_condition_started", condition=name, index=index, total=len(names))
        with StageTimer(logger, f"condition_{name}", heartbeat_seconds=15):
            subprocess.run(command, check=True)
        logger.event(
            "ablation_condition_completed", condition=name, completed=index, total=len(names)
        )
    frames = {}
    provenance = {}
    for name in names:
        output = work / name
        config = root / f"configs/ablations/{name}.json"
        count = validate_benchmark_state(
            output,
            git_commit=commit,
            feature_manifest_sha256=plan["feature_manifest_sha256"],
            validation_protocol_sha256=plan["protocol_sha256"],
            benchmark_config_sha256=sha256_file(config),
            smoke=smoke,
        )
        require(count == (1 if smoke else 5), "incomplete ablation checkpoints")
        state = read_json(output / "benchmark_state.json")
        # Rebuild comparison inputs from the hash-verified fold predictions.
        frames[name] = pl.concat(
            [pl.read_parquet(output / row["prediction_path"]) for row in state["folds"].values()]
        )
        provenance[name] = {
            "config_sha256": sha256_file(config),
            "state_sha256": sha256_file(output / "benchmark_state.json"),
        }
    folds = read_json(root / "configs/validation_protocol.json")["inner_temporal_cv"]["folds"]
    with StageTimer(logger, "ablation_comparison", heartbeat_seconds=15):
        result = compare_predictions(frames, folds[:1] if smoke else folds)
        result.update({"smoke": smoke, "git_commit": commit, "provenance": provenance})
        report_dir = work / "report"
        atomic_write(
            report_dir / "comparison.json",
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        )
        write_report(result, report_dir / "report.html")
        notebook_path = report_dir / "06_feature_ablation.ipynb"
        write_notebook(result, notebook_path)
        execute_notebook(
            root,
            notebook_path,
            logger,
            dependencies=[report_dir / "comparison.json", root / "uv.lock", Path(__file__)],
            receipt_path=work / "notebook_receipt.json",
            execution_root=report_dir,
        )
    # Models/predictions have already been saved by the supervisor. Save the derived report too.
    s3 = boto3.client("s3", region_name="us-west-2")
    with StageTimer(logger, "ablation_report_upload", heartbeat_seconds=15):
        for path in sorted(report_dir.iterdir()):
            digest = sha256_file(path)
            key = f"{prefix}/reports/{digest}/{path.name}"
            s3.upload_file(
                str(path),
                bucket,
                key,
                ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": digest}},
            )
            head = s3.head_object(Bucket=bucket, Key=key)
            require(
                head["ContentLength"] == path.stat().st_size
                and head.get("Metadata", {}).get("sha256") == digest,
                "report upload verification failed",
            )
            logger.event("ablation_report_saved", path=path, s3_key=key)


def write_notebook(result: dict[str, Any], destination: Path) -> None:
    """Create an aggregate-only notebook beside its verified comparison JSON."""
    import nbformat

    notebook = nbformat.v4.new_notebook(  # type: ignore[no-untyped-call]
        cells=[
            nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
                "# Home Credit: Feature ablation\n\n"
                "Development comparison. The final holdout remains locked. "
                f"Smoke run: {result['smoke']}."
            ),
            nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
                """import json
from pathlib import Path
import pandas as pd
import plotly.express as px
result = json.loads(Path('comparison.json').read_text())
assert result['outer_holdout_touched'] is False
print('Smoke run:', result['smoke'])
frame = pd.DataFrame(result['rows'])
frame"""
            ),
            nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
                """px.bar(
    frame, x='experiment', y='delta_vs_control',
    title='Mean fold stability change vs control', template='plotly_white'
).show()"""
            ),
            nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
                """folds = pd.DataFrame(result['folds'])
px.line(
    folds, x='fold', y='stability_score', color='experiment', markers=True,
    template='plotly_white', title='Stability by temporal fold'
).show()"""
            ),
            nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
                "Only feature blocks change. Early stopping and selection use development folds, "
                "so these are not final test scores. "
                "ROC AUC and average precision measure ranking; "
                "Brier score and log loss assess probabilities."
            ),
        ]
    )
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ablation-{index}"
    nbformat.validate(notebook)
    if destination.is_file():
        previous = nbformat.read(destination, as_version=4)  # type: ignore[no-untyped-call]
        if [c.source for c in previous.cells] == [c.source for c in notebook.cells]:
            return
    atomic_write(destination, nbformat.writes(notebook).encode())  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    raise SystemExit(main())
