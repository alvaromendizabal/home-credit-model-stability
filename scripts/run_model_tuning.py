#!/usr/bin/env python3
"""Run a bounded, resumable Optuna LightGBM study on the frozen temporal protocol."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.modeling.acceptance import read_json, require
from home_credit.modeling.checkpoints import (
    atomic_write,
    canonical_json_bytes,
    sha256_file,
    validate_benchmark_state,
)
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.tuning import (
    evaluate_trial,
    load_plan,
    propose,
    rank_records,
    search_space,
    trial_config,
)
from home_credit.modeling.tuning_report import publish_report
from home_credit.observability.logging import RunLogger, iso_utc
from home_credit.observability.runtime import StageTimer


def main() -> int:
    signal.signal(signal.SIGTERM, handle_termination)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="One capped trial in separate storage."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(Path.cwd() == root, "run from the repository root")
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "commit tracked source changes before tuning",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plan = load_plan(root)
    mode = "smoke" if args.smoke else "full"
    work = root / "artifacts/model_tuning" / commit / mode
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    logger = RunLogger("model-tuning", work / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 3},
            connect_timeout=5,
            read_timeout=30,
            max_pool_connections=16,
        ),
    )
    prefix = f"home-credit-model-stability/model-tuning/{commit}/{mode}"
    store = ExperimentStore(client, args.bucket, prefix, work, logger)
    with (work / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this tuning study already has a local worker") from exc
        try:
            with WriterLease(store) as lease:
                run_study(root, work, plan, commit, args.smoke, store, lease, logger)
        except Exception as exc:
            logger.event(
                "model_tuning_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                total_elapsed_seconds=round(time.monotonic() - started, 3),
            )
            # Checkpoints and the ledger are already durable. Preserve diagnostics when reachable.
            try:
                store.publish(logger.jsonl_path, "logs")
            except Exception as upload_error:
                logger.event(
                    "log_upload_pending",
                    error_type=type(upload_error).__name__,
                    local_path=logger.jsonl_path,
                )
            raise
        logger.event(
            "model_tuning_completed",
            mode=mode,
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        store.publish(logger.jsonl_path, "logs")
    print("MODEL_TUNING_SMOKE_COMPLETED" if args.smoke else "MODEL_TUNING_COMPLETED", flush=True)
    return 0


def handle_termination(signum: int, frame: Any) -> None:
    """Unwind the active worker so its process group stops before releasing the lease."""
    raise InterruptedError(f"received signal {signum}; committed folds remain resumable")


def baseline_predictions(root: Path, store: ExperimentStore, smoke: bool) -> pl.DataFrame:
    """Restore only the original control predictions using the committed immutable manifest."""
    manifest = read_json(root / "reports/feature_ablation/control_manifest.json")
    require(
        manifest["smoke"] is False and manifest["completed_model_folds"] == 5,
        "baseline must be a completed full-data control",
    )
    paths = []
    for item in manifest["files"]:
        if not item["path"].startswith("predictions/lightgbm/"):
            continue
        if smoke and item["path"] != "predictions/lightgbm/fold_1.parquet":
            continue
        path = store.root / "control" / item["path"]
        require(path.resolve().is_relative_to(store.root.resolve()), "unsafe baseline path")
        store.download(item["object_key"], path, item["sha256"])
        paths.append(path)
    require(len(paths) == (1 if smoke else 5), "baseline fold predictions are incomplete")
    return pl.concat([pl.read_parquet(p) for p in paths])


def run_study(
    root: Path,
    work: Path,
    plan: dict[str, Any],
    commit: str,
    smoke: bool,
    store: ExperimentStore,
    lease: WriterLease,
    logger: RunLogger,
) -> None:
    """Commit proposals before training, and observations after verified fold completion."""
    started = time.monotonic()
    identity = {
        "git_commit": commit,
        "plan_sha256": sha256_file(root / "configs/model_tuning.json"),
        "smoke": smoke,
    }
    state = store.restore(identity)
    budget = 1 if smoke else plan["new_trials"]
    folds = read_json(root / "configs/validation_protocol.json")["inner_temporal_cv"]["folds"]
    if smoke:
        folds = folds[:1]
    base = read_json(root / "configs/ablations/control.json")
    with StageTimer(logger, "restore_control", heartbeat_seconds=15):
        control = baseline_predictions(root, store, smoke)
    if state is None:
        baseline = {
            "slot": 0,
            "name": "control",
            "state": "complete",
            "reused": True,
            "params": {key: base["models"]["lightgbm"][key] for key in search_space(plan)},
            **evaluate_trial(control, control, folds, "baseline"),
            "source_commit": plan["baseline_commit"],
        }
        baseline["metrics"]["experiment"] = "control"
        for row in baseline["folds"]:
            row["experiment"] = "control"
        if not smoke:
            accepted = read_json(root / "reports/feature_ablation/comparison.json")["rows"][0]
            require(
                abs(baseline["value"] - accepted["mean_fold_stability"]) < 1e-10,
                "baseline metric differs from accepted ablation",
            )
        state = {
            "schema_version": 1,
            "identity": identity,
            "created_utc": iso_utc(),
            "revision": 0,
            "trials": [baseline],
            "new_trial_budget": budget,
            "outer_holdout_touched": False,
            "complete": False,
        }
        lease.check()
        store.commit(state)
    require(state["new_trial_budget"] == budget, "study budget changed")
    require(
        [r["slot"] for r in state["trials"]] == list(range(len(state["trials"]))),
        "study trial sequence is not contiguous",
    )
    logger.event(
        "model_tuning_started",
        new_trial_budget=budget,
        new_fit_budget=budget * len(folds),
        reused_control_folds=len(folds),
        smoke=smoke,
        git_commit=commit,
    )
    while sum(r["state"] == "complete" for r in state["trials"][1:]) < budget:
        lease.check()
        if state["trials"][-1]["state"] == "complete":
            proposal = propose(plan, state["trials"], len(state["trials"]))
            proposal["created_utc"] = iso_utc()
            state["trials"].append(proposal)
            state["revision"] += 1
            store.commit(state)
        record = state["trials"][-1]
        config_path = work / "configs" / f"{record['name']}.json"
        config_payload = canonical_json_bytes(trial_config(base, record))
        if config_path.exists():
            require(config_path.read_bytes() == config_payload, "trial configuration changed")
        else:
            atomic_write(config_path, config_payload)
        BenchmarkConfig.load(config_path)
        output = work / record["name"]
        logger.event(
            "tuning_trial_started",
            trial=record["slot"],
            total=budget,
            params=json.dumps(record["params"], sort_keys=True),
        )
        trial_started = time.monotonic()
        command = supervisor_command(root, work, plan, config_path, output, store, smoke)
        run_process(
            command, output, record["slot"], budget, len(folds), started, store, lease, logger
        )
        require(
            validate_benchmark_state(
                output,
                git_commit=commit,
                feature_manifest_sha256=plan["feature_manifest_sha256"],
                validation_protocol_sha256=plan["protocol_sha256"],
                benchmark_config_sha256=sha256_file(config_path),
                smoke=smoke,
            )
            == len(folds),
            "trial is missing verified model-fold checkpoints",
        )
        benchmark = read_json(output / "benchmark_state.json")
        candidate = pl.concat(
            [pl.read_parquet(output / r["prediction_path"]) for r in benchmark["folds"].values()]
        )
        reference = (
            control.filter(pl.col("case_id").is_in(candidate["case_id"].implode()))
            if smoke
            else control
        )
        if smoke:
            # Smoke data are capped; display the control on precisely the same cases.
            baseline_result = evaluate_trial(reference, reference, folds, "baseline")
            baseline_result["metrics"]["experiment"] = "control"
            for row in baseline_result["folds"]:
                row["experiment"] = "control"
            state["trials"][0].update(baseline_result)
        with StageTimer(logger, "verify_trial_metrics", heartbeat_seconds=15):
            evidence = evaluate_trial(reference, candidate, folds, record["name"])
        record.update(evidence)
        record.update(
            {
                "state": "complete",
                "completed_utc": iso_utc(),
                "attempt_elapsed_seconds": round(time.monotonic() - trial_started, 3),
                "config_sha256": sha256_file(config_path),
                "checkpoint_state_sha256": sha256_file(output / "benchmark_state.json"),
                "checkpoint_prefix": f"{store.prefix}/{record['name']}/checkpoints",
            }
        )
        state["revision"] += 1
        lease.check()
        store.commit(state)
        logger.event(
            "tuning_trial_completed",
            trial=record["slot"],
            total=budget,
            mean_fold_stability=record["value"],
            delta_vs_control=record["metrics"]["delta_vs_control"],
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        with StageTimer(logger, "tuning_report", heartbeat_seconds=15):
            publish_report(root, state, store, logger, execute=False)
        store.publish(logger.jsonl_path, "logs")
    if not state["complete"]:
        state["complete"] = True
        state["selected_trial"] = rank_records(state["trials"])[0]["name"]
        state["revision"] += 1
        lease.check()
        store.commit(state)
    with StageTimer(logger, "tuning_report", heartbeat_seconds=15):
        publish_report(root, state, store, logger, execute=True)
    logger.event(
        "tuning_selection_ready",
        selected_trial=state["selected_trial"],
        outer_holdout_touched=False,
        report=work / "report/report.html",
    )


def supervisor_command(
    root: Path,
    work: Path,
    plan: dict[str, Any],
    config: Path,
    output: Path,
    store: ExperimentStore,
    smoke: bool,
) -> list[str]:
    """Use the existing process-isolated, S3-checkpointed training engine."""
    command = [
        sys.executable,
        "scripts/run_model_benchmark_supervisor.py",
        "--feature-s3-uri",
        f"s3://{store.bucket}/home-credit-model-stability/feature-snapshots/{plan['feature_manifest_sha256']}",
        "--expected-feature-manifest-sha256",
        plan["feature_manifest_sha256"],
        "--expected-protocol-sha256",
        plan["protocol_sha256"],
        "--config",
        str(config),
        "--expected-config-sha256",
        sha256_file(config),
        "--checkpoint-s3-uri",
        f"s3://{store.bucket}/{store.prefix}/{output.name}/checkpoints",
        "--final-s3-uri",
        f"s3://{store.bucket}/{store.prefix}/{output.name}/results",
        "--feature-cache-dir",
        str(root / "artifacts/feature_cache"),
        "--work-dir",
        str(output),
        "--status-log",
        str(work / f"logs/{output.name}.jsonl"),
        "--child-heartbeat-seconds",
        "15",
        "--max-processes",
        "6",
    ]
    if smoke:
        command.append("--smoke")
    return command


def run_process(
    command: list[str],
    output: Path,
    slot: int,
    budget: int,
    fold_count: int,
    started: float,
    store: ExperimentStore,
    lease: WriterLease,
    logger: RunLogger,
) -> None:
    """Stream child output and show whole-study progress; stop descendants on failure."""
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            lease.check()
            try:
                result = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                current = output / "benchmark_state.json"
                folds_done = len(read_json(current)["folds"]) if current.exists() else 0
                completed = (slot - 1) * fold_count + folds_done
                progress = {
                    "timestamp": iso_utc(),
                    "trial": slot,
                    "trial_budget": budget,
                    "completed_new_fits": completed,
                    "total_new_fits": budget * fold_count,
                    "fit_progress_percent": round(100 * completed / (budget * fold_count), 1),
                    "total_elapsed_seconds": round(time.monotonic() - started, 1),
                    "progress_basis": "completed fits; not remaining wall time",
                }
                atomic_write(store.root / "progress.json", canonical_json_bytes(progress))
                logger.event("tuning_heartbeat", **progress)
                continue
            require(result == 0, f"trial worker exited {result}; rerun the same launcher to resume")
            return
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
