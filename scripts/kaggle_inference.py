#!/usr/bin/env python3
"""Run the frozen native ensemble in an isolated, offline Python 3.12 environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Any

BUNDLE_SHA256 = "71f338a66b15d0f8b549c3a9c9defe4defdd4169f5f57be3c65036ac9bcbf84a"


def digest(path: Path) -> str:
    """Hash large files without materializing their contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets(directory: Path, expected: str) -> dict[str, Any]:
    """Bind every executable, model and dependency to the supplied manifest digest."""
    manifest = directory / "runtime.json"
    if digest(manifest) != expected:
        raise ValueError("offline runtime manifest digest mismatch")
    data: dict[str, Any] = json.loads(manifest.read_text())
    if data["schema_version"] != 1 or data["bundle_sha256"] != BUNDLE_SHA256:
        raise ValueError("offline runtime does not describe the frozen release")
    paths = [member["path"] for member in data["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate runtime member")
    for member in data["files"]:
        path = directory / member["path"]
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise ValueError("unsafe runtime member path")
        if digest(path) != member["sha256"] or path.stat().st_size != member["bytes"]:
            raise ValueError(f"offline runtime member changed: {member['path']}")
    required = {"bundle/bundle.json", "requirements.txt", "kaggle_inference.py"}
    if not required <= set(paths):
        raise ValueError("incomplete offline runtime")
    return data


def run_command(command: list[str], *, environment: dict[str, str]) -> None:
    """Preserve child output and emit UTC heartbeats even during a long silent stage."""
    started = time.monotonic()
    with subprocess.Popen(command, env=environment) as child:
        while True:
            try:
                code = child.wait(timeout=15)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
            except subprocess.TimeoutExpired:
                print(
                    time.strftime("[%Y-%m-%dT%H:%M:%SZ]", time.gmtime()),
                    f"inference_process_heartbeat elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )


def verify_versions(versions: dict[str, str]) -> None:
    """Fail before scoring if any locked dependency changed."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("the frozen inference runtime requires Python 3.12")
    for name, expected in versions.items():
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {observed}")


def worker(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Score the test population supplied to this run; never fit or contact Kaggle."""
    verify_versions(manifest["versions"])
    bundle = args.assets / "bundle"
    sys.path.insert(0, str(bundle / "source"))

    import pandas as pd

    from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes
    from home_credit.modeling.inference import export_submission, local_test_records, predict_raw
    from home_credit.modeling.release_workflow import ReleaseLogger

    logger = ReleaseLogger("kaggle-inference", args.output / "logs")
    logger.event("offline_inference_started", runtime_sha256=args.runtime_sha256)
    predictions = predict_raw(
        bundle,
        args.raw,
        args.work / "prediction_batches",
        expected_bundle_sha256=BUNDLE_SHA256,
        logger=logger,
        batch_rows=args.batch_rows,
    )
    records = local_test_records(args.raw)
    input_sha256 = sha256_bytes(canonical_json_bytes({r.file: r.sha256 for r in records}))
    destination = export_submission(
        pd.read_csv(args.sample),
        predictions,
        args.output / "submission.csv",
        lineage={
            "bundle_sha256": BUNDLE_SHA256,
            "input_sha256": input_sha256,
            "runtime_sha256": args.runtime_sha256,
            "sample_sha256": digest(args.sample),
            "model_fits": 0,
            "evaluation_scope": "test_population_supplied_by_this_run",
        },
        owner_confirmed=True,
    )
    logger.event(
        "submission_validated", rows=len(predictions), sha256=digest(destination), model_fits=0
    )
    print(f"SUBMISSION_VALIDATED rows={len(predictions)} sha256={digest(destination)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=20000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("use a Python 3.12 Kaggle CPU notebook")
    for field in ("assets", "raw", "sample", "work", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if not args.raw.is_dir() or not args.sample.is_file() or args.batch_rows < 1:
        raise ValueError("missing competition input or invalid batch size")
    manifest = verify_assets(args.assets, args.runtime_sha256)
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, manifest)
        return
    environment = dict(os.environ)
    environment.update(
        POLARS_MAX_THREADS="4", OMP_NUM_THREADS="4", PYTHONWARNINGS="error", PYTHONUNBUFFERED="1"
    )
    runtime = args.work / "venv"
    python = runtime / "bin/python"
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(runtime)
    run_command(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-cache-dir",
            "--require-hashes",
            "--only-binary=:all:",
            "--find-links",
            str(args.assets / "wheels"),
            "-r",
            str(args.assets / "requirements.txt"),
        ],
        environment=environment,
    )
    run_command(
        [str(python), "-I", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        environment=environment,
    )


if __name__ == "__main__":
    main()
