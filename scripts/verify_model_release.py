#!/usr/bin/env python3
"""Independently verify archived artifacts and execute raw inference without any fitting/export."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.data.loader import parse_manifest_bytes
from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.inference import load_bundle, predict_raw
from home_credit.modeling.portfolio import load_portfolio
from home_credit.modeling.portfolio_report import write_notebook
from home_credit.modeling.release import checked_member, evaluation_report, verify_file
from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook


def members(value: Any) -> dict[str, dict[str, Any]]:
    """Collect all immutable release members, deduplicating shared encoders and source files."""
    result = {}
    if isinstance(value, dict):
        if {"key", "sha256", "bytes", "path"} <= value.keys():
            result[value["key"]] = value
        for child in value.values():
            result.update(members(child))
    elif isinstance(value, list):
        for child in value:
            result.update(members(child))
    return result


def verify(root: Path, bucket: str) -> dict[str, Any]:
    evidence = load_portfolio(root)
    state = evidence["state"]
    work = root / "artifacts/release_verification"
    work.mkdir(parents=True, exist_ok=True)
    logger = ReleaseLogger("release-verification", root / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    artifacts = members(state)
    paths = {}
    with StageTimer(logger, "verify_all_release_members", heartbeat_seconds=15):
        for number, (key, member) in enumerate(sorted(artifacts.items()), 1):
            path = work / "objects" / member["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".download")
                client.download_file(bucket, key, str(temporary))
                verify_file(temporary, member["sha256"])
                temporary.replace(path)
            verify_file(path, member["sha256"])
            if path.stat().st_size != member["bytes"]:
                raise ValueError("release member byte count changed")
            paths[key] = path
            logger.event("release_member_verified", completed=number, total=len(artifacts))
    with StageTimer(logger, "recompute_saved_holdout_metrics", heartbeat_seconds=15):
        frame = pl.read_parquet(paths[state["stages"]["holdout"]["predictions"]["key"]])
        if frame.height != frame["case_id"].n_unique() or frame["case_id"].null_count():
            raise ValueError("invalid saved holdout case IDs")
        result = evaluation_report(
            frame, frame["prediction"].to_numpy(), expected_weeks=list(range(73, 92))
        )
        for name, value in result["metrics"].items():
            if abs(value - evidence["evaluation"]["metrics"][name]) > 1e-12:
                raise ValueError("saved prediction metrics changed")
    archive = paths[state["stages"]["bundle"]["archive"]["key"]]
    bundle = work / "bundle"
    with zipfile.ZipFile(archive) as stream:
        for archive_member in stream.infolist():
            checked_member(bundle, archive_member.filename)
        stream.extractall(bundle)
    bundle_sha = state["stages"]["bundle"]["manifest"]["sha256"]
    load_bundle(bundle, expected_sha256=bundle_sha)
    protocol = json.loads((root / "configs/validation_protocol.json").read_text())
    raw_key = protocol["data_lock"]["manifest_uri"].split("/", 3)[3]
    manifest = work / "raw_manifest.jsonl"
    client.download_file(bucket, raw_key, str(manifest))
    verify_file(manifest, protocol["data_lock"]["manifest_sha256"])
    raw = work / "raw_test"
    raw.mkdir(exist_ok=True)
    records = [
        r
        for r in parse_manifest_bytes(manifest.read_bytes())
        if Path(r.file).name.startswith("test_")
    ]
    with StageTimer(logger, "restore_verified_public_example", heartbeat_seconds=15):
        for record in records:
            path = raw / Path(record.file).name
            if not path.exists():
                client.download_file(bucket, record.s3_key, str(path))
            verify_file(path, record.sha256)
    cache = work / "prediction_batches"
    with StageTimer(logger, "raw_inference_and_resume", heartbeat_seconds=15):
        first = predict_raw(
            bundle, raw, cache, expected_bundle_sha256=bundle_sha, logger=logger, batch_rows=3
        )
        before = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in cache.rglob("*.parquet")}
        second = predict_raw(
            bundle, raw, cache, expected_bundle_sha256=bundle_sha, logger=logger, batch_rows=3
        )
        after = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in cache.rglob("*.parquet")}
        if not first.equals(second) or before != after or len(first) != 10:
            raise ValueError("public inference or unchanged batch reuse failed")
    os.environ.update(
        HOME_CREDIT_VERIFY_INFERENCE="1",
        HOME_CREDIT_BUNDLE_DIRECTORY=str(bundle),
        HOME_CREDIT_RAW_TEST_DIRECTORY=str(raw),
    )
    notebook = work / "10_submission.ipynb"
    write_notebook(notebook, runpy.run_path(str(root / "scripts/review_submission.py"))["CELLS"])
    with StageTimer(logger, "execute_submission_with_real_raw_inputs", heartbeat_seconds=15):
        execute_notebook(
            root,
            notebook,
            logger,
            force=True,
            dependencies=[root / "scripts/review_submission.py", archive, manifest],
            receipt_path=work / "submission_receipt.json",
            execution_root=root,
        )
    if (root / "submissions/submission.csv").exists():
        raise ValueError("verification unexpectedly produced a real submission CSV")
    receipt = {
        "schema_version": 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "release_key": evidence["policy"]["release_key"],
        "verified_release_objects": len(artifacts),
        "verified_release_bytes": sum(m["bytes"] for m in artifacts.values()),
        "holdout_rows": frame.height,
        "recomputed_metrics": result["metrics"],
        "raw_example_rows": len(first),
        "raw_test_shards": len(records),
        "unchanged_batches_reused": len(before),
        "submission_notebook_executed": True,
        "new_model_fits": 0,
        "submission_generated": False,
        "kaggle_submitted": False,
        "hidden_test_executed": False,
    }
    atomic_write(work / "verification.json", canonical_json_bytes(receipt))
    logger.event("model_release_independently_verified", **receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.bucket)


if __name__ == "__main__":
    main()
