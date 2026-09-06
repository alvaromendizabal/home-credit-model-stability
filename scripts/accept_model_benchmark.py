#!/usr/bin/env python3
"""Accept a completed benchmark and produce its portable evidence report."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from home_credit.modeling.acceptance import accept_benchmark, read_json, require
from home_credit.modeling.checkpoints import (
    atomic_write,
    load_manifest_bytes,
    object_key_for_sha,
    sha256_bytes,
    sha256_file,
)
from home_credit.modeling.report import build_report
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import Heartbeat, StageTimer


def restore_bundle(
    root: Path, policy: dict[str, Any], logger: RunLogger, *, bucket: str, client: Any = None
) -> None:
    """Download verified files, atomically replacing only the acceptance cache."""
    require(bool(bucket) and "/" not in bucket, "a valid artifact bucket name is required")
    prefix = str(policy["s3_prefix"]).strip("/")
    if client is None:
        client = boto3.client(
            "s3",
            region_name=policy["region"],
            config=Config(
                connect_timeout=10,
                read_timeout=60,
                retries={"mode": "standard", "total_max_attempts": 3},
            ),
        )
    key = f"{prefix}/benchmark-{policy['manifest_sha256']}.json"
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        payload = body.read()
    require(sha256_bytes(payload) == policy["manifest_sha256"], "remote manifest hash mismatch")
    manifest = load_manifest_bytes(payload)
    require(manifest.run_key == policy["run_key"], "remote run key mismatch")
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    for member in manifest.files:
        require(
            member.object_key == object_key_for_sha(prefix, member.sha256),
            f"remote object key mismatch: {member.path}",
        )
        target = (root / member.path).resolve()
        require(target.is_relative_to(root.resolve()), "download destination escaped cache")
        if (
            target.is_file()
            and target.stat().st_size == member.bytes
            and sha256_file(target) == member.sha256
        ):
            logger.event("artifact_cached", path=member.path)
        else:
            pending.append((member, target))
    needed = sum(m.bytes for m, _ in pending) + 64 * 1024 * 1024
    require(shutil.disk_usage(root).free >= needed, f"insufficient free disk; need {needed} bytes")
    for index, (member, target) in enumerate(pending, 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        try:
            client.download_file(bucket, member.object_key, str(temporary))
            require(
                temporary.stat().st_size == member.bytes, f"download size mismatch: {member.path}"
            )
            require(
                sha256_file(temporary) == member.sha256, f"download hash mismatch: {member.path}"
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        logger.event("artifact_downloaded", path=member.path, completed=index, total=len(pending))
    atomic_write(root / "checkpoint_manifest.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/benchmark_acceptance.json"))
    parser.add_argument(
        "--benchmark-dir", type=Path, default=Path("artifacts/benchmark_acceptance")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/benchmark"))
    parser.add_argument("--download", action="store_true", help="Restore the pinned S3 publication")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("HOME_CREDIT_ARTIFACT_BUCKET"),
        help="Private artifact bucket; alternatively set HOME_CREDIT_ARTIFACT_BUCKET",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.download and not args.bucket:
        parser.error("--download requires --bucket or HOME_CREDIT_ARTIFACT_BUCKET")
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive and finite")
    started = time.monotonic()
    logger = RunLogger("benchmark-acceptance", Path("logs"))
    logger.event("acceptance_started", benchmark_dir=args.benchmark_dir, log=logger.jsonl_path)
    try:
        policy = read_json(args.policy)
        args.benchmark_dir.mkdir(parents=True, exist_ok=True)
        # Locks are released by the OS even if the terminal or process dies.
        with (args.benchmark_dir / ".acceptance.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with Heartbeat(logger, interval_seconds=args.heartbeat_seconds, label="acceptance"):
                if args.download:
                    with StageTimer(logger, "restore_published_benchmark"):
                        restore_bundle(args.benchmark_dir, policy, logger, bucket=args.bucket)
                with StageTimer(logger, "verify_artifacts_and_recompute_metrics"):
                    evidence = accept_benchmark(
                        args.benchmark_dir,
                        policy=policy,
                        config_path=Path("configs/model_benchmark.json"),
                        protocol_path=Path("configs/validation_protocol.json"),
                        logger=logger,
                    )
                with StageTimer(logger, "render_benchmark_report"):
                    build_report(evidence, args.output_dir)
        logger.event(
            "acceptance_completed",
            status="accepted",
            leader=evidence["leader"],
            model_folds=evidence["model_folds"],
            report=args.output_dir / "report.html",
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        print("PHASE_5A_ACCEPTANCE_COMPLETED", flush=True)
        return 0
    except KeyboardInterrupt:
        logger.event(
            "acceptance_interrupted", total_elapsed_seconds=round(time.monotonic() - started, 3)
        )
        return 130
    except Exception as exc:
        logger.event(
            "acceptance_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
