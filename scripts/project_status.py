#!/usr/bin/env python3
"""Inspect verified project evidence; optionally read AWS jobs and hash cloud checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]

from home_credit.observability.logging import RunLogger
from home_credit.observability.project import (
    cloud_artifacts,
    cloud_jobs,
    published_status,
    verify_cloud_artifacts,
)
from home_credit.observability.runtime import StageTimer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cloud", action="store_true", help="Read live SageMaker job status")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--verify-cloud", action="store_true", help="Stream and hash saved checkpoints"
    )
    parser.add_argument("--bucket", default="sagemaker-us-west-2-560403859723")
    parser.add_argument(
        "--json", action="store_true", help="Write machine-readable status to stdout"
    )
    args = parser.parse_args()
    started = time.monotonic()
    try:
        # Keep stdout parseable while preserving existing UTC/heartbeat logging on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            logger = RunLogger("project-status", args.root / "logs")
            with StageTimer(logger, "verify_published_evidence", heartbeat_seconds=15):
                result = published_status(args.root)
            if args.cloud or args.verify_cloud:
                session = boto3.Session(region_name=args.region)
                with StageTimer(logger, "inspect_cloud_jobs", heartbeat_seconds=15):
                    result["cloud"] = cloud_jobs(session.client("sagemaker"), args.region)
                if args.verify_cloud:
                    with StageTimer(logger, "verify_cloud_checkpoints", heartbeat_seconds=15):
                        result["cloud"]["artifacts"] = verify_cloud_artifacts(
                            session.client("s3"), args.bucket, cloud_artifacts(args.root)
                        )
        result["checked_utc"] = datetime.now(UTC).isoformat()
        result["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        else:
            features = result["features"]
            print(f"Checked (UTC): {result['checked_utc']}")
            print("Published release and completed development studies: VERIFIED")
            print(
                "Expanded feature completion gate: "
                + ("PASSED" if result["feature_completion_gate_passed"] else "OPEN")
            )
            print(
                f"Features: {features['combined_hypotheses']:,} hypotheses; "
                f"{features['release_retained']:,} in the release; "
                f"{features['additional_retained_for_experiment']} tested additions, none promoted."
            )
            print(f"Frozen holdout stability: {result['frozen_evaluation']['stability_score']:.6f}")
            print("Holdout weeks 73-91 are observed and unavailable for new selection.")
            for row in result["remaining_requirements"]:
                print(f"Remaining: {row['next_action']}")
            cloud = result["cloud"]
            if cloud["checked"]:
                print(f"Active Home Credit processing/training jobs: {len(cloud['active_jobs'])}")
                for job in cloud["jobs"]:
                    print(f"  {job['status']:12} {job['name']}")
                if "artifacts" in cloud:
                    print(f"Cloud checkpoints: {cloud['artifacts']}")
            else:
                print("AWS: not queried; use --cloud for current job status.")
            print(f"Elapsed: {result['total_elapsed_seconds']:.3f}s; new model fits: 0")
        return 0
    except (ValueError, KeyError, OSError, BotoCoreError, ClientError) as exc:
        print(f"PROJECT_STATUS_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
