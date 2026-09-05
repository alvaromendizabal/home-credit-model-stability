#!/usr/bin/env python3
"""Audit Home Credit temporal structure before freezing validation boundaries."""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq

from home_credit.data.loader import RawManifestRecord, S3RawStore, load_s3_manifest
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import Heartbeat, StageTimer
from home_credit.validation.temporal import (
    as_serializable,
    build_holdout_candidates,
    build_week_profile,
    prepare_base_frame,
    summarize_base,
    temporal_integrity,
    write_json,
    write_week_profile,
)

_TRAIN_BASE = "parquet_files/train/train_base.parquet"
_TEST_BASE = "parquet_files/test/test_base.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--heartbeat-seconds", type=float, default=2.0)
    return parser.parse_args()


def _find_record(records: Sequence[RawManifestRecord], file_name: str) -> RawManifestRecord:
    matches = [record for record in records if record.file == file_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest record for {file_name}, found {len(matches)}"
        )
    return matches[0]


def _read_base(
    record: RawManifestRecord,
    store: S3RawStore,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    store.verify_record(record)
    with store.open_input_file(record.s3_key) as stream:
        table = pq.read_table(stream, columns=list(columns), use_threads=True)
    return cast(pd.DataFrame, table.to_pandas())


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    """Run temporal audit, emit candidate OOT splits, and fail on leakage risks."""
    args = parse_args()
    if args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be positive")

    logger = RunLogger("validation-audit", Path("logs"))
    started = time.monotonic()
    logger.event(
        "validation_audit_started",
        manifest_uri=args.manifest_uri,
        region=args.region,
        output_dir=args.output_dir.as_posix(),
    )

    errors: list[str] = []
    warnings: list[str] = []

    with Heartbeat(
        logger,
        interval_seconds=args.heartbeat_seconds,
        label="validation_audit",
    ):
        with StageTimer(logger, "manifest_load"):
            records, manifest_sha256, store = load_s3_manifest(
                args.manifest_uri,
                region=args.region,
            )

        if manifest_sha256 != args.expected_manifest_sha256:
            logger.event(
                "validation_audit_failed",
                reason="manifest_sha256_mismatch",
                expected=args.expected_manifest_sha256,
                actual=manifest_sha256,
            )
            return 1

        train_record = _find_record(records, _TRAIN_BASE)
        test_record = _find_record(records, _TEST_BASE)

        with StageTimer(logger, "train_base_read", heartbeat_seconds=args.heartbeat_seconds):
            train_raw = _read_base(
                train_record,
                store,
                columns=("case_id", "date_decision", "WEEK_NUM", "target"),
            )

        logger.event("train_base_loaded", rows=len(train_raw), source_bytes=train_record.bytes)

        with StageTimer(logger, "test_base_read", heartbeat_seconds=args.heartbeat_seconds):
            test_raw = _read_base(
                test_record,
                store,
                columns=("case_id", "date_decision", "WEEK_NUM"),
            )

        logger.event("test_base_loaded", rows=len(test_raw), source_bytes=test_record.bytes)

        with StageTimer(logger, "base_contracts"):
            train = prepare_base_frame(train_raw, require_target=True)
            test = prepare_base_frame(test_raw, require_target=False)

        with StageTimer(logger, "weekly_profile", heartbeat_seconds=args.heartbeat_seconds):
            week_profile = build_week_profile(train)
            candidates = build_holdout_candidates(week_profile)

        train_summary = summarize_base(train, split="train")
        test_summary = summarize_base(test, split="test")
        integrity = temporal_integrity(train, test, week_profile)

        if integrity.case_id_overlap != 0:
            errors.append(f"train/test case_id overlap detected: {integrity.case_id_overlap}")
        if integrity.week_start_reversals != 0:
            errors.append(
                f"WEEK_NUM/date_decision start-order reversals: {integrity.week_start_reversals}"
            )
        if integrity.week_end_reversals != 0:
            errors.append(
                f"WEEK_NUM/date_decision end-order reversals: {integrity.week_end_reversals}"
            )
        if not integrity.test_starts_after_train_week:
            warnings.append(
                "test WEEK_NUM does not start strictly after the training WEEK_NUM range"
            )
        if not integrity.test_starts_after_train_date:
            warnings.append(
                "test date_decision does not start strictly after the training date range"
            )
        if not any(candidate.eligible for candidate in candidates):
            errors.append("no robust trailing-week validation candidate is metric-eligible")

        for candidate in candidates:
            logger.event(
                "holdout_candidate",
                name=candidate.name,
                train_weeks=candidate.train_weeks,
                validation_weeks=candidate.validation_weeks,
                validation_week_min=candidate.validation_week_min,
                validation_week_max=candidate.validation_week_max,
                validation_rows=candidate.validation_rows,
                validation_row_fraction=round(candidate.validation_row_fraction, 6),
                target_rate_shift=round(candidate.target_rate_shift, 6),
                metric_eligible_weeks=candidate.validation_metric_eligible_weeks,
                eligible=candidate.eligible,
            )

        passed = not errors
        args.output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = args.output_dir / "temporal_audit.json"
        profile_path = args.output_dir / "week_profile.jsonl"
        candidates_path = args.output_dir / "holdout_candidates.json"
        metadata_path = args.output_dir / "run_metadata.json"

        write_week_profile(week_profile, profile_path)
        write_json([as_serializable(candidate) for candidate in candidates], candidates_path)
        write_json(
            {
                "passed": passed,
                "errors": errors,
                "warnings": warnings,
                "train": asdict(train_summary),
                "test": asdict(test_summary),
                "integrity": asdict(integrity),
                "week_profile_rows": len(week_profile),
                "holdout_candidates": len(candidates),
            },
            audit_path,
        )
        write_json(
            {
                "manifest_uri": args.manifest_uri,
                "manifest_sha256": manifest_sha256,
                "train_base_sha256": train_record.sha256,
                "test_base_sha256": test_record.sha256,
                "git_commit": _git_commit(),
            },
            metadata_path,
        )

    for warning in warnings:
        logger.event("validation_audit_warning", detail=warning)
    for error in errors:
        logger.event("validation_audit_error", detail=error)

    logger.event(
        "validation_audit_completed",
        passed=not errors,
        weeks=train_summary.weeks,
        week_min=train_summary.week_min,
        week_max=train_summary.week_max,
        candidates=len(candidates),
        errors=len(errors),
        warnings=len(warnings),
        total_elapsed_seconds=round(time.monotonic() - started, 3),
        audit_path=audit_path.as_posix(),
        profile_path=profile_path.as_posix(),
        candidates_path=candidates_path.as_posix(),
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
