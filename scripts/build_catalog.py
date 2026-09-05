#!/usr/bin/env python3
"""Build and validate the immutable Home Credit raw-data catalog from S3."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from home_credit.data.catalog import (
    build_s3_catalog,
    write_file_catalog,
    write_json,
    write_table_catalog,
)
from home_credit.data.contracts import validate_catalog
from home_credit.data.loader import load_s3_manifest, parquet_records
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parquet-files", type=int, default=68)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/catalog"))
    return parser.parse_args()


def main() -> int:
    """Run catalog generation, structural contracts, and canonical artifact writes."""
    args = parse_args()
    logger = RunLogger("catalog", Path("logs"))
    started = time.monotonic()
    logger.event("catalog_run_started", manifest_uri=args.manifest_uri, region=args.region)

    with StageTimer(logger, "manifest_load"):
        records, manifest_sha256, store = load_s3_manifest(
            args.manifest_uri,
            region=args.region,
        )

    parquet_count = sum(1 for _ in parquet_records(records))
    logger.event(
        "manifest_loaded",
        records=len(records),
        parquet_files=parquet_count,
        manifest_sha256=manifest_sha256,
        bucket=store.bucket,
    )

    if manifest_sha256 != args.expected_manifest_sha256:
        logger.event(
            "catalog_run_failed",
            reason="manifest_sha256_mismatch",
            expected=args.expected_manifest_sha256,
            actual=manifest_sha256,
        )
        return 1

    if parquet_count != args.expected_parquet_files:
        logger.event(
            "catalog_run_failed",
            reason="parquet_file_count_mismatch",
            expected=args.expected_parquet_files,
            actual=parquet_count,
        )
        return 1

    entries = build_s3_catalog(records, store, logger=logger)

    with StageTimer(logger, "catalog_contracts"):
        report = validate_catalog(entries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_catalog = args.output_dir / "file_catalog.jsonl"
    table_catalog = args.output_dir / "table_catalog.json"
    contract_report = args.output_dir / "contract_report.json"
    run_metadata = args.output_dir / "run_metadata.json"

    write_file_catalog(entries, file_catalog)
    write_table_catalog(entries, table_catalog)
    write_json(report.to_dict(), contract_report)
    write_json(
        {
            "manifest_uri": args.manifest_uri,
            "manifest_sha256": manifest_sha256,
            "parquet_files": len(entries),
        },
        run_metadata,
    )

    logger.event(
        "catalog_contracts_completed",
        passed=report.passed,
        checks=report.checks,
        errors=len(report.errors),
        warnings=len(report.warnings),
    )
    for warning in report.warnings:
        logger.event("catalog_contract_warning", detail=warning)
    for error in report.errors:
        logger.event("catalog_contract_error", detail=error)

    logger.event(
        "catalog_run_completed",
        passed=report.passed,
        files=len(entries),
        total_elapsed_seconds=round(time.monotonic() - started, 3),
        file_catalog=file_catalog.as_posix(),
        table_catalog=table_catalog.as_posix(),
        contract_report=contract_report.as_posix(),
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
