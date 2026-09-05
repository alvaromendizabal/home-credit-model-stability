#!/usr/bin/env python3
"""Build leakage-safe Home Credit case-level feature blocks from locked S3 raw data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import polars as pl

from home_credit.data.loader import load_s3_manifest
from home_credit.features.builder import (
    FeatureRecipe,
    block_totals,
    build_feature_blocks,
    group_logical_sources,
    load_validation_protocol,
    select_sources,
    write_feature_manifest,
)
from home_credit.features.execution import FeatureExecutionPolicy
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import Heartbeat, StageTimer


def parse_args() -> argparse.Namespace:
    """Parse the feature-build command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--validation-protocol",
        type=Path,
        default=Path("configs/validation_protocol.json"),
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/feature_recipe.json"),
    )
    parser.add_argument(
        "--execution-policy",
        type=Path,
        default=Path("configs/feature_execution.json"),
    )
    parser.add_argument("--expected-execution-sha256", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/features"),
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated subset of train,test.",
    )
    parser.add_argument(
        "--families",
        default=None,
        help="Optional comma-separated logical families. Base is always included.",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    return parser.parse_args()


def _csv_set(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    result = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("comma-separated selection must not be empty")
    return result


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _atomic_json(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Build selected feature blocks and provenance artifacts."""
    args = parse_args()
    if args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat-seconds must be positive")

    splits = _csv_set(args.splits)
    assert splits is not None
    unknown_splits = splits - {"train", "test"}
    if unknown_splits:
        raise ValueError(f"unknown split(s): {sorted(unknown_splits)}")
    families = _csv_set(args.families)

    logger = RunLogger("feature-build", Path("logs"))
    started = time.monotonic()
    logger.event(
        "feature_run_started",
        manifest_uri=args.manifest_uri,
        output_dir=args.output_dir,
        splits=sorted(splits),
        families="all" if families is None else sorted(families),
        region=args.region,
    )

    with Heartbeat(
        logger,
        interval_seconds=args.heartbeat_seconds,
        label="feature_run",
    ):
        with StageTimer(logger, "recipe_load"):
            recipe, recipe_sha256 = FeatureRecipe.load(args.recipe)

        with StageTimer(logger, "execution_policy_load"):
            execution_policy, execution_sha256 = FeatureExecutionPolicy.load(args.execution_policy)
            if execution_sha256 != args.expected_execution_sha256:
                raise ValueError(
                    "feature execution SHA-256 mismatch: "
                    f"expected={args.expected_execution_sha256} "
                    f"actual={execution_sha256}"
                )
            actual_threads = pl.thread_pool_size()
            if actual_threads > execution_policy.max_threads:
                raise ValueError(
                    "Polars thread pool exceeds the bounded-memory policy: "
                    f"actual={actual_threads} max={execution_policy.max_threads}. "
                    "Set POLARS_MAX_THREADS before starting Python."
                )
            logger.event(
                "feature_execution_runtime",
                polars_threads=actual_threads,
                partition_rows=execution_policy.partition_rows,
                partition_threshold_rows=execution_policy.partition_threshold_rows,
                partition_min_source_bytes=execution_policy.partition_min_source_bytes,
                resume=execution_policy.resume,
            )

        with StageTimer(logger, "protocol_load"):
            protocol = load_validation_protocol(
                args.validation_protocol,
                expected_sha256=args.expected_protocol_sha256,
            )

        with StageTimer(logger, "manifest_load"):
            records, manifest_sha256, store = load_s3_manifest(
                args.manifest_uri,
                region=args.region,
            )

        if manifest_sha256 != args.expected_manifest_sha256:
            raise ValueError(
                "raw manifest SHA-256 mismatch: "
                f"expected={args.expected_manifest_sha256} actual={manifest_sha256}"
            )

        sources = group_logical_sources(records)
        selected = select_sources(sources, splits=splits, families=families)
        selected_records = {
            record.s3_key: record for source in selected for record in source.records
        }

        with StageTimer(
            logger,
            "raw_object_verification",
            heartbeat_seconds=args.heartbeat_seconds,
        ):
            for index, record in enumerate(
                sorted(selected_records.values(), key=lambda item: item.file),
                start=1,
            ):
                store.verify_record(record)
                logger.event(
                    "raw_object_verified",
                    index=index,
                    total=len(selected_records),
                    file=record.file,
                )

        protocol_sha256 = str(protocol["protocol_sha256"])
        blocks = build_feature_blocks(
            records,
            store,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            protocol_sha256=protocol_sha256,
            raw_manifest_sha256=manifest_sha256,
            execution_policy=execution_policy,
            execution_sha256=execution_sha256,
            output_dir=args.output_dir,
            logger=logger,
            heartbeat_seconds=args.heartbeat_seconds,
            splits=splits,
            families=families,
        )

        with StageTimer(logger, "manifest_write"):
            feature_manifest = args.output_dir / "feature_manifest.json"
            write_feature_manifest(
                blocks,
                output=feature_manifest,
                manifest_uri=args.manifest_uri,
                manifest_sha256=manifest_sha256,
                protocol_sha256=protocol_sha256,
                recipe_sha256=recipe_sha256,
                execution_sha256=execution_sha256,
            )

            totals = block_totals(blocks)
            run_metadata = {
                "schema_version": 1,
                "git_commit": _git_commit(),
                "raw_manifest_uri": args.manifest_uri,
                "raw_manifest_sha256": manifest_sha256,
                "validation_protocol_sha256": protocol_sha256,
                "feature_recipe": recipe.name,
                "feature_recipe_sha256": recipe_sha256,
                "feature_execution_policy": execution_policy.mode,
                "feature_execution_sha256": execution_sha256,
                "partition_rows": execution_policy.partition_rows,
                "polars_threads": pl.thread_pool_size(),
                "splits": sorted(splits),
                "families": "all" if families is None else sorted(families),
                "blocks": totals["blocks"],
                "feature_columns": totals["feature_columns"],
                "output_bytes": totals["output_bytes"],
            }
            _atomic_json(run_metadata, args.output_dir / "run_metadata.json")

    elapsed = round(time.monotonic() - started, 3)
    totals = block_totals(blocks)
    logger.event(
        "feature_run_completed",
        blocks=totals["blocks"],
        feature_columns=totals["feature_columns"],
        output_bytes=totals["output_bytes"],
        total_elapsed_seconds=elapsed,
        feature_manifest=feature_manifest.as_posix(),
    )

    print(f"FEATURE_BLOCKS={totals['blocks']}")
    print(f"FEATURE_COLUMNS={totals['feature_columns']}")
    print(f"FEATURE_OUTPUT_BYTES={totals['output_bytes']}")
    print(f"FEATURE_RECIPE_SHA256={recipe_sha256}")
    print(f"FEATURE_EXECUTION_SHA256={execution_sha256}")
    print(f"VALIDATION_PROTOCOL_SHA256={protocol_sha256}")
    print("FEATURE_BUILD_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
