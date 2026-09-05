#!/usr/bin/env python3
"""Freeze the audited Home Credit temporal validation protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import Heartbeat, StageTimer
from home_credit.validation.protocol import (
    attach_protocol_sha256,
    build_expanding_folds,
    fold_payload,
    verify_protocol_sha256,
    write_protocol,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/validation"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/validation_protocol.json"),
    )

    parser.add_argument(
        "--holdout",
        default="tail_20pct_weeks",
    )

    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
    )

    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _candidate_by_name(
    candidates: list[dict[str, object]],
    name: str,
) -> dict[str, object]:
    matches = [candidate for candidate in candidates if candidate.get("name") == name]

    if len(matches) != 1:
        raise ValueError(f"expected exactly one holdout candidate named {name!r}")

    return matches[0]


def _as_int(
    payload: dict[str, object],
    key: str,
) -> int:
    value = payload[key]

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")

    return value


def _as_float(
    payload: dict[str, object],
    key: str,
) -> float:
    value = payload[key]

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{key} must be numeric")

    return float(value)


def _as_bool(
    payload: dict[str, object],
    key: str,
) -> bool:
    value = payload[key]

    if not isinstance(value, bool):
        raise TypeError(f"{key} must be boolean")

    return value


def main() -> int:
    """Validate provenance and freeze validation."""
    args = parse_args()

    if args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be positive")

    logger = RunLogger(
        "validation-protocol",
        Path("logs"),
    )

    started = time.monotonic()

    logger.event(
        "validation_protocol_started",
        audit_dir=args.audit_dir.as_posix(),
        output=args.output.as_posix(),
        holdout=args.holdout,
    )

    with Heartbeat(
        logger,
        interval_seconds=args.heartbeat_seconds,
        label="validation_protocol",
    ):
        with StageTimer(
            logger,
            "load_audit",
        ):
            audit = cast(
                dict[str, object],
                _load_json(args.audit_dir / "temporal_audit.json"),
            )

            metadata = cast(
                dict[str, object],
                _load_json(args.audit_dir / "run_metadata.json"),
            )

            candidates = cast(
                list[dict[str, object]],
                _load_json(args.audit_dir / "holdout_candidates.json"),
            )

        with StageTimer(
            logger,
            "validate_audit",
        ):
            if audit.get("passed") is not True:
                raise ValueError("temporal audit must pass before protocol freeze")

            if audit.get("errors") != []:
                raise ValueError("temporal audit contains errors")

            if metadata.get("manifest_sha256") != args.expected_manifest_sha256:
                raise ValueError("audit manifest SHA-256 does not match the locked raw snapshot")

            current_commit = _git_commit()

            if metadata.get("git_commit") != current_commit:
                raise ValueError(
                    "audit artifacts were not produced by current committed source; rerun the audit"
                )

            integrity = cast(
                dict[str, object],
                audit["integrity"],
            )

            if integrity.get("case_id_overlap") != 0:
                raise ValueError("train/test case_id overlap must be zero")

            if integrity.get("week_start_reversals") != 0:
                raise ValueError("week/date start ordering must be monotonic")

            if integrity.get("week_end_reversals") != 0:
                raise ValueError("week/date end ordering must be monotonic")

            if integrity.get("test_starts_after_train_week") is not True:
                raise ValueError("test weeks must start after training weeks")

            if integrity.get("test_starts_after_train_date") is not True:
                raise ValueError("test dates must start after training dates")

            candidate = _candidate_by_name(
                candidates,
                args.holdout,
            )

            if not _as_bool(
                candidate,
                "eligible",
            ):
                raise ValueError("selected holdout candidate is not eligible")

            if not _as_bool(
                candidate,
                "all_validation_weeks_metric_eligible",
            ):
                raise ValueError("every outer holdout week must contain both target classes")

        with StageTimer(
            logger,
            "build_protocol",
        ):
            development_week_min = _as_int(
                candidate,
                "train_week_min",
            )

            development_week_max = _as_int(
                candidate,
                "train_week_max",
            )

            outer_week_min = _as_int(
                candidate,
                "validation_week_min",
            )

            outer_week_max = _as_int(
                candidate,
                "validation_week_max",
            )

            folds = build_expanding_folds(
                development_week_min=(development_week_min),
                development_week_max=(development_week_max),
                n_splits=5,
                validation_window_weeks=8,
                min_initial_train_weeks=20,
            )

            train_summary = cast(
                dict[str, object],
                audit["train"],
            )

            test_summary = cast(
                dict[str, object],
                audit["test"],
            )

            payload: dict[str, object] = {
                "schema_version": 1,
                "name": ("home_credit_temporal_validation"),
                "seed": 20260905,
                "data_lock": {
                    "manifest_uri": (metadata["manifest_uri"]),
                    "manifest_sha256": (metadata["manifest_sha256"]),
                    "train_base_sha256": (metadata["train_base_sha256"]),
                    "test_base_sha256": (metadata["test_base_sha256"]),
                    "audit_git_commit": (current_commit),
                },
                "observed_temporal_structure": {
                    "train_week_min": (train_summary["week_min"]),
                    "train_week_max": (train_summary["week_max"]),
                    "train_weeks": (train_summary["weeks"]),
                    "train_rows": (train_summary["rows"]),
                    "train_positive_rate": (train_summary["positive_rate"]),
                    "test_week_min": (test_summary["week_min"]),
                    "test_week_max": (test_summary["week_max"]),
                    "case_id_overlap": (integrity["case_id_overlap"]),
                },
                "outer_holdout": {
                    "strategy": ("contiguous_trailing_weeks"),
                    "candidate": (candidate["name"]),
                    "development_week_min": (development_week_min),
                    "development_week_max": (development_week_max),
                    "validation_week_min": (outer_week_min),
                    "validation_week_max": (outer_week_max),
                    "development_weeks": (candidate["train_weeks"]),
                    "validation_weeks": (candidate["validation_weeks"]),
                    "development_rows": (candidate["train_rows"]),
                    "validation_rows": (candidate["validation_rows"]),
                    "validation_row_fraction": (
                        _as_float(
                            candidate,
                            "validation_row_fraction",
                        )
                    ),
                    "development_positive_rate": (
                        _as_float(
                            candidate,
                            "train_positive_rate",
                        )
                    ),
                    "validation_positive_rate": (
                        _as_float(
                            candidate,
                            "validation_positive_rate",
                        )
                    ),
                    "target_rate_shift": (
                        _as_float(
                            candidate,
                            "target_rate_shift",
                        )
                    ),
                    "metric_eligible_weeks": (candidate["validation_metric_eligible_weeks"]),
                    "min_positives_per_week": (candidate["min_validation_positives_per_week"]),
                    "min_negatives_per_week": (candidate["min_validation_negatives_per_week"]),
                    "locked": True,
                },
                "inner_temporal_cv": {
                    "strategy": ("expanding_window"),
                    "folds": fold_payload(folds),
                    "n_splits": len(folds),
                    "validation_window_weeks": 8,
                    "group_column": "WEEK_NUM",
                    "target_column": "target",
                    "case_id_column": "case_id",
                    "shuffle": False,
                },
                "metric": {
                    "name": ("home_credit_stability"),
                    "weekly_statistic": ("normalized_gini"),
                    "slope_penalty": 88.0,
                    "residual_penalty": 0.5,
                    "require_both_classes_per_week": (True),
                },
                "selection_policy": {
                    "primary_model_selection": ("mean_inner_stability_score"),
                    "secondary_model_selection": [
                        ("worst_fold_stability_score"),
                        "mean_weekly_gini",
                        "temporal_slope",
                        "residual_std",
                        "brier_score",
                    ],
                    "outer_holdout_usage": (
                        "single locked evaluation after model-family and tuning decisions"
                    ),
                    "outer_holdout_reuse_for_tuning": (False),
                },
                "rationale": (
                    "The 20% trailing-week candidate "
                    "provides 19 fully metric-eligible "
                    "future weeks while retaining 73 "
                    "weeks for development. Five "
                    "deterministic expanding-window "
                    "folds cover the final 40 "
                    "development weeks in 8-week "
                    "validation blocks without "
                    "future-to-past leakage."
                ),
            }

            frozen = attach_protocol_sha256(payload)

            if not verify_protocol_sha256(frozen):
                raise RuntimeError("protocol fingerprint verification failed")

            write_protocol(
                frozen,
                args.output,
            )

    elapsed = round(
        time.monotonic() - started,
        3,
    )

    logger.event(
        "validation_protocol_completed",
        protocol_sha256=(frozen["protocol_sha256"]),
        outer_validation_week_min=(outer_week_min),
        outer_validation_week_max=(outer_week_max),
        inner_folds=len(folds),
        total_elapsed_seconds=elapsed,
        output=args.output.as_posix(),
    )

    print(f"PROTOCOL_SHA256={frozen['protocol_sha256']}")

    print(f"OUTER_HOLDOUT_WEEKS={outer_week_min}-{outer_week_max}")

    print(f"INNER_TEMPORAL_FOLDS={len(folds)}")

    print("VALIDATION_PROTOCOL_FROZEN")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
