#!/usr/bin/env python3
"""Independently recompute the feature study from verified saved predictions.

Uses pandas, sklearn and numpy.polyfit instead of the training comparison and
project stability implementation. No model fitting or holdout input is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.observability.runtime import StageTimer


def check_bytes(path: Path, member: dict[str, Any]) -> None:
    raw = path.read_bytes()
    if len(raw) != member["bytes"] or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise ValueError(f"Unverified checkpoint: {path.name}")


def checked(path: Path, member: dict[str, Any]) -> pd.DataFrame:
    check_bytes(path, member)
    frame = pd.read_parquet(path).sort_values("case_id").reset_index(drop=True)
    metadata = frame[["case_id", "WEEK_NUM", "target"]]
    if (
        not frame.case_id.is_unique
        or metadata.isna().any().any()
        or not all(dtype.kind in "iu" for dtype in metadata.dtypes)
        or set(frame.target.unique()) != {0, 1}
        or not np.isfinite(frame.prediction).all()
        or not frame.prediction.between(0, 1).all()
    ):
        raise ValueError("Invalid prediction population or probabilities")
    return frame


def independent_metrics(frame: pd.DataFrame) -> dict[str, float]:
    weekly = np.asarray(
        [
            2 * roc_auc_score(part.target, part.prediction) - 1
            for _, part in frame.groupby("WEEK_NUM", sort=True)
        ]
    )
    x = np.arange(len(weekly))
    slope, intercept = np.polyfit(x, weekly, 1)
    residual = float(np.std(weekly - (intercept + slope * x)))
    return {
        "stability_score": float(weekly.mean() + 88 * min(slope, 0) - 0.5 * residual),
        "mean_gini": float(weekly.mean()),
        "temporal_slope": float(slope),
        "residual_std": residual,
        "auc": float(roc_auc_score(frame.target, frame.prediction)),
        "pr_auc": float(average_precision_score(frame.target, frame.prediction)),
        "brier_score": float(brier_score_loss(frame.target, frame.prediction)),
        "log_loss": float(log_loss(frame.target, np.clip(frame.prediction, 1e-7, 1 - 1e-7))),
    }


def verify(root: Path, predictions: Path) -> dict[str, Any]:
    policy = json.loads((root / "configs/feature_research_review.json").read_text())
    plan = json.loads((root / "configs/feature_research.json").read_text())
    path = root / "reports/feature_research/comparison.json"
    check_bytes(path, policy["files"]["comparison.json"])
    report = json.loads(path.read_text())
    check_bytes(root / "reports/feature_research/screen.json", report["screen"]["screen"])
    check_bytes(predictions / "screen_peer_references.json", report["screen"]["peer_references"])
    grid = {(name, fold) for name in plan["experiments"] for fold in range(1, 6)}
    if (
        report["new_model_fits"] != 20
        or len(report["fit_records"]) != 20
        or {(r["experiment"], r["fold"]) for r in report["fit_records"]} != grid
    ):
        raise ValueError("Incomplete feature study")
    folds = json.loads((root / "configs/validation_protocol.json").read_text())[
        "inner_temporal_cv"
    ]["folds"]
    reference = checked(predictions / "control.parquet", plan["control"])
    if len(reference) != 727187 or set(reference.WEEK_NUM) != set(range(33, 73)):
        raise ValueError("Reference development population changed")
    metadata = ["case_id", "WEEK_NUM", "target"]
    frames = {"control": reference}
    for name in plan["experiments"]:
        parts = []
        for record in report["fit_records"]:
            if record["experiment"] != name:
                continue
            member = record["artifacts"]["predictions"]
            relative = Path(member["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe checkpoint path")
            frame = checked(predictions / relative, member)
            fold = next(f for f in folds if f["fold"] == record["fold"])
            expected = reference.loc[
                reference.WEEK_NUM.between(fold["validation_week_min"], fold["validation_week_max"])
            ]
            if not np.array_equal(frame[metadata].to_numpy(), expected[metadata].to_numpy()):
                raise ValueError("Checkpoint rows, targets or temporal assignment changed")
            parts.append(frame)
        frames[name] = pd.concat(parts).sort_values("case_id").reset_index(drop=True)
        if not np.array_equal(frames[name][metadata].to_numpy(), reference[metadata].to_numpy()):
            raise ValueError("Pooled population changed")
    maximum_error, comparisons = 0.0, 0

    def compare(actual: float, expected: float) -> None:
        nonlocal maximum_error, comparisons
        error = abs(actual - expected)
        if not np.isfinite(actual) or not np.isfinite(expected) or error > 1e-12:
            raise ValueError("Independent feature metric disagrees")
        maximum_error = max(maximum_error, error)
        comparisons += 1

    means = {}
    for name, frame in frames.items():
        scores = []
        for fold in folds:
            part = frame.loc[
                frame.WEEK_NUM.between(fold["validation_week_min"], fold["validation_week_max"])
            ]
            actual = independent_metrics(part)
            expected = next(
                r for r in report["folds"] if r["experiment"] == name and r["fold"] == fold["fold"]
            )
            for metric, value in actual.items():
                compare(value, expected[metric])
            scores.append(actual["stability_score"])
        means[name] = float(np.mean(scores))
        row = next(r for r in report["rows"] if r["experiment"] == name)
        compare(means[name], row["mean_fold_stability"])
        compare(min(scores), row["worst_fold_stability"])
        pooled = independent_metrics(frame)
        for metric in ("auc", "pr_auc", "brier_score", "log_loss"):
            compare(pooled[metric], row[f"oof_{metric}"])
    for row in report["rows"]:
        compare(means[row["experiment"]] - means["control"], row["delta_vs_control"])
    return {
        "schema_version": 1,
        "verified_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": policy["files"]["comparison.json"]["sha256"],
        "python_version": platform.python_version(),
        "prediction_files_verified": 21,
        "screen_checkpoints_verified": 2,
        "model_fold_comparisons": 25,
        "evaluation_cases_per_condition": len(reference),
        "predictions_verified": sum(len(frame) for frame in frames.values()),
        "metric_comparisons": comparisons,
        "maximum_metric_absolute_error": maximum_error,
        "new_model_fits": 0,
        "holdout_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/feature_research_verification")
    )
    args = parser.parse_args()
    logger = ReleaseLogger("feature-research-verification", args.root / "logs")
    with StageTimer(logger, "independent_prediction_verification", heartbeat_seconds=15):
        receipt = verify(args.root, args.predictions)
    (args.root / "reports/feature_research/verification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    logger.event("feature_research_independently_verified", **receipt)
