#!/usr/bin/env python3
"""Independently replay published calibration parameters; never fit a model.

Inputs are the two original OOF files and twelve restored prediction checkpoints.
Their byte counts and SHA-256 identities are pinned by the committed evidence.
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


def same_population(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return bool(
        left.shape == right.shape
        and list(left.columns) == list(right.columns)
        and all(dtype.kind in "iu" for dtype in [*left.dtypes, *right.dtypes])
        and np.array_equal(left.to_numpy(), right.to_numpy())
    )


def checked(path: Path, member: dict[str, Any]) -> pd.DataFrame:
    raw = path.read_bytes()
    if len(raw) != member["bytes"] or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise ValueError(f"Unverified input: {path.name}")
    return pd.read_parquet(path).sort_values("case_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y, p = frame.target.to_numpy(), frame.prediction.to_numpy()
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
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def verify(root: Path, inputs: Path, predictions: Path) -> dict[str, Any]:
    plan = json.loads((root / "configs/calibration_research.json").read_text())
    policy = json.loads((root / "configs/calibration_review.json").read_text())
    raw = (root / "reports/calibration/comparison.json").read_bytes()
    if len(raw) != policy["bytes"] or hashlib.sha256(raw).hexdigest() != policy["sha256"]:
        raise ValueError("Unverified calibration report")
    report = json.loads(raw)
    frames = {
        name: checked(inputs / f"{name}.parquet", member)
        for name, member in plan["sources"].items()
    }
    metadata = ["case_id", "target", "WEEK_NUM", "fold"]
    if not same_population(frames["original"][metadata], frames["tuned"][metadata]):
        raise ValueError("Original OOF population alignment changed")
    reference = frames["original"]
    blend = 0.9 * frames["tuned"].prediction + 0.1 * reference.prediction
    replay_error, metric_error = 0.0, 0.0
    restored: dict[str, list[pd.DataFrame]] = {}
    for record in report["folds"]:
        actual = checked(predictions / record["predictions"]["path"], record["predictions"])
        selected = reference.fold == record["fold"]
        expected_meta = reference.loc[selected, metadata[:-1]].reset_index(drop=True)
        if not same_population(actual[metadata[:-1]], expected_meta):
            raise ValueError("Calibration checkpoint population changed")
        p = blend[selected].to_numpy()
        model = record["model"]
        if record["method"] == "sigmoid":
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            z = model["coefficient"] * np.log(clipped / (1 - clipped)) + model["intercept"]
            p = 1 / (1 + np.exp(-z))
        elif record["method"] == "isotonic":
            p = np.interp(p, model["x"], model["y"])
        error = float(np.max(np.abs(actual.prediction.to_numpy() - p)))
        replay_error = max(replay_error, error)
        if error > 1e-14:
            raise ValueError("Independent parameter replay failed")
        for name, value in metrics(actual).items():
            error = abs(value - record["metrics"][name])
            metric_error = max(metric_error, error)
            if error > 1e-12:
                raise ValueError(f"Independent fold metric disagrees: {name}")
        restored.setdefault(record["method"], []).append(actual)
    for row in report["rows"]:
        frame = pd.concat(restored[row["method"]], ignore_index=True)
        if len(frame) != 544611 or not frame.case_id.is_unique:
            raise ValueError("Pooled calibration coverage changed")
        for name, value in metrics(frame).items():
            error = abs(value - row[f"pooled_{name}"])
            metric_error = max(metric_error, error)
            if error > 1e-12:
                raise ValueError(f"Independent pooled metric disagrees: {name}")
    return {
        "schema_version": 1,
        "verified_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": policy["sha256"],
        "python_version": platform.python_version(),
        "input_oof_files_verified": 2,
        "prediction_checkpoints_verified": len(report["folds"]),
        "evaluation_cases_per_method": 544611,
        "replayed_predictions": 3 * 544611,
        "maximum_prediction_absolute_error": replay_error,
        "maximum_metric_absolute_error": metric_error,
        "new_base_model_fits": 0,
        "new_calibrator_fits": 0,
        "holdout_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/calibration_inputs"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/calibration_verification")
    )
    args = parser.parse_args()
    receipt = verify(args.root, args.inputs, args.predictions)
    (args.root / "reports/calibration/verification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
