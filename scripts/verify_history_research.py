#!/usr/bin/env python3
"""Replay saved history models and independently recompute every development metric."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from home_credit.modeling.acceptance import require
from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.data import FeatureRef, load_feature_frame
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.history_research import build_history_snapshot, validate_plan
from home_credit.modeling.release import read_object, save_json, transform, verify_file
from home_credit.modeling.release_workflow import ReleaseLogger, restore_member, restore_snapshot
from home_credit.observability.runtime import StageTimer


def independent_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Use pandas/sklearn/polyfit rather than the training stability implementation."""
    ginis = np.asarray(
        [
            2 * roc_auc_score(part.target, part.prediction) - 1
            for _, part in frame.groupby("WEEK_NUM", sort=True)
        ]
    )
    x = np.arange(len(ginis))
    slope, intercept = np.polyfit(x, ginis, 1)
    residual = float(np.std(ginis - (intercept + slope * x)))
    return {
        "stability_score": float(ginis.mean() + 88 * min(slope, 0) - 0.5 * residual),
        "mean_gini": float(ginis.mean()),
        "temporal_slope": float(slope),
        "residual_std": residual,
        "auc": float(roc_auc_score(frame.target, frame.prediction)),
        "pr_auc": float(average_precision_score(frame.target, frame.prediction)),
        "brier_score": float(brier_score_loss(frame.target, frame.prediction)),
        "log_loss": float(log_loss(frame.target, np.clip(frame.prediction, 1e-7, 1 - 1e-7))),
    }


def verify(root: Path, bucket: str, study: Path) -> dict[str, Any]:
    plan = read_object(root / "configs/history_research.json")
    protocol = read_object(root / "configs/validation_protocol.json")
    validate_plan(plan, protocol)
    state = read_object(study / "study.json")
    verify_file(root / "uv.lock", state["identity"]["lock_sha256"])
    logger = ReleaseLogger("verify-history", root / "logs")
    store = ExperimentStore(
        boto3.client("s3", region_name="us-west-2"),
        bucket,
        f"home-credit-model-stability/history-research/{study.name}",
        study,
        logger,
    )
    authoritative = store.restore(state["identity"])
    require(authoritative == state and state["complete"], "incomplete or changed durable study")
    report = read_object(restore_member(store, state["stages"]["report"]))
    require(report["identity"] == state["identity"], "report identity changed")
    require(len(state["trials"]) == 10, "incomplete trial count")
    require(
        {(t["experiment"], t["fold"]) for t in state["trials"]}
        == {(name, fold) for name in plan["experiments"] for fold in range(1, 6)},
        "incomplete grid",
    )
    verified_members: set[str] = set()

    def restore_all(value: Any) -> None:
        if isinstance(value, dict):
            if {"path", "key", "sha256", "bytes"} <= set(value):
                restore_member(store, value)
                verified_members.add(value["key"])
            else:
                for child in value.values():
                    restore_all(child)
        elif isinstance(value, list):
            for child in value:
                restore_all(child)

    restore_all(state)
    with WriterLease(store) as lease:
        snapshot = restore_snapshot(root, plan, store)
        view, _, _ = build_history_snapshot(snapshot, plan, protocol, store, lease, state)
    control_path = study / "control.parquet"
    store.download(plan["control"]["object_key"], control_path, plan["control"]["sha256"])
    reference = pd.read_parquet(control_path).sort_values("case_id").reset_index(drop=True)
    require(
        len(reference) == 727187 and set(reference.WEEK_NUM) == set(range(33, 73)),
        "control population changed",
    )
    metadata = ["case_id", "WEEK_NUM", "target"]
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in plan["experiments"]}
    maximum_replay_error = 0.0
    replayed = 0
    for trial in state["trials"]:
        fold = next(f for f in protocol["inner_temporal_cv"]["folds"] if f["fold"] == trial["fold"])
        with StageTimer(
            logger, f"replay_{trial['experiment']}_{trial['fold']}", heartbeat_seconds=15
        ):
            artifacts = trial["artifacts"]
            expected = (
                pd.read_parquet(restore_member(store, artifacts["predictions"]))
                .sort_values("case_id")
                .reset_index(drop=True)
            )
            subset = reference.loc[
                reference.WEEK_NUM.between(fold["validation_week_min"], fold["validation_week_max"])
            ]
            require(
                expected.case_id.is_unique
                and np.array_equal(expected[metadata].to_numpy(), subset[metadata].to_numpy()),
                "case/target/week alignment changed",
            )
            require(
                bool(
                    np.isfinite(expected.prediction).all()
                    and expected.prediction.between(0, 1).all()
                ),
                "invalid probability",
            )
            feature_payload = read_object(restore_member(store, artifacts["features"]))
            require(not feature_payload["hypotheses"], "unexpected case-level transforms")
            features = tuple(FeatureRef(**f) for f in feature_payload["original"])
            validation = load_feature_frame(
                view,
                features,
                week_min=fold["validation_week_min"],
                week_max=fold["validation_week_max"],
                max_rows=None,
                seed=plan["seed"],
            )
            validation = validation.sort("case_id")
            require(
                np.array_equal(
                    validation.select(metadata).to_numpy(), expected[metadata].to_numpy()
                ),
                "model replay population changed",
            )
            encoder = read_object(restore_member(store, artifacts["encoder"]))
            model = lgb.Booster(model_file=str(restore_member(store, artifacts["model"])))
            actual = model.predict(
                transform(validation, features, encoder), num_threads=plan["threads"]
            )
            error = float(np.max(np.abs(actual - expected.prediction.to_numpy())))
            require(np.isfinite(error) and error <= 1e-12, "native model replay differs")
            maximum_replay_error = max(maximum_replay_error, error)
            replayed += len(expected)
            frames[trial["experiment"]].append(expected)
    pooled = {"control": reference, **{k: pd.concat(v) for k, v in frames.items()}}
    maximum_metric_error, checks = 0.0, 0

    def compare(actual: float, expected: float) -> None:
        nonlocal maximum_metric_error, checks
        error = abs(actual - expected)
        require(np.isfinite(error) and error <= 1e-12, "independent metric disagrees")
        maximum_metric_error = max(maximum_metric_error, error)
        checks += 1

    means: dict[str, float] = {}
    for name, frame in pooled.items():
        scores = []
        for fold in protocol["inner_temporal_cv"]["folds"]:
            part = frame.loc[
                frame.WEEK_NUM.between(fold["validation_week_min"], fold["validation_week_max"])
            ]
            actual_metrics = independent_metrics(part)
            expected_metrics = next(
                r for r in report["folds"] if r["experiment"] == name and r["fold"] == fold["fold"]
            )
            for metric, value in actual_metrics.items():
                compare(value, expected_metrics[metric])
            scores.append(actual_metrics["stability_score"])
        means[name] = float(np.mean(scores))
        row = next(r for r in report["rows"] if r["experiment"] == name)
        compare(means[name], row["mean_fold_stability"])
        compare(min(scores), row["worst_fold_stability"])
        metrics = independent_metrics(frame)
        for metric in ("auc", "pr_auc", "brier_score", "log_loss"):
            compare(metrics[metric], row[f"oof_{metric}"])
    for row in report["rows"]:
        compare(means[row["experiment"]] - means["control"], row["delta_vs_control"])
    result = {
        "schema_version": 1,
        "status": "passed",
        "study_key": study.name,
        "comparison_sha256": sha256_file(study / "comparison.json"),
        "identity": state["identity"],
        "verified_checkpoint_objects": len(verified_members),
        "native_models_replayed": 10,
        "predictions_replayed": replayed,
        "maximum_prediction_absolute_error": maximum_replay_error,
        "metric_identities_checked": checks,
        "maximum_metric_absolute_error": maximum_metric_error,
        "model_fits": 0,
        "holdout_accessed": False,
    }
    save_json(study / "verification.json", result)
    store.publish(study / "verification.json", "verification")
    logger.event("history_verification_passed", **result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.bucket, args.study)
