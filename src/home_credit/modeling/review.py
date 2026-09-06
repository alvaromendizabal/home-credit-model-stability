"""Reconstruct temporal diagnostics from hash-pinned aggregate acceptance evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.acceptance import compare_number, read_json, require, validate_predictions
from home_credit.modeling.checkpoints import sha256_file
from home_credit.validation.protocol import verify_protocol_sha256


def stability_components(weeks: list[int], ginis: list[float]) -> dict[str, float]:
    """Decompose the official formula without aggregating its nonlinear penalty."""
    require(len(weeks) == len(ginis) >= 2, "at least two aligned weeks are required")
    require(weeks == sorted(set(weeks)), "weeks must be unique and increasing")
    require(bool(np.isfinite(ginis).all()), "Gini values must be finite")
    require(all(-1 <= g <= 1 for g in ginis), "Gini values must be in [-1, 1]")
    slope, intercept = np.polyfit(weeks, ginis, deg=1)
    residual_std = float(np.std(np.asarray(ginis) - (slope * np.asarray(weeks) + intercept)))
    mean_gini = float(np.mean(ginis))
    slope_penalty = 88.0 * min(0.0, float(slope))
    residual_penalty = -0.5 * residual_std
    return {
        "mean_gini": mean_gini,
        "temporal_slope": float(slope),
        "residual_std": residual_std,
        "slope_penalty": slope_penalty,
        "residual_penalty": residual_penalty,
        "stability_score": mean_gini + slope_penalty + residual_penalty,
    }


def review_evidence(
    evidence: dict[str, Any], protocol: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Check fold/weekly consistency and return descriptive development diagnostics."""
    require(evidence["status"] == "accepted", "review requires accepted evidence")
    require(metrics["summary_sha256"] == evidence["summary_sha256"], "rescoring identity mismatch")
    require(verify_protocol_sha256(protocol), "protocol content hash mismatch")
    require(
        evidence["validation_protocol_sha256"] == protocol["protocol_sha256"],
        "protocol identity mismatch",
    )
    require(protocol["outer_holdout"]["locked"] is True, "holdout must remain locked")
    require(evidence["holdout_predictions_present"] is False, "holdout predictions are forbidden")
    require(
        protocol["metric"]["slope_penalty"] == 88.0
        and protocol["metric"]["residual_penalty"] == 0.5,
        "competition metric constants differ",
    )
    windows = protocol["inner_temporal_cv"]["folds"]
    models = evidence["models"]
    names = [m["model"] for m in models]
    require(len(names) == len(set(names)) == 4, "expected four unique models")
    require(
        sorted(r["model"] for r in metrics["models"]) == sorted(names),
        "rescored model coverage mismatch",
    )
    expected_folds = {(name, f["fold"]) for name in names for f in windows}
    require(
        len(metrics["folds"]) == len(expected_folds)
        and {(r["model"], r["fold"]) for r in metrics["folds"]} == expected_folds,
        "rescored fold coverage mismatch",
    )
    require(len(evidence["fold_metrics"]) == len(names) * len(windows), "fold count mismatch")
    expected_weeks = {
        w for f in windows for w in range(f["validation_week_min"], f["validation_week_max"] + 1)
    }
    require(
        len(evidence["weekly_metrics"]) == len(names) * len(expected_weeks), "week count mismatch"
    )
    fold_diagnostics, comparisons = [], []
    population: list[dict[str, Any]] = []
    for model in models:
        name = model["model"]
        weekly = sorted(
            [w for w in evidence["weekly_metrics"] if w["model"] == name],
            key=lambda w: w["week_num"],
        )
        require([w["week_num"] for w in weekly] == sorted(expected_weeks), "week coverage mismatch")
        counts = [
            {k: w[k] for k in ("week_num", "rows", "positives", "positive_rate")} for w in weekly
        ]
        if population:
            require(counts == population, "cross-model population mismatch")
        population = counts
        for w in weekly:
            require(0 < w["positives"] < w["rows"], "single-class or empty week")
            compare_number(w["positive_rate"], w["positives"] / w["rows"], "weekly positive rate")
        current = []
        for window in windows:
            fold = window["fold"]
            records = [
                r for r in evidence["fold_metrics"] if r["model"] == name and r["fold"] == fold
            ]
            require(len(records) == 1, "missing or duplicate fold")
            recorded = records[0]
            subset = [
                w
                for w in weekly
                if window["validation_week_min"] <= w["week_num"] <= window["validation_week_max"]
            ]
            components = stability_components(
                [w["week_num"] for w in subset], [w["gini"] for w in subset]
            )
            rescored = next(r for r in metrics["folds"] if r["model"] == name and r["fold"] == fold)
            for key in ("mean_gini", "temporal_slope", "residual_std", "stability_score"):
                compare_number(components[key], rescored[key], f"{name}/fold_{fold}/{key}")
            archived_score = (
                recorded["mean_gini"]
                + 88 * min(0, recorded["temporal_slope"])
                - 0.5 * recorded["residual_std"]
            )
            compare_number(archived_score, recorded["stability_score"], "archived score components")
            rows = sum(w["rows"] for w in subset)
            positives = sum(w["positives"] for w in subset)
            require(rows == recorded["rows"], "fold population mismatch")
            current.append(
                {
                    "model": name,
                    "fold": fold,
                    "week_min": window["validation_week_min"],
                    "week_max": window["validation_week_max"],
                    "rows": rows,
                    "positives": positives,
                    "positive_rate": positives / rows,
                    "archived_stability_score": recorded["stability_score"],
                    **components,
                }
            )
        mean_score = float(np.mean([r["stability_score"] for r in current]))
        archived_mean = float(np.mean([r["archived_stability_score"] for r in current]))
        compare_number(archived_mean, model["mean_inner_stability_score"], f"{name}/archived mean")
        weakest = min(current, key=lambda r: r["stability_score"])
        pooled = stability_components([w["week_num"] for w in weekly], [w["gini"] for w in weekly])
        rescored_model = next(r for r in metrics["models"] if r["model"] == name)
        compare_number(
            pooled["stability_score"], rescored_model["stability_score"], f"{name}/pooled score"
        )
        comparisons.append(
            {
                "model": name,
                "archived_rank": model["rank"],
                "mean_fold_stability": mean_score,
                "archived_mean_fold_stability": archived_mean,
                **{
                    f"oof_{k}": rescored_model[k]
                    for k in ("stability_score", "auc", "pr_auc", "brier_score", "log_loss")
                },
                "weakest_fold": weakest["fold"],
                "mean_gini": float(np.mean([r["mean_gini"] for r in current])),
                "mean_slope_penalty": float(np.mean([r["slope_penalty"] for r in current])),
                "mean_residual_penalty": float(np.mean([r["residual_penalty"] for r in current])),
            }
        )
        fold_diagnostics.extend(current)
    comparisons.sort(key=lambda r: (-r["mean_fold_stability"], r["model"]))
    for rank, row in enumerate(comparisons, 1):
        row["rank"] = rank
    return {
        "schema_version": 1,
        "scope": "accepted development aggregates; no training or holdout evaluation",
        "run_key": evidence["run_key"],
        "summary_sha256": evidence["summary_sha256"],
        "leader": comparisons[0]["model"],
        "models": comparisons,
        "folds": fold_diagnostics,
        "weekly_population": population,
        "screening_drift_auc": evidence["screening_drift_auc"],
        "selected_features_by_block": evidence["selected_features_by_block"],
    }


def load_review(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail before notebook execution if the reviewed aggregate snapshot has changed."""
    policy = read_json(root / "configs/benchmark_review.json")
    path = root / "reports/benchmark/acceptance.json"
    require(sha256_file(path) == policy["evidence_sha256"], "review evidence hash mismatch")
    evidence = read_json(path)
    require(evidence["summary_sha256"] == policy["summary_sha256"], "review summary mismatch")
    metric_path = root / "reports/benchmark/metrics.json"
    require(sha256_file(metric_path) == policy["metrics_sha256"], "review metrics hash mismatch")
    review = review_evidence(
        evidence, read_json(root / "configs/validation_protocol.json"), read_json(metric_path)
    )
    review["evidence_sha256"] = policy["evidence_sha256"]
    return evidence, review


def rescore_predictions(
    evidence: dict[str, Any], protocol: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    """Evaluate verified saved OOF predictions; never deserialize or fit a model."""
    require(evidence["status"] == "accepted", "accepted evidence required")
    require(verify_protocol_sha256(protocol), "protocol content hash mismatch")
    require(protocol["outer_holdout"]["locked"] is True, "holdout must remain locked")
    require(
        evidence["validation_protocol_sha256"] == protocol["protocol_sha256"],
        "protocol identity mismatch",
    )
    windows = protocol["inner_temporal_cv"]["folds"]
    weeks = {
        w for f in windows for w in range(f["validation_week_min"], f["validation_week_max"] + 1)
    }
    model_results, fold_results = [], []
    for model in evidence["models"]:
        path = (bundle / model["oof_path"]).resolve()
        require(path.is_relative_to(bundle.resolve()), "prediction path escaped bundle")
        require(sha256_file(path) == model["oof_sha256"], "OOF prediction hash mismatch")
        frame = pl.read_parquet(path)
        validate_predictions(frame, weeks, model["model"])

        def evaluate(part: pl.DataFrame) -> dict[str, float]:
            return evaluate_probabilities(
                part["target"].to_numpy(),
                part["prediction"].to_numpy(),
                part["WEEK_NUM"].to_numpy(),
            )

        model_results.append(
            {"model": model["model"], "oof_sha256": model["oof_sha256"], **evaluate(frame)}
        )
        for window in windows:
            subset = frame.filter(
                pl.col("WEEK_NUM").is_between(
                    window["validation_week_min"], window["validation_week_max"]
                )
            )
            fold_results.append(
                {"model": model["model"], "fold": window["fold"], **evaluate(subset)}
            )
    return {
        "schema_version": 1,
        "summary_sha256": evidence["summary_sha256"],
        "ranking_policy": "unclipped prediction ranks for Gini, ROC AUC, and AP",
        "probability_policy": "raw Brier; log loss clips to [1e-7, 1-1e-7]",
        "models": model_results,
        "folds": fold_results,
    }


def verify_rescored_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Verify identities exactly and floating point metrics within acceptance tolerance."""
    for key in ("schema_version", "summary_sha256", "ranking_policy", "probability_policy"):
        require(actual[key] == expected[key], f"rescoring metadata mismatch: {key}")
    for group, identity_keys in (("models", ("model",)), ("folds", ("model", "fold"))):

        def indexed(
            rows: list[dict[str, Any]], keys: tuple[str, ...] = identity_keys
        ) -> dict[tuple[Any, ...], dict[str, Any]]:
            result = {tuple(row[k] for k in keys): row for row in rows}
            require(len(result) == len(rows), "duplicate rescored metric identity")
            return result

        left, right = indexed(actual[group]), indexed(expected[group])
        require(left.keys() == right.keys(), "rescored metric coverage mismatch")
        for identity, row in left.items():
            require(row.keys() == right[identity].keys(), "rescored metric fields mismatch")
            for key, value in row.items():
                if key in (*identity_keys, "oof_sha256"):
                    require(value == right[identity][key], "rescored prediction identity mismatch")
                else:
                    compare_number(value, right[identity][key], f"{identity}/{key}")
