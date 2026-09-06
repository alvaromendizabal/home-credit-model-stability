"""Independent artifact acceptance for a completed temporal development benchmark.

This module never trains or loads a serialized model. It verifies its bytes and
recomputes metrics from predictions. Raw feature lineage and final generalization
are separate audits; neither is inferred from successful artifact acceptance.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from home_credit.metrics.stability import stability_score
from home_credit.modeling.checkpoints import (
    derive_run_key,
    load_manifest_bytes,
    object_key_for_sha,
    sha256_file,
    validate_benchmark_state,
    verify_checkpoint_manifest,
)
from home_credit.modeling.config import BenchmarkConfig
from home_credit.observability.logging import RunLogger
from home_credit.validation.protocol import (
    TemporalFold,
    validate_expanding_folds,
    verify_protocol_sha256,
)


def require(condition: bool, message: str) -> None:
    """Raise a durable acceptance failure (also under optimized Python)."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    """Read an object, rejecting ambiguous top-level data."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return dict(value)


def compare_number(actual: float, expected: float, label: str) -> None:
    """Compare persisted metrics with a tight, explicit floating point tolerance."""
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10),
        f"metric mismatch: {label}: computed={actual} stored={expected}",
    )


def validate_predictions(frame: pl.DataFrame, weeks: set[int], label: str) -> None:
    """Reject duplicate, incomplete, nonfinite, or out-of-window predictions."""
    columns = ("case_id", "WEEK_NUM", "target", "prediction")
    require(set(columns).issubset(frame.columns), f"missing prediction columns: {label}")
    require(frame.height > 0, f"empty predictions: {label}")
    for col in columns:
        require(frame[col].null_count() == 0, f"null {col}: {label}")
    for col in ("case_id", "WEEK_NUM", "target"):
        require(frame.schema[col].is_integer(), f"noninteger {col}: {label}")
    require(frame["case_id"].n_unique() == frame.height, f"duplicate case_id: {label}")
    require(set(frame["WEEK_NUM"].unique().to_list()) == weeks, f"week coverage mismatch: {label}")
    require(set(frame["target"].unique().to_list()) == {0, 1}, f"nonbinary target: {label}")
    prediction = frame["prediction"].to_numpy()
    require(bool(np.isfinite(prediction).all()), f"nonfinite prediction: {label}")
    require(bool(((prediction >= 0) & (prediction <= 1)).all()), f"invalid probability: {label}")
    counts = frame.group_by("WEEK_NUM").agg(pl.col("target").n_unique().alias("classes"))
    require(bool((counts["classes"] == 2).all()), f"single-class week: {label}")


def prediction_metrics(frame: pl.DataFrame) -> dict[str, float]:
    """Verify the immutable benchmark's historical all-metric clipping policy.

    This is an archival compatibility check. New evaluation uses
    metrics.classification.evaluate_probabilities, which preserves raw ranks.
    """
    target = frame["target"].to_numpy()
    prediction = np.clip(frame["prediction"].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
    result = stability_score(target, prediction, frame["WEEK_NUM"].to_numpy())
    return {
        "stability_score": result.score,
        "mean_gini": result.mean_gini,
        "temporal_slope": result.slope,
        "residual_std": result.residual_std,
        "auc": float(roc_auc_score(target, prediction)),
        "pr_auc": float(average_precision_score(target, prediction)),
        "brier_score": float(brier_score_loss(target, prediction)),
        "log_loss": float(log_loss(target, prediction, labels=[0, 1])),
    }


def indexed_jsonl(path: Path, keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Read metric records, rejecting duplicate keys."""
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = tuple(row[k] for k in keys)
        require(key not in result, f"duplicate metric record: {path.name} {key}")
        result[key] = row
    return result


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Apply the predeclared mean-fold objective and ordered tie breakers."""
    return (
        -float(row["mean_inner_stability_score"]),
        -float(row["worst_fold_stability_score"]),
        -float(row["mean_weekly_gini"]),
        -float(row["mean_temporal_slope"]),
        float(row["mean_residual_std"]),
        float(row["mean_brier_score"]),
    )


def accept_benchmark(
    root: Path,
    *,
    policy: dict[str, Any],
    config_path: Path,
    protocol_path: Path,
    logger: RunLogger,
) -> dict[str, Any]:
    """Verify the complete published bundle and return aggregate-only evidence."""
    manifest_path = root / "checkpoint_manifest.json"
    require(sha256_file(manifest_path) == policy["manifest_sha256"], "manifest hash mismatch")
    manifest = load_manifest_bytes(manifest_path.read_bytes())
    require(manifest.run_key == policy["run_key"], "run key mismatch")
    require(manifest.smoke is False, "smoke run is not acceptable")
    require(
        sha256_file(root / "benchmark_summary.json") == policy["summary_sha256"],
        "summary hash mismatch",
    )
    verify_checkpoint_manifest(root, manifest)
    logger.event("artifact_hashes_verified", files=len(manifest.files))

    config, config_sha = BenchmarkConfig.load(config_path)
    require(config_sha == manifest.benchmark_config_sha256, "benchmark config hash mismatch")
    protocol = read_json(protocol_path)
    require(verify_protocol_sha256(protocol), "protocol content hash mismatch")
    require(
        protocol["protocol_sha256"] == manifest.validation_protocol_sha256,
        "protocol identity mismatch",
    )
    outer = protocol["outer_holdout"]
    require(outer["locked"] is True, "outer holdout is unlocked")
    require(
        outer["validation_week_min"] == config.outer_holdout_guard_week_min,
        "outer holdout boundary mismatch",
    )
    folds = tuple(TemporalFold(**row) for row in protocol["inner_temporal_cv"]["folds"])
    validate_expanding_folds(
        folds,
        development_week_min=outer["development_week_min"],
        development_week_max=outer["development_week_max"],
    )
    require(len({fold.fold for fold in folds}) == len(folds), "duplicate fold numbers")
    require(folds[-1].validation_week_max < outer["validation_week_min"], "holdout overlap")
    require(
        config.screening.validation_week_max < folds[0].validation_week_min,
        "feature screening overlaps evaluation",
    )
    require(
        protocol["selection_policy"]["primary_model_selection"] == "mean_inner_stability_score",
        "unsupported model selection policy",
    )
    require(
        protocol["metric"]["slope_penalty"] == 88.0
        and protocol["metric"]["residual_penalty"] == 0.5,
        "unsupported metric policy",
    )
    identity: dict[str, Any] = {
        "git_commit": manifest.git_commit,
        "feature_manifest_sha256": manifest.feature_manifest_sha256,
        "validation_protocol_sha256": manifest.validation_protocol_sha256,
        "benchmark_config_sha256": manifest.benchmark_config_sha256,
        "smoke": False,
    }
    require(derive_run_key(**identity) == manifest.run_key, "derived run identity mismatch")
    count = validate_benchmark_state(root, **identity)
    names = config.enabled_model_names
    expected = {(name, f.fold) for name in names for f in folds}
    require(
        count == len(expected) == manifest.completed_model_folds == manifest.sequence,
        "incomplete model-fold count",
    )
    summary = read_json(root / "benchmark_summary.json")
    metadata = read_json(root / "run_metadata.json")
    state = read_json(root / "benchmark_state.json")
    screen = read_json(root / "feature_screen.json")
    for obj in (summary, metadata, state["identity"], screen["identity"]):
        for key, value in identity.items():
            require(obj[key] == value, f"provenance mismatch: {key}")
    for obj in (summary, metadata):
        require(obj["outer_holdout_touched"] is False, "holdout use reported")
        require(
            obj["feature_screen_sha256"] == sha256_file(root / "feature_screen.json"),
            "feature screen identity mismatch",
        )
    for name in ("benchmark_summary", "benchmark_state", "fold_metrics", "weekly_metrics"):
        ext = ".jsonl" if name.endswith("metrics") else ".json"
        require(
            metadata[f"{name}_sha256"] == sha256_file(root / f"{name}{ext}"),
            f"metadata hash mismatch: {name}",
        )
    require(summary["folds"] == len(folds), "summary fold count mismatch")
    require(
        summary["selection_policy"]["primary"] == "mean_inner_stability_score",
        "summary selection policy mismatch",
    )
    selected = screen["result"]["selected_features"]
    require(
        len(selected) == len({f["name"] for f in selected}) == summary["selected_features"],
        "selected feature count mismatch",
    )
    require(
        not {f["name"] for f in selected}.intersection(config.excluded_predictors),
        "excluded predictor selected",
    )
    require(
        dict(Counter(f["block"] for f in selected)) == summary["selected_features_by_block"],
        "feature block count mismatch",
    )
    require(
        sum(f["categorical"] for f in selected) == summary["categorical_features"],
        "categorical feature count mismatch",
    )

    members = {member.path: member for member in manifest.files}
    for path in (
        "benchmark_state.json",
        "benchmark_summary.json",
        "feature_screen.json",
        "fold_metrics.jsonl",
        "weekly_metrics.jsonl",
        "run_metadata.json",
    ):
        require(path in members, f"required file missing from manifest: {path}")
    prefix = str(policy["s3_prefix"]).strip("/")
    for member in manifest.files:
        require(
            member.object_key == object_key_for_sha(prefix, member.sha256),
            f"object key mismatch: {member.path}",
        )
    receipts = {(r["model"], r["fold"]): r for r in state["folds"].values()}
    require(set(receipts) == expected, "model-fold identities mismatch")
    feature_counts = {}
    for name in names:
        expected_width = min(
            summary["selected_features"],
            int(config.model(name).params.get("max_features", summary["selected_features"])),
        )
        require(
            all(
                receipts[name, fold.fold]["metrics"]["features"] == expected_width for fold in folds
            ),
            f"model feature count mismatch: {name}",
        )
        feature_counts[name] = expected_width
    stored_folds = indexed_jsonl(root / "fold_metrics.jsonl", ("model", "fold"))
    stored_weeks = indexed_jsonl(root / "weekly_metrics.jsonl", ("model", "week_num"))
    require(set(stored_folds) == expected, "fold metric coverage mismatch")
    weeks = {
        week for f in folds for week in range(f.validation_week_min, f.validation_week_max + 1)
    }
    require(
        set(stored_weeks) == {(name, w) for name in names for w in weeks},
        "weekly metric coverage mismatch",
    )
    rows = {r["model"]: r for r in summary["models"]}
    require(
        len(rows) == len(summary["models"]) and set(rows) == set(names),
        "summary model coverage mismatch",
    )
    reference: pl.DataFrame | None = None
    fold_results: list[dict[str, Any]] = []
    weekly_results: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for name in names:
        frames = []
        for fold in folds:
            receipt = receipts[name, fold.fold]
            for field in ("prediction", "model"):
                path = receipt[f"{field}_path"]
                require(path in members, f"receipt missing from manifest: {path}")
                require(
                    members[path].sha256 == receipt[f"{field}_sha256"],
                    f"receipt hash mismatch: {path}",
                )
            frame = (
                pl.read_parquet(root / receipt["prediction_path"])
                .select("case_id", "WEEK_NUM", "target", "prediction")
                .sort("case_id")
            )
            validate_predictions(
                frame,
                set(range(fold.validation_week_min, fold.validation_week_max + 1)),
                f"{name} fold {fold.fold}",
            )
            computed = prediction_metrics(frame)
            for key, value in computed.items():
                compare_number(value, receipt["metrics"][key], f"{name}/{fold.fold}/{key}")
                compare_number(value, stored_folds[name, fold.fold][key], f"fold_metrics/{key}")
            compare_number(frame.height, receipt["metrics"]["validation_rows"], "validation rows")
            compare_number(
                float(frame["target"].to_numpy().mean()),
                receipt["metrics"]["validation_positive_rate"],
                "positive rate",
            )
            fold_results.append(
                {
                    "model": name,
                    "fold": fold.fold,
                    **computed,
                    "rows": frame.height,
                    "best_iteration": receipt["metrics"]["best_iteration"],
                }
            )
            frames.append(frame)
            logger.event(
                "model_fold_accepted",
                model=name,
                fold=fold.fold,
                rows=frame.height,
                completed=len(fold_results),
                total=len(expected),
            )
        combined = pl.concat(frames).sort("case_id")
        oof_path = rows[name]["oof_path"]
        require(
            oof_path in members and members[oof_path].sha256 == rows[name]["oof_sha256"],
            f"OOF manifest identity mismatch: {name}",
        )
        oof = (
            pl.read_parquet(root / oof_path)
            .select("case_id", "WEEK_NUM", "target", "prediction")
            .sort("case_id")
        )
        validate_predictions(oof, weeks, name)
        require(oof.equals(combined), f"OOF differs from fold predictions: {name}")
        ids = oof.select("case_id", "WEEK_NUM", "target")
        if reference is None:
            reference = ids
        else:
            require(ids.equals(reference), f"cross-model OOF alignment mismatch: {name}")
        for key, value in prediction_metrics(oof).items():
            compare_number(value, rows[name][f"oof_{key}"], f"{name}/oof_{key}")
        model_folds = [r for r in fold_results if r["model"] == name]
        values = np.asarray([r["stability_score"] for r in model_folds])
        aggregates = {
            "mean_inner_stability_score": float(values.mean()),
            "std_inner_stability_score": float(values.std()),
            "worst_fold_stability_score": float(values.min()),
        }
        for dest, source in (
            ("mean_weekly_gini", "mean_gini"),
            ("mean_auc", "auc"),
            ("mean_temporal_slope", "temporal_slope"),
            ("mean_residual_std", "residual_std"),
            ("mean_brier_score", "brier_score"),
        ):
            aggregates[dest] = float(np.mean([r[source] for r in model_folds]))
        for key, value in aggregates.items():
            compare_number(value, rows[name][key], f"{name}/{key}")
        for week in sorted(weeks):
            part = oof.filter(pl.col("WEEK_NUM") == week)
            y, p = part["target"].to_numpy(), part["prediction"].to_numpy()
            observed = {
                "rows": part.height,
                "positives": int(y.sum()),
                "positive_rate": float(y.mean()),
                "prediction_mean": float(p.mean()),
                "gini": float(2 * roc_auc_score(y, p) - 1),
            }
            for key, value in observed.items():
                compare_number(value, stored_weeks[name, week][key], f"{name}/week{week}/{key}")
            weekly_results.append({"model": name, "week_num": week, **observed})
        # Fixed probability bins support a like-for-like reliability comparison.
        p = oof["prediction"].to_numpy()
        y = oof["target"].to_numpy()
        edges = np.asarray([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0])
        bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
        for index in range(len(edges) - 1):
            mask = bins == index
            if mask.any():
                calibration.append(
                    {
                        "model": name,
                        "bin_low": float(edges[index]),
                        "bin_high": float(edges[index + 1]),
                        "rows": int(mask.sum()),
                        "prediction_mean": float(p[mask].mean()),
                        "observed_rate": float(y[mask].mean()),
                    }
                )
    ranked = sorted(rows.values(), key=ranking_key)
    require(
        [r["model"] for r in summary["models"]] == [r["model"] for r in ranked],
        "predeclared ranking mismatch",
    )
    require([r["rank"] for r in ranked] == list(range(1, len(ranked) + 1)), "rank labels mismatch")
    require(reference is not None, "no OOF rows")
    assert reference is not None  # narrowed after the runtime guard above
    prevalence = float(reference["target"].to_numpy().mean())
    fit_events: dict[str, float] = {}
    for member in manifest.files:
        if not member.path.startswith("logs/") or not member.path.endswith(".jsonl"):
            continue
        for line in (root / member.path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            stage = str(event.get("stage", ""))
            if event["event"] == "stage_completed" and stage.startswith("fit_"):
                require(stage not in fit_events, f"duplicate fit timing: {stage}")
                elapsed = float(event["elapsed_seconds"])
                require(math.isfinite(elapsed) and elapsed >= 0, f"invalid fit timing: {stage}")
                fit_events[stage] = elapsed
    require(
        set(fit_events) == {f"fit_{name}_fold_{fold}" for name, fold in expected},
        "fit timing coverage mismatch",
    )
    fit_seconds = {
        name: sum(fit_events[f"fit_{name}_fold_{f.fold}"] for f in folds) for name in names
    }
    return {
        "schema_version": 1,
        "status": "accepted",
        "scope": "development_benchmark_artifacts",
        "run_key": manifest.run_key,
        "training_commit": manifest.git_commit,
        "summary_sha256": policy["summary_sha256"],
        "manifest_sha256": policy["manifest_sha256"],
        "validation_protocol_sha256": manifest.validation_protocol_sha256,
        "verified_files": len(manifest.files),
        "verified_bytes": sum(m.bytes for m in manifest.files),
        "model_folds": len(expected),
        "oof_rows_per_model": reference.height,
        "oof_week_min": min(weeks),
        "oof_week_max": max(weeks),
        "holdout_week_min": outer["validation_week_min"],
        "holdout_week_max": outer["validation_week_max"],
        "holdout_predictions_present": False,
        "oof_positive_rate": prevalence,
        "descriptive_constant_brier": prevalence * (1 - prevalence),
        "selected_features": summary["selected_features"],
        "model_feature_counts": feature_counts,
        "categorical_features": summary["categorical_features"],
        "selected_features_by_block": summary["selected_features_by_block"],
        "screening_drift_auc": screen["result"]["drift_validation_auc"],
        "leader": ranked[0]["model"],
        "models": ranked,
        "fold_metrics": fold_results,
        "fit_seconds_by_model": fit_seconds,
        "weekly_metrics": weekly_results,
        "calibration": calibration,
        "limitations": [
            (
                "Development folds were used for early stopping and model selection; scores are "
                "not an unbiased final test."
            ),
            (
                "Pooled OOF stability spans predictions from five refitted models; mean fold "
                "stability remains the selection objective."
            ),
            (
                "Artifact acceptance verifies recorded predictions and provenance, not raw-data "
                "point-in-time correctness or every historical process."
            ),
            (
                "The adversarial screening AUC is a shift diagnostic from the screening sample; "
                "it is not the credit-risk model AUC."
            ),
            (
                "The constant Brier reference uses OOF prevalence descriptively and is not a "
                "trained baseline."
            ),
            (
                "Final holdout evaluation, model calibration, subgroup robustness, and "
                "deployment readiness remain pending."
            ),
        ],
    }
