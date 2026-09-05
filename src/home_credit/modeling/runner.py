"""End-to-end leakage-safe temporal benchmark orchestration."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from home_credit.metrics.stability import normalized_gini, stability_score
from home_credit.modeling.config import BenchmarkConfig, ModelConfig
from home_credit.modeling.data import (
    CASE_ID,
    TARGET,
    WEEK_NUM,
    FeatureRef,
    FeatureSnapshot,
    load_feature_frame,
    load_fold_frames,
)
from home_credit.modeling.encoding import catboost_encode, frequency_encode, standardize_for_linear
from home_credit.modeling.models import (
    ModelFitResult,
    fit_catboost,
    fit_lightgbm,
    fit_linear_logistic,
    fit_xgboost,
)
from home_credit.modeling.screening import feature_refs_from_payload, screen_features
from home_credit.modeling.state import (
    BenchmarkIdentity,
    BenchmarkStateStore,
    FoldReceipt,
    relative_artifact,
    sha256_file,
)
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


class BenchmarkRunner:
    """Run feature screening and identical frozen temporal CV across model families."""

    def __init__(
        self,
        *,
        feature_dir: Path,
        expected_feature_manifest_sha256: str,
        protocol_path: Path,
        expected_protocol_sha256: str,
        config_path: Path,
        expected_config_sha256: str,
        output_dir: Path,
        logs_dir: Path,
        smoke: bool,
        max_new_checkpoints: int | None = None,
    ) -> None:
        self.feature_dir = feature_dir
        self.expected_feature_manifest_sha256 = expected_feature_manifest_sha256
        self.protocol_path = protocol_path
        self.expected_protocol_sha256 = expected_protocol_sha256
        self.config_path = config_path
        self.expected_config_sha256 = expected_config_sha256
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.smoke = smoke
        self.max_new_checkpoints = _validate_checkpoint_budget(max_new_checkpoints)
        self.logger = RunLogger("model-benchmark", logs_dir)

    def run(self) -> dict[str, Any]:
        """Execute the benchmark with resumable model-fold checkpoints."""
        started = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.event(
            "model_benchmark_started",
            feature_dir=self.feature_dir,
            output_dir=self.output_dir,
            smoke=self.smoke,
            max_new_checkpoints=self.max_new_checkpoints,
        )

        try:
            with StageTimer(self.logger, "benchmark_config_load"):
                config, config_sha256 = BenchmarkConfig.load(self.config_path)
                if config_sha256 != self.expected_config_sha256:
                    raise ValueError(
                        "model benchmark config SHA-256 mismatch: "
                        f"expected={self.expected_config_sha256} actual={config_sha256}"
                    )
                if self.smoke:
                    config = _smoke_config(config)

            with StageTimer(self.logger, "validation_protocol_load"):
                protocol = _load_protocol(
                    self.protocol_path,
                    expected_sha256=self.expected_protocol_sha256,
                    config=config,
                )
                folds = _protocol_folds(protocol)
                if self.smoke:
                    folds = folds[:1]

            with StageTimer(
                self.logger,
                "feature_snapshot_verify",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                snapshot = FeatureSnapshot.load(
                    self.feature_dir,
                    expected_manifest_sha256=self.expected_feature_manifest_sha256,
                    expected_protocol_sha256=self.expected_protocol_sha256,
                    verify_hashes=True,
                )

            git_commit = _git_commit()
            selected_features, feature_screen_sha256 = self._load_or_screen_features(
                snapshot,
                config=config,
                config_sha256=config_sha256,
                git_commit=git_commit,
            )
            identity = BenchmarkIdentity(
                git_commit=git_commit,
                feature_manifest_sha256=snapshot.manifest_sha256,
                validation_protocol_sha256=self.expected_protocol_sha256,
                benchmark_config_sha256=config_sha256,
                feature_screen_sha256=feature_screen_sha256,
                smoke=self.smoke,
            )
            state = BenchmarkStateStore(
                self.output_dir / "benchmark_state.json",
                identity=identity,
            )
            self.logger.event(
                "benchmark_provenance",
                git_commit=git_commit,
                feature_manifest_sha256=snapshot.manifest_sha256,
                feature_git_commit=snapshot.feature_git_commit,
                validation_protocol_sha256=self.expected_protocol_sha256,
                benchmark_config_sha256=config_sha256,
                feature_screen_sha256=feature_screen_sha256,
            )

            self.logger.event(
                "benchmark_feature_plan",
                selected_features=len(selected_features),
                categorical_features=sum(feature.categorical for feature in selected_features),
                feature_screen_sha256=feature_screen_sha256,
                models=list(config.enabled_model_names),
                folds=len(folds),
            )

            new_checkpoints = 0
            for fold in folds:
                remaining = _remaining_checkpoint_budget(
                    self.max_new_checkpoints,
                    completed=new_checkpoints,
                )
                if remaining == 0:
                    break
                new_checkpoints += self._run_fold(
                    fold,
                    snapshot=snapshot,
                    selected_features=selected_features,
                    config=config,
                    state=state,
                    max_new_checkpoints=remaining,
                )

            completed_model_folds = _verified_model_fold_count(
                state,
                folds=folds,
                model_names=config.enabled_model_names,
                output_root=self.output_dir,
            )
            total_model_folds = len(config.enabled_model_names) * len(folds)
            if completed_model_folds < total_model_folds:
                elapsed = round(time.monotonic() - started, 3)
                self.logger.event(
                    "model_benchmark_checkpoint_yielded",
                    completed_model_folds=completed_model_folds,
                    total_model_folds=total_model_folds,
                    new_checkpoints=new_checkpoints,
                    total_elapsed_seconds=elapsed,
                )
                return {
                    "schema_version": 1,
                    "partial": True,
                    "smoke": self.smoke,
                    "git_commit": git_commit,
                    "feature_manifest_sha256": snapshot.manifest_sha256,
                    "validation_protocol_sha256": self.expected_protocol_sha256,
                    "benchmark_config_sha256": config_sha256,
                    "feature_screen_sha256": feature_screen_sha256,
                    "outer_holdout_touched": False,
                    "selected_features": len(selected_features),
                    "completed_model_folds": completed_model_folds,
                    "total_model_folds": total_model_folds,
                    "new_checkpoints": new_checkpoints,
                }

            with StageTimer(self.logger, "benchmark_aggregate"):
                summary = _aggregate_benchmark(
                    self.output_dir,
                    config=config,
                    state=state,
                    folds=folds,
                    selected_features=selected_features,
                    snapshot=snapshot,
                    feature_screen_sha256=feature_screen_sha256,
                    protocol_sha256=self.expected_protocol_sha256,
                    benchmark_config_sha256=config_sha256,
                    git_commit=git_commit,
                    smoke=self.smoke,
                )

            elapsed = round(time.monotonic() - started, 3)
            self.logger.event(
                "model_benchmark_completed",
                models=len(config.enabled_model_names),
                folds=len(folds),
                selected_features=len(selected_features),
                total_elapsed_seconds=elapsed,
                summary=self.output_dir / "benchmark_summary.json",
            )
            return summary

        except Exception as exc:
            self.logger.event(
                "model_benchmark_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                total_elapsed_seconds=round(time.monotonic() - started, 3),
            )
            raise

    def _load_or_screen_features(
        self,
        snapshot: FeatureSnapshot,
        *,
        config: BenchmarkConfig,
        config_sha256: str,
        git_commit: str,
    ) -> tuple[tuple[FeatureRef, ...], str]:
        screen_path = self.output_dir / "feature_screen.json"
        expected_identity = {
            "git_commit": git_commit,
            "feature_manifest_sha256": snapshot.manifest_sha256,
            "validation_protocol_sha256": self.expected_protocol_sha256,
            "benchmark_config_sha256": config_sha256,
            "smoke": self.smoke,
        }
        if screen_path.is_file():
            raw = json.loads(screen_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("feature_screen.json must be an object")
            payload = cast(dict[str, Any], raw)
            if payload.get("identity") != expected_identity:
                raise ValueError("feature-screen identity mismatch; refuse unsafe reuse")
            result_raw = payload.get("result")
            if not isinstance(result_raw, dict):
                raise ValueError("feature-screen result must be an object")
            features = feature_refs_from_payload(cast(dict[str, Any], result_raw))
            screen_sha256 = sha256_file(screen_path)
            self.logger.event(
                "feature_screen_resumed",
                selected_features=len(features),
                feature_screen_sha256=screen_sha256,
                path=screen_path,
            )
            return features, screen_sha256

        candidates = snapshot.candidate_features(excluded=frozenset(config.excluded_predictors))
        screening = config.screening
        with StageTimer(
            self.logger,
            "feature_screen_train_frame",
            heartbeat_seconds=config.heartbeat_seconds,
        ):
            train = load_feature_frame(
                snapshot,
                candidates,
                week_min=screening.train_week_min,
                week_max=screening.train_week_max,
                max_rows=screening.max_train_rows,
                seed=config.seed,
            )
        with StageTimer(
            self.logger,
            "feature_screen_validation_frame",
            heartbeat_seconds=config.heartbeat_seconds,
        ):
            validation = load_feature_frame(
                snapshot,
                candidates,
                week_min=screening.validation_week_min,
                week_max=screening.validation_week_max,
                max_rows=screening.max_validation_rows,
                seed=config.seed + 1,
            )
        with StageTimer(
            self.logger,
            "feature_screen_fit",
            heartbeat_seconds=config.heartbeat_seconds,
        ):
            result = screen_features(
                train,
                validation,
                candidates,
                config=screening,
                seed=config.seed,
                threads=config.threads,
                logger=self.logger,
            )

        payload = {
            "schema_version": 1,
            "identity": expected_identity,
            "screening_config": asdict(screening),
            "result": result.to_payload(),
        }
        _atomic_json(screen_path, payload)
        screen_sha256 = sha256_file(screen_path)
        self.logger.event(
            "feature_screen_artifact_written",
            selected_features=len(result.selected_features),
            feature_screen_sha256=screen_sha256,
            path=screen_path,
        )
        del train, validation
        gc.collect()
        return result.selected_features, screen_sha256

    def _run_fold(
        self,
        fold: dict[str, int],
        *,
        snapshot: FeatureSnapshot,
        selected_features: tuple[FeatureRef, ...],
        config: BenchmarkConfig,
        state: BenchmarkStateStore,
        max_new_checkpoints: int | None,
    ) -> int:
        fold_number = fold["fold"]
        pending = [
            model
            for model in config.enabled_model_names
            if state.receipt(model, fold_number, output_root=self.output_dir) is None
        ]
        if not pending:
            self.logger.event("benchmark_fold_resumed", fold=fold_number, models="all")
            return 0

        pending = _limit_pending_models(
            pending,
            max_new_checkpoints=max_new_checkpoints,
        )

        self.logger.event(
            "benchmark_fold_started",
            fold=fold_number,
            train_week_min=fold["train_week_min"],
            train_week_max=fold["train_week_max"],
            validation_week_min=fold["validation_week_min"],
            validation_week_max=fold["validation_week_max"],
            pending_models=pending,
        )
        train_cap = 100_000 if self.smoke else None
        validation_cap = 40_000 if self.smoke else None

        with StageTimer(
            self.logger,
            f"materialize_fold_{fold_number}",
            heartbeat_seconds=config.heartbeat_seconds,
        ):
            train, validation = load_fold_frames(
                snapshot,
                selected_features,
                train_week_min=fold["train_week_min"],
                train_week_max=fold["train_week_max"],
                validation_week_min=fold["validation_week_min"],
                validation_week_max=fold["validation_week_max"],
                seed=config.seed + fold_number * 101,
                train_row_cap=train_cap,
                validation_row_cap=validation_cap,
            )

        y_train = train.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
        y_validation = validation.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
        validation_case_ids = validation.get_column(CASE_ID).to_numpy()
        validation_weeks = validation.get_column(WEEK_NUM).to_numpy()

        numeric_pending = [
            model for model in pending if model in {"linear_logistic", "lightgbm", "xgboost"}
        ]
        if numeric_pending:
            with StageTimer(
                self.logger,
                f"encode_frequency_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                encoded = frequency_encode(train, validation, selected_features)

            for model_name in numeric_pending:
                self._run_numeric_model(
                    model_name,
                    fold_number=fold_number,
                    encoded_train=encoded.train,
                    encoded_validation=encoded.validation,
                    feature_names=encoded.feature_names,
                    y_train=y_train,
                    y_validation=y_validation,
                    validation_case_ids=validation_case_ids,
                    validation_weeks=validation_weeks,
                    config=config,
                    state=state,
                )
            del encoded
            gc.collect()

        if "catboost" in pending:
            with StageTimer(
                self.logger,
                f"encode_catboost_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                encoded_cat = catboost_encode(train, validation, selected_features)
            artifact_path = self.output_dir / "models" / "catboost" / f"fold_{fold_number}.cbm"
            with StageTimer(
                self.logger,
                f"fit_catboost_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                fit = fit_catboost(
                    encoded_cat.train,
                    y_train,
                    encoded_cat.validation,
                    y_validation,
                    categorical_indices=encoded_cat.categorical_indices,
                    params=_model_params(config.model("catboost"), smoke=self.smoke),
                    seed=config.seed + fold_number,
                    threads=config.threads,
                    artifact_path=artifact_path,
                    logger=self.logger,
                )
            self._record_fold(
                "catboost",
                fold_number=fold_number,
                fit=fit,
                y_validation=y_validation,
                validation_case_ids=validation_case_ids,
                validation_weeks=validation_weeks,
                feature_count=len(selected_features),
                train_rows=train.height,
                train_positive_rate=float(y_train.mean()),
                state=state,
            )
            del encoded_cat
            gc.collect()

        del train, validation
        gc.collect()

        remaining_models = [
            model
            for model in config.enabled_model_names
            if state.receipt(model, fold_number, output_root=self.output_dir) is None
        ]
        if remaining_models:
            self.logger.event(
                "benchmark_fold_checkpoint_yielded",
                fold=fold_number,
                completed_models=pending,
                remaining_models=remaining_models,
            )
        else:
            self.logger.event("benchmark_fold_completed", fold=fold_number)
        return len(pending)

    def _run_numeric_model(
        self,
        model_name: str,
        *,
        fold_number: int,
        encoded_train: NDArray[np.float32],
        encoded_validation: NDArray[np.float32],
        feature_names: tuple[str, ...],
        y_train: NDArray[np.int8],
        y_validation: NDArray[np.int8],
        validation_case_ids: NDArray[np.generic],
        validation_weeks: NDArray[np.generic],
        config: BenchmarkConfig,
        state: BenchmarkStateStore,
    ) -> None:
        model_config = config.model(model_name)
        params = _model_params(model_config, smoke=self.smoke)
        if model_name == "linear_logistic":
            width = min(int(params["max_features"]), encoded_train.shape[1])
            x_train = np.asarray(encoded_train[:, :width], dtype=np.float32)
            x_validation = np.asarray(encoded_validation[:, :width], dtype=np.float32)
            names = feature_names[:width]
            with StageTimer(
                self.logger,
                f"preprocess_linear_logistic_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                x_train, x_validation, medians, means, scales = standardize_for_linear(
                    x_train,
                    x_validation,
                )
            artifact_path = (
                self.output_dir / "models" / "linear_logistic" / f"fold_{fold_number}.npz"
            )
            with StageTimer(
                self.logger,
                f"fit_linear_logistic_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                fit = fit_linear_logistic(
                    x_train,
                    y_train.astype(np.int8, copy=False),
                    x_validation,
                    params=params,
                    seed=config.seed + fold_number,
                    artifact_path=artifact_path,
                    feature_names=names,
                    medians=medians,
                    means=means,
                    scales=scales,
                    logger=self.logger,
                )
            feature_count = width
            del x_train, x_validation, medians, means, scales
        elif model_name == "lightgbm":
            artifact_path = self.output_dir / "models" / "lightgbm" / f"fold_{fold_number}.txt"
            with StageTimer(
                self.logger,
                f"fit_lightgbm_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                fit = fit_lightgbm(
                    encoded_train,
                    y_train.astype(np.int8, copy=False),
                    encoded_validation,
                    y_validation.astype(np.int8, copy=False),
                    params=params,
                    seed=config.seed + fold_number,
                    threads=config.threads,
                    artifact_path=artifact_path,
                    feature_names=feature_names,
                    logger=self.logger,
                )
            feature_count = len(feature_names)
        elif model_name == "xgboost":
            artifact_path = self.output_dir / "models" / "xgboost" / f"fold_{fold_number}.ubj"
            with StageTimer(
                self.logger,
                f"fit_xgboost_fold_{fold_number}",
                heartbeat_seconds=config.heartbeat_seconds,
            ):
                fit = fit_xgboost(
                    encoded_train,
                    y_train.astype(np.int8, copy=False),
                    encoded_validation,
                    y_validation.astype(np.int8, copy=False),
                    params=params,
                    seed=config.seed + fold_number,
                    threads=config.threads,
                    artifact_path=artifact_path,
                    feature_names=feature_names,
                    logger=self.logger,
                )
            feature_count = len(feature_names)
        else:
            raise ValueError(f"unsupported numeric model: {model_name}")

        self._record_fold(
            model_name,
            fold_number=fold_number,
            fit=fit,
            y_validation=y_validation,
            validation_case_ids=validation_case_ids,
            validation_weeks=validation_weeks,
            feature_count=feature_count,
            train_rows=encoded_train.shape[0],
            train_positive_rate=float(y_train.mean()),
            state=state,
        )
        gc.collect()

    def _record_fold(
        self,
        model_name: str,
        *,
        fold_number: int,
        fit: ModelFitResult,
        y_validation: NDArray[np.int8],
        validation_case_ids: NDArray[np.generic],
        validation_weeks: NDArray[np.generic],
        feature_count: int,
        train_rows: int,
        train_positive_rate: float,
        state: BenchmarkStateStore,
    ) -> None:
        metrics = _evaluate_predictions(
            y_validation,
            fit.prediction,
            validation_weeks,
        )
        metrics.update(
            {
                "best_iteration": fit.best_iteration,
                "train_rows": int(train_rows),
                "validation_rows": len(y_validation),
                "train_positive_rate": float(train_positive_rate),
                "validation_positive_rate": float(y_validation.mean()),
                "features": int(feature_count),
            }
        )
        prediction_path = (
            self.output_dir / "predictions" / model_name / f"fold_{fold_number}.parquet"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                CASE_ID: validation_case_ids,
                TARGET: y_validation,
                WEEK_NUM: validation_weeks,
                "fold": np.full(len(y_validation), fold_number, dtype=np.int8),
                "prediction": fit.prediction,
            }
        ).sort(CASE_ID).write_parquet(
            prediction_path,
            compression="zstd",
            statistics=True,
        )
        receipt = FoldReceipt(
            model=model_name,
            fold=fold_number,
            prediction_path=relative_artifact(prediction_path, output_root=self.output_dir),
            prediction_sha256=sha256_file(prediction_path),
            model_path=relative_artifact(fit.artifact_path, output_root=self.output_dir),
            model_sha256=sha256_file(fit.artifact_path),
            metrics=metrics,
        )
        state.record(receipt)
        self.logger.event(
            "benchmark_fold_model_completed",
            model=model_name,
            fold=fold_number,
            stability_score=round(float(metrics["stability_score"]), 6),
            mean_gini=round(float(metrics["mean_gini"]), 6),
            auc=round(float(metrics["auc"]), 6),
            brier_score=round(float(metrics["brier_score"]), 6),
            best_iteration=fit.best_iteration,
        )


def _verified_model_fold_count(
    state: BenchmarkStateStore,
    *,
    folds: tuple[dict[str, int], ...],
    model_names: tuple[str, ...],
    output_root: Path,
) -> int:
    count = 0
    for fold in folds:
        fold_number = fold["fold"]
        for model_name in model_names:
            if state.receipt(model_name, fold_number, output_root=output_root) is not None:
                count += 1
    return count


def _validate_checkpoint_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("max_new_checkpoints must be positive")
    return value


def _remaining_checkpoint_budget(
    budget: int | None,
    *,
    completed: int,
) -> int | None:
    if completed < 0:
        raise ValueError("completed checkpoints cannot be negative")
    if budget is None:
        return None
    return max(0, budget - completed)


def _limit_pending_models(
    pending: list[str],
    *,
    max_new_checkpoints: int | None,
) -> list[str]:
    budget = _validate_checkpoint_budget(max_new_checkpoints)
    if budget is None:
        return list(pending)
    return list(pending[:budget])


def _load_protocol(
    path: Path,
    *,
    expected_sha256: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validation protocol must be a JSON object")
    protocol = cast(dict[str, Any], raw)
    if protocol.get("protocol_sha256") != expected_sha256:
        raise ValueError("validation protocol SHA-256 mismatch")
    outer = protocol.get("outer_holdout")
    if not isinstance(outer, dict):
        raise ValueError("validation protocol outer_holdout must be an object")
    outer_payload = cast(dict[str, Any], outer)
    if outer_payload.get("locked") is not True:
        raise ValueError("outer holdout must remain locked")
    outer_min = outer_payload.get("validation_week_min")
    if outer_min != config.outer_holdout_guard_week_min:
        raise ValueError(
            "outer-holdout guard mismatch: "
            f"protocol={outer_min} config={config.outer_holdout_guard_week_min}"
        )
    folds = _protocol_folds(protocol)
    if not folds:
        raise ValueError("validation protocol contains no inner folds")
    first_validation = min(fold["validation_week_min"] for fold in folds)
    if config.screening.validation_week_max >= first_validation:
        raise ValueError("screening overlaps frozen inner validation")
    for fold in folds:
        if fold["validation_week_max"] >= config.outer_holdout_guard_week_min:
            raise ValueError("inner CV touches locked outer holdout")
        if fold["train_week_max"] >= fold["validation_week_min"]:
            raise ValueError("inner fold train/validation overlap")
    return protocol


def _protocol_folds(protocol: dict[str, Any]) -> tuple[dict[str, int], ...]:
    inner = protocol.get("inner_temporal_cv")
    if not isinstance(inner, dict):
        raise ValueError("validation protocol inner_temporal_cv must be an object")
    raw_folds = cast(dict[str, Any], inner).get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError("validation protocol inner_temporal_cv.folds must be a list")
    folds: list[dict[str, int]] = []
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, dict):
            raise ValueError("inner fold must be an object")
        payload = cast(dict[str, Any], raw_fold)
        folds.append(
            {
                "fold": int(payload["fold"]),
                "train_week_min": int(payload["train_week_min"]),
                "train_week_max": int(payload["train_week_max"]),
                "validation_week_min": int(payload["validation_week_min"]),
                "validation_week_max": int(payload["validation_week_max"]),
            }
        )
    return tuple(sorted(folds, key=lambda item: item["fold"]))


def _evaluate_predictions(
    y_true: NDArray[np.generic],
    prediction: NDArray[np.generic],
    week_num: NDArray[np.generic],
) -> dict[str, float]:
    if len(y_true) != len(prediction) or len(y_true) != len(week_num):
        raise ValueError("prediction metric arrays must have equal length")
    if not np.isfinite(prediction).all():
        raise ValueError("model predictions must be finite")
    clipped = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    stability = stability_score(y_true, clipped, week_num)
    return {
        "stability_score": float(stability.score),
        "mean_gini": float(stability.mean_gini),
        "temporal_slope": float(stability.slope),
        "residual_std": float(stability.residual_std),
        "auc": float(roc_auc_score(y_true, clipped)),
        "pr_auc": float(average_precision_score(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def _weekly_gini_rows(
    model_name: str,
    frame: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Return transparent per-week Gini diagnostics for one OOF prediction set."""
    rows: list[dict[str, Any]] = []
    weeks = sorted(int(value) for value in frame.get_column(WEEK_NUM).unique().to_list())
    for week in weeks:
        week_frame = frame.filter(pl.col(WEEK_NUM) == week)
        target = week_frame.get_column(TARGET).to_numpy()
        prediction = week_frame.get_column("prediction").to_numpy()
        if np.unique(target).size < 2:
            continue
        rows.append(
            {
                "model": model_name,
                "week_num": week,
                "rows": week_frame.height,
                "positives": int(np.asarray(target).sum()),
                "positive_rate": float(np.asarray(target).mean()),
                "gini": float(normalized_gini(target, prediction)),
                "prediction_mean": float(np.asarray(prediction).mean()),
            }
        )
    return rows


def _aggregate_benchmark(
    output_dir: Path,
    *,
    config: BenchmarkConfig,
    state: BenchmarkStateStore,
    folds: tuple[dict[str, int], ...],
    selected_features: tuple[FeatureRef, ...],
    snapshot: FeatureSnapshot,
    feature_screen_sha256: str,
    protocol_sha256: str,
    benchmark_config_sha256: str,
    git_commit: str,
    smoke: bool,
) -> dict[str, Any]:
    receipts = state.completed_receipts()
    by_model: dict[str, list[FoldReceipt]] = {}
    for receipt in receipts:
        by_model.setdefault(receipt.model, []).append(receipt)

    expected_fold_numbers = {fold["fold"] for fold in folds}
    model_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    weekly_metric_rows: list[dict[str, Any]] = []

    for model_name in config.enabled_model_names:
        model_receipts = sorted(by_model.get(model_name, []), key=lambda item: item.fold)
        observed_folds = {receipt.fold for receipt in model_receipts}
        if observed_folds != expected_fold_numbers:
            raise RuntimeError(
                f"incomplete benchmark model {model_name}: "
                f"expected={sorted(expected_fold_numbers)} observed={sorted(observed_folds)}"
            )

        prediction_frames = [
            pl.read_parquet(output_dir / receipt.prediction_path) for receipt in model_receipts
        ]
        oof = pl.concat(prediction_frames, how="vertical").sort(CASE_ID)
        if oof.get_column(CASE_ID).n_unique() != oof.height:
            raise RuntimeError(f"OOF case_id duplication for {model_name}")
        if not smoke:
            observed_weeks = {int(value) for value in oof.get_column(WEEK_NUM).unique().to_list()}
            expected_weeks = set(range(33, 73))
            if observed_weeks != expected_weeks:
                raise RuntimeError(
                    f"OOF weeks are incomplete for {model_name}: "
                    f"missing={sorted(expected_weeks - observed_weeks)} "
                    f"unexpected={sorted(observed_weeks - expected_weeks)}"
                )

        oof_path = output_dir / "oof" / f"{model_name}.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof.write_parquet(oof_path, compression="zstd", statistics=True)
        oof_metrics = _evaluate_predictions(
            oof.get_column(TARGET).to_numpy(),
            oof.get_column("prediction").to_numpy(),
            oof.get_column(WEEK_NUM).to_numpy(),
        )
        weekly_metric_rows.extend(_weekly_gini_rows(model_name, oof))

        fold_metrics = [receipt.metrics for receipt in model_receipts]
        stabilities = np.asarray(
            [float(metrics["stability_score"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        ginis = np.asarray(
            [float(metrics["mean_gini"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        slopes = np.asarray(
            [float(metrics["temporal_slope"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        residuals = np.asarray(
            [float(metrics["residual_std"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        briers = np.asarray(
            [float(metrics["brier_score"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        aucs = np.asarray(
            [float(metrics["auc"]) for metrics in fold_metrics],
            dtype=np.float64,
        )
        row = {
            "model": model_name,
            "folds": len(model_receipts),
            "mean_inner_stability_score": float(stabilities.mean()),
            "std_inner_stability_score": float(stabilities.std(ddof=0)),
            "worst_fold_stability_score": float(stabilities.min()),
            "mean_weekly_gini": float(ginis.mean()),
            "mean_temporal_slope": float(slopes.mean()),
            "mean_residual_std": float(residuals.mean()),
            "mean_brier_score": float(briers.mean()),
            "mean_auc": float(aucs.mean()),
            "oof_stability_score": float(oof_metrics["stability_score"]),
            "oof_mean_gini": float(oof_metrics["mean_gini"]),
            "oof_temporal_slope": float(oof_metrics["temporal_slope"]),
            "oof_residual_std": float(oof_metrics["residual_std"]),
            "oof_auc": float(oof_metrics["auc"]),
            "oof_pr_auc": float(oof_metrics["pr_auc"]),
            "oof_brier_score": float(oof_metrics["brier_score"]),
            "oof_log_loss": float(oof_metrics["log_loss"]),
            "oof_path": oof_path.relative_to(output_dir).as_posix(),
            "oof_sha256": sha256_file(oof_path),
        }
        model_rows.append(row)
        for receipt in model_receipts:
            fold_metric_rows.append(
                {
                    "model": receipt.model,
                    "fold": receipt.fold,
                    **receipt.metrics,
                }
            )

    ranking = sorted(
        model_rows,
        key=lambda item: (
            -float(item["mean_inner_stability_score"]),
            -float(item["worst_fold_stability_score"]),
            -float(item["mean_weekly_gini"]),
            -float(item["mean_temporal_slope"]),
            float(item["mean_residual_std"]),
            float(item["mean_brier_score"]),
        ),
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    summary: dict[str, Any] = {
        "schema_version": 1,
        "name": config.name,
        "smoke": smoke,
        "git_commit": git_commit,
        "feature_manifest_sha256": snapshot.manifest_sha256,
        "feature_git_commit": snapshot.feature_git_commit,
        "feature_screen_sha256": feature_screen_sha256,
        "validation_protocol_sha256": protocol_sha256,
        "benchmark_config_sha256": benchmark_config_sha256,
        "outer_holdout_touched": False,
        "selected_features": len(selected_features),
        "categorical_features": sum(feature.categorical for feature in selected_features),
        "selected_features_by_block": dict(
            sorted(Counter(feature.block for feature in selected_features).items())
        ),
        "folds": len(folds),
        "models": ranking,
        "selection_policy": {
            "primary": "mean_inner_stability_score",
            "secondary": [
                "worst_fold_stability_score",
                "mean_weekly_gini",
                "mean_temporal_slope",
                "mean_residual_std",
                "mean_brier_score",
            ],
        },
    }
    summary_path = output_dir / "benchmark_summary.json"
    fold_metrics_path = output_dir / "fold_metrics.jsonl"
    weekly_metrics_path = output_dir / "weekly_metrics.jsonl"
    state_path = output_dir / "benchmark_state.json"

    _atomic_json(summary_path, summary)
    _write_jsonl(
        fold_metrics_path,
        sorted(
            fold_metric_rows,
            key=lambda item: (str(item["model"]), int(item["fold"])),
        ),
    )
    _write_jsonl(
        weekly_metrics_path,
        sorted(
            weekly_metric_rows,
            key=lambda item: (str(item["model"]), int(item["week_num"])),
        ),
    )
    _atomic_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "git_commit": git_commit,
            "feature_manifest_sha256": snapshot.manifest_sha256,
            "feature_screen_sha256": feature_screen_sha256,
            "validation_protocol_sha256": protocol_sha256,
            "benchmark_config_sha256": benchmark_config_sha256,
            "benchmark_summary_sha256": sha256_file(summary_path),
            "benchmark_state_sha256": sha256_file(state_path),
            "fold_metrics_sha256": sha256_file(fold_metrics_path),
            "weekly_metrics_sha256": sha256_file(weekly_metrics_path),
            "smoke": smoke,
            "models": list(config.enabled_model_names),
            "folds": [fold["fold"] for fold in folds],
            "selected_features": len(selected_features),
            "outer_holdout_touched": False,
        },
    )
    return summary


def _smoke_config(config: BenchmarkConfig) -> BenchmarkConfig:
    screening = replace(
        config.screening,
        max_train_rows=min(config.screening.max_train_rows, 50_000),
        max_validation_rows=min(config.screening.max_validation_rows, 25_000),
        max_features=min(config.screening.max_features, 128),
        min_features_per_block=min(config.screening.min_features_per_block, 2),
        num_boost_round=min(config.screening.num_boost_round, 180),
        early_stopping_rounds=min(config.screening.early_stopping_rounds, 30),
    )
    return replace(
        config,
        screening=screening,
        heartbeat_seconds=min(config.heartbeat_seconds, 5.0),
    )


def _model_params(model: ModelConfig, *, smoke: bool) -> dict[str, Any]:
    params = dict(model.params)
    if not smoke:
        return params
    if model.name == "linear_logistic":
        params["max_features"] = min(int(params["max_features"]), 96)
        params["max_iter"] = min(int(params["max_iter"]), 25)
    elif model.name == "lightgbm" or model.name == "xgboost":
        params["num_boost_round"] = min(int(params["num_boost_round"]), 180)
        params["early_stopping_rounds"] = min(int(params["early_stopping_rounds"]), 30)
    elif model.name == "catboost":
        params["iterations"] = min(int(params["iterations"]), 180)
        params["early_stopping_rounds"] = min(int(params["early_stopping_rounds"]), 30)
    return params


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summary_sha256(path: Path) -> str:
    """Return SHA-256 for a benchmark summary artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
