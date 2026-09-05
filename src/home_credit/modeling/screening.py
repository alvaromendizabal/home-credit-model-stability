"""Leakage-safe predictive feature screening with temporal drift regularization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from home_credit.metrics.stability import stability_score
from home_credit.modeling.config import ScreeningConfig
from home_credit.modeling.data import TARGET, WEEK_NUM, FeatureRef
from home_credit.modeling.encoding import frequency_encode
from home_credit.observability.logging import RunLogger


@dataclass(frozen=True, slots=True)
class FeatureScore:
    """Screening statistics and importance for one candidate predictor."""

    name: str
    block: str
    family: str
    depth: int
    dtype: str
    categorical: bool
    missing_fraction: float
    unique_values: int
    target_gain: float
    drift_gain: float
    selection_score: float
    selected: bool


@dataclass(frozen=True, slots=True)
class FeatureScreenResult:
    """Frozen result of early-window predictive and drift-aware screening."""

    selected_features: tuple[FeatureRef, ...]
    scores: tuple[FeatureScore, ...]
    target_validation_auc: float
    target_validation_stability: float
    target_validation_mean_gini: float
    target_validation_slope: float
    target_validation_residual_std: float
    drift_validation_auc: float
    structural_candidates: int
    eligible_candidates: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize the result to deterministic JSON-compatible objects."""
        return {
            "selected_features": [asdict(feature) for feature in self.selected_features],
            "scores": [asdict(score) for score in self.scores],
            "target_validation_auc": self.target_validation_auc,
            "target_validation_stability": self.target_validation_stability,
            "target_validation_mean_gini": self.target_validation_mean_gini,
            "target_validation_slope": self.target_validation_slope,
            "target_validation_residual_std": self.target_validation_residual_std,
            "drift_validation_auc": self.drift_validation_auc,
            "structural_candidates": self.structural_candidates,
            "eligible_candidates": self.eligible_candidates,
        }


def screen_features(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    features: tuple[FeatureRef, ...],
    *,
    config: ScreeningConfig,
    seed: int,
    threads: int,
    logger: RunLogger,
) -> FeatureScreenResult:
    """Rank predictors using early target gain penalized by temporal drift gain.

    Screening uses only the configured weeks that precede every frozen inner
    validation fold. No outer-holdout target or feature distribution is used.
    """
    if train.height == 0 or validation.height == 0:
        raise ValueError("screening train and validation frames must be non-empty")
    if not features:
        raise ValueError("screening requires candidate features")

    structural = _structural_screen(train, features, config=config, logger=logger)
    eligible = tuple(item[0] for item in structural if item[1] is None)
    if len(eligible) < config.max_features:
        logger.event(
            "feature_screen_candidate_floor",
            eligible=len(eligible),
            requested=config.max_features,
        )
    if len(eligible) < 16:
        raise ValueError(f"too few eligible features after structural screen: {len(eligible)}")

    logger.event(
        "feature_screen_encoding_started",
        train_rows=train.height,
        validation_rows=validation.height,
        eligible_features=len(eligible),
    )
    encoded = frequency_encode(train, validation, eligible)
    y_train = train.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
    y_validation = validation.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
    validation_weeks = validation.get_column(WEEK_NUM).to_numpy()

    target_model = _train_screen_model(
        encoded.train,
        y_train,
        encoded.validation,
        y_validation,
        config=config,
        seed=seed,
        threads=threads,
        logger=logger,
        label="target",
    )
    target_prediction = target_model.predict(
        encoded.validation,
        num_iteration=target_model.best_iteration,
    )
    target_auc = float(roc_auc_score(y_validation, target_prediction))
    target_stability = stability_score(y_validation, target_prediction, validation_weeks)
    target_gain = target_model.feature_importance(importance_type="gain").astype(np.float64)

    drift_train_rows = min(encoded.train.shape[0], encoded.validation.shape[0])
    if drift_train_rows < 1000:
        raise ValueError("too few rows for drift screen")
    drift_x = np.concatenate(
        [encoded.train[:drift_train_rows], encoded.validation[:drift_train_rows]],
        axis=0,
    )
    drift_y = np.concatenate(
        [
            np.zeros(drift_train_rows, dtype=np.int8),
            np.ones(drift_train_rows, dtype=np.int8),
        ]
    )
    rng = np.random.default_rng(seed + 29)
    order = rng.permutation(drift_x.shape[0])
    drift_x = drift_x[order]
    drift_y = drift_y[order]
    split = int(drift_x.shape[0] * 0.8)
    drift_model = _train_screen_model(
        drift_x[:split],
        drift_y[:split],
        drift_x[split:],
        drift_y[split:],
        config=config,
        seed=seed + 17,
        threads=threads,
        logger=logger,
        label="drift",
    )
    drift_prediction = drift_model.predict(
        drift_x[split:],
        num_iteration=drift_model.best_iteration,
    )
    drift_auc = float(roc_auc_score(drift_y[split:], drift_prediction))
    drift_gain = drift_model.feature_importance(importance_type="gain").astype(np.float64)

    target_scaled = _scale_importance(target_gain)
    drift_scaled = _scale_importance(drift_gain)
    selection_scores = target_scaled * np.maximum(
        0.0,
        1.0 - config.drift_penalty * drift_scaled,
    )

    ranking = sorted(
        zip(eligible, selection_scores, target_gain, drift_gain, strict=True),
        key=lambda item: (-float(item[1]), -float(item[2]), item[0].name),
    )
    selected_names = _select_with_block_floor(
        ranking,
        max_features=min(config.max_features, len(ranking)),
        min_features_per_block=config.min_features_per_block,
    )
    selected_features = tuple(item[0] for item in ranking if item[0].name in selected_names)

    score_by_name = {
        feature.name: (
            float(selection_score),
            float(target_value),
            float(drift_value),
        )
        for feature, selection_score, target_value, drift_value in ranking
    }

    score_rows: list[FeatureScore] = []
    for feature, reason, missing_fraction, unique_values in structural:
        if reason is None:
            selection_score, target_value, drift_value = score_by_name[feature.name]
        else:
            selection_score = 0.0
            target_value = 0.0
            drift_value = 0.0
        score_rows.append(
            FeatureScore(
                name=feature.name,
                block=feature.block,
                family=feature.family,
                depth=feature.depth,
                dtype=feature.dtype,
                categorical=feature.categorical,
                missing_fraction=missing_fraction,
                unique_values=unique_values,
                target_gain=target_value,
                drift_gain=drift_value,
                selection_score=selection_score,
                selected=feature.name in selected_names,
            )
        )

    logger.event(
        "feature_screen_completed",
        structural_candidates=len(features),
        eligible_candidates=len(eligible),
        selected_features=len(selected_features),
        target_validation_auc=round(target_auc, 6),
        target_validation_stability=round(target_stability.score, 6),
        drift_validation_auc=round(drift_auc, 6),
    )
    return FeatureScreenResult(
        selected_features=selected_features,
        scores=tuple(sorted(score_rows, key=lambda item: item.name)),
        target_validation_auc=target_auc,
        target_validation_stability=target_stability.score,
        target_validation_mean_gini=target_stability.mean_gini,
        target_validation_slope=target_stability.slope,
        target_validation_residual_std=target_stability.residual_std,
        drift_validation_auc=drift_auc,
        structural_candidates=len(features),
        eligible_candidates=len(eligible),
    )


def feature_refs_from_payload(payload: dict[str, Any]) -> tuple[FeatureRef, ...]:
    """Restore selected feature references from a verified screen artifact."""
    raw = payload.get("selected_features")
    if not isinstance(raw, list):
        raise ValueError("feature screen selected_features must be a list")
    refs: list[FeatureRef] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("feature screen entry must be an object")
        refs.append(
            FeatureRef(
                name=str(item["name"]),
                block=str(item["block"]),
                family=str(item["family"]),
                depth=int(item["depth"]),
                dtype=str(item["dtype"]),
                categorical=bool(item["categorical"]),
            )
        )
    if len({item.name for item in refs}) != len(refs):
        raise ValueError("feature screen contains duplicate feature names")
    return tuple(refs)


def _structural_screen(
    train: pl.DataFrame,
    features: tuple[FeatureRef, ...],
    *,
    config: ScreeningConfig,
    logger: RunLogger,
) -> tuple[tuple[FeatureRef, str | None, float, int], ...]:
    rows: list[tuple[FeatureRef, str | None, float, int]] = []
    batch_size = 96
    for offset in range(0, len(features), batch_size):
        batch = features[offset : offset + batch_size]
        expressions: list[pl.Expr] = []
        for index, feature in enumerate(batch):
            expressions.extend(
                [
                    pl.col(feature.name).null_count().alias(f"n_{index}"),
                    pl.col(feature.name).n_unique().alias(f"u_{index}"),
                ]
            )
        values = train.select(expressions).row(0, named=True)
        for index, feature in enumerate(batch):
            nulls = int(values[f"n_{index}"])
            unique_values = int(values[f"u_{index}"])
            missing_fraction = nulls / float(train.height)
            reason: str | None = None
            if missing_fraction > config.max_missing_fraction:
                reason = "missingness"
            elif unique_values <= 1:
                reason = "constant"
            elif feature.categorical and unique_values > config.max_categorical_cardinality:
                reason = "categorical_cardinality"
            rows.append((feature, reason, missing_fraction, unique_values))

        logger.event(
            "feature_screen_profile_progress",
            processed=min(offset + len(batch), len(features)),
            total=len(features),
        )
    return tuple(rows)


def _train_screen_model(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int8],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int8],
    *,
    config: ScreeningConfig,
    seed: int,
    threads: int,
    logger: RunLogger,
    label: str,
) -> Any:
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
    }
    logger.event(
        "feature_screen_model_started",
        label=label,
        train_rows=x_train.shape[0],
        validation_rows=x_validation.shape[0],
        features=x_train.shape[1],
    )
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    validation_set = lgb.Dataset(
        x_validation,
        label=y_validation,
        reference=train_set,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    logger.event(
        "feature_screen_model_completed",
        label=label,
        best_iteration=model.best_iteration,
    )
    return model


def _scale_importance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values / maximum, dtype=np.float64)


def _select_with_block_floor(
    ranking: list[tuple[FeatureRef, float, float, float]],
    *,
    max_features: int,
    min_features_per_block: int,
) -> set[str]:
    if max_features < 1:
        raise ValueError("max_features must be positive")
    selected: set[str] = set()
    if min_features_per_block > 0:
        by_block: dict[str, list[FeatureRef]] = {}
        for feature, _, _, _ in ranking:
            by_block.setdefault(feature.block, []).append(feature)
        for block in sorted(by_block):
            for feature in by_block[block][:min_features_per_block]:
                if len(selected) >= max_features:
                    break
                selected.add(feature.name)

    for feature, _, _, _ in ranking:
        if len(selected) >= max_features:
            break
        selected.add(feature.name)
    return selected
