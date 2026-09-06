"""Typed configuration for leakage-safe temporal model benchmarking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    """Feature-screening policy restricted to the common early train window."""

    train_week_min: int
    train_week_max: int
    validation_week_min: int
    validation_week_max: int
    max_train_rows: int
    max_validation_rows: int
    max_features: int
    min_features_per_block: int
    max_missing_fraction: float
    max_categorical_cardinality: int
    drift_penalty: float
    num_boost_round: int
    early_stopping_rounds: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int

    def validate(self, *, outer_holdout_guard_week_min: int) -> None:
        """Reject screening settings that could contaminate frozen validation."""
        if self.train_week_min < 0:
            raise ValueError("screening train_week_min must be non-negative")
        if self.train_week_max < self.train_week_min:
            raise ValueError("screening train week range is invalid")
        if self.validation_week_min <= self.train_week_max:
            raise ValueError("screening validation must follow screening train")
        if self.validation_week_max < self.validation_week_min:
            raise ValueError("screening validation week range is invalid")
        if self.validation_week_max >= outer_holdout_guard_week_min:
            raise ValueError("screening must not touch the outer holdout")
        if self.max_train_rows < 1000 or self.max_validation_rows < 1000:
            raise ValueError("screening row caps must be at least 1000")
        if self.max_features < 32:
            raise ValueError("screening max_features must be at least 32")
        if self.min_features_per_block < 0:
            raise ValueError("min_features_per_block must be non-negative")
        if not 0.0 <= self.max_missing_fraction < 1.0:
            raise ValueError("max_missing_fraction must be in [0, 1)")
        if self.max_categorical_cardinality < 2:
            raise ValueError("max_categorical_cardinality must be at least 2")
        if not 0.0 <= self.drift_penalty <= 1.0:
            raise ValueError("drift_penalty must be in [0, 1]")
        if self.num_boost_round < 10 or self.early_stopping_rounds < 5:
            raise ValueError("screening boosting settings are too small")
        if self.learning_rate <= 0.0:
            raise ValueError("screening learning_rate must be positive")
        if self.num_leaves < 2:
            raise ValueError("screening num_leaves must be at least 2")
        if self.max_depth < 2:
            raise ValueError("screening max_depth must be at least 2")
        if self.min_data_in_leaf < 1:
            raise ValueError("screening min_data_in_leaf must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One model family and its validated parameter payload."""

    name: str
    enabled: bool
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Frozen model-benchmark policy."""

    schema_version: int
    name: str
    seed: int
    threads: int
    heartbeat_seconds: float
    outer_holdout_guard_week_min: int
    excluded_predictors: tuple[str, ...]
    screening: ScreeningConfig
    models: tuple[ModelConfig, ...]
    notes: str
    feature_selection: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> tuple[BenchmarkConfig, str]:
        """Load, validate, and SHA-256 fingerprint a benchmark config."""
        payload_bytes = path.read_bytes()
        raw = json.loads(payload_bytes)
        if not isinstance(raw, dict):
            raise ValueError("model benchmark config must be a JSON object")
        payload = cast(dict[str, Any], raw)

        screening_raw = _required_mapping(payload, "screening")
        models_raw = _required_mapping(payload, "models")

        screening = ScreeningConfig(
            train_week_min=_required_int(screening_raw, "train_week_min"),
            train_week_max=_required_int(screening_raw, "train_week_max"),
            validation_week_min=_required_int(screening_raw, "validation_week_min"),
            validation_week_max=_required_int(screening_raw, "validation_week_max"),
            max_train_rows=_required_int(screening_raw, "max_train_rows"),
            max_validation_rows=_required_int(screening_raw, "max_validation_rows"),
            max_features=_required_int(screening_raw, "max_features"),
            min_features_per_block=_required_int(
                screening_raw,
                "min_features_per_block",
            ),
            max_missing_fraction=_required_float(
                screening_raw,
                "max_missing_fraction",
            ),
            max_categorical_cardinality=_required_int(
                screening_raw,
                "max_categorical_cardinality",
            ),
            drift_penalty=_required_float(screening_raw, "drift_penalty"),
            num_boost_round=_required_int(screening_raw, "num_boost_round"),
            early_stopping_rounds=_required_int(
                screening_raw,
                "early_stopping_rounds",
            ),
            learning_rate=_required_float(screening_raw, "learning_rate"),
            num_leaves=_required_int(screening_raw, "num_leaves"),
            max_depth=_required_int(screening_raw, "max_depth"),
            min_data_in_leaf=_required_int(screening_raw, "min_data_in_leaf"),
        )

        models: list[ModelConfig] = []
        allowed_models = {
            "linear_logistic",
            "lightgbm",
            "catboost",
            "xgboost",
        }
        unexpected = sorted(set(models_raw) - allowed_models)
        missing = sorted(allowed_models - set(models_raw))
        if unexpected or missing:
            raise ValueError(
                f"model config keys are invalid: missing={missing} unexpected={unexpected}"
            )

        for name in sorted(allowed_models):
            model_raw = models_raw[name]
            if not isinstance(model_raw, dict):
                raise ValueError(f"model config {name!r} must be an object")
            model_payload = cast(dict[str, Any], model_raw)
            enabled = model_payload.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError(f"model config {name!r}.enabled must be boolean")
            params = {key: value for key, value in model_payload.items() if key != "enabled"}
            models.append(ModelConfig(name=name, enabled=enabled, params=params))

        config = cls(
            schema_version=_required_int(payload, "schema_version"),
            name=_required_str(payload, "name"),
            seed=_required_int(payload, "seed"),
            threads=_required_int(payload, "threads"),
            heartbeat_seconds=_required_float(payload, "heartbeat_seconds"),
            outer_holdout_guard_week_min=_required_int(
                payload,
                "outer_holdout_guard_week_min",
            ),
            excluded_predictors=_required_str_tuple(payload, "excluded_predictors"),
            screening=screening,
            models=tuple(models),
            notes=_required_str(payload, "notes"),
            feature_selection=payload.get("feature_selection"),
        )
        config.validate()
        return config, hashlib.sha256(payload_bytes).hexdigest()

    def validate(self) -> None:
        """Reject unsafe or irreproducible benchmark settings."""
        if self.schema_version != 1:
            raise ValueError("unsupported benchmark schema_version")
        if not self.name:
            raise ValueError("benchmark name must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 1 <= self.threads <= 8:
            raise ValueError("threads must be between 1 and 8")
        if self.heartbeat_seconds <= 0.0:
            raise ValueError("heartbeat_seconds must be positive")
        if self.outer_holdout_guard_week_min < 1:
            raise ValueError("outer_holdout_guard_week_min must be positive")
        if len(set(self.excluded_predictors)) != len(self.excluded_predictors):
            raise ValueError("excluded_predictors must be unique")
        required_exclusions = {
            "case_id",
            "target",
            "WEEK_NUM",
            "MONTH",
            "base__decision_year",
            "base__decision_month",
            "base__decision_day",
            "base__decision_weekday",
            "base__decision_ordinal_day",
        }
        if not required_exclusions.issubset(self.excluded_predictors):
            missing = sorted(required_exclusions - set(self.excluded_predictors))
            raise ValueError(f"excluded_predictors is missing leakage/hack guards: {missing}")
        self.screening.validate(outer_holdout_guard_week_min=self.outer_holdout_guard_week_min)
        if not any(model.enabled for model in self.models):
            raise ValueError("at least one model must be enabled")
        _validate_model_params(self.models)
        if self.feature_selection is not None:
            policy = self.feature_selection
            if not isinstance(policy, dict) or set(policy) != {"path", "sha256", "exclude_blocks"}:
                raise ValueError("invalid frozen feature policy")
            if not isinstance(policy["path"], str) or not policy["path"]:
                raise ValueError("frozen feature path must be nonempty")
            digest = policy["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("invalid frozen feature SHA-256")
            blocks = policy["exclude_blocks"]
            if (
                not isinstance(blocks, list)
                or any(not isinstance(b, str) or not b for b in blocks)
                or len(set(blocks)) != len(blocks)
            ):
                raise ValueError("excluded blocks must be unique nonempty strings")

    def model(self, name: str) -> ModelConfig:
        """Return one configured model by canonical name."""
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    @property
    def enabled_model_names(self) -> tuple[str, ...]:
        """Return enabled model names in deterministic benchmark order."""
        order = ("linear_logistic", "lightgbm", "xgboost", "catboost")
        enabled = {model.name for model in self.models if model.enabled}
        return tuple(name for name in order if name in enabled)


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark field {key!r} must be an object")
    return cast(dict[str, Any], value)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark field {key!r} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"benchmark field {key!r} must be an integer")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"benchmark field {key!r} must be numeric")
    return float(value)


def _required_str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"benchmark field {key!r} must be a list of strings")
    return tuple(cast(list[str], value))


def _require_positive(params: dict[str, Any], model: str, key: str) -> None:
    value = params.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{model}.{key} must be positive")


def _require_fraction(params: dict[str, Any], model: str, key: str) -> None:
    value = params.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError(f"{model}.{key} must be in (0, 1]")


def _validate_model_params(models: tuple[ModelConfig, ...]) -> None:
    by_name = {model.name: model for model in models}

    linear = by_name["linear_logistic"].params
    for key in ("max_features", "alpha", "max_iter", "tol"):
        _require_positive(linear, "linear_logistic", key)

    lightgbm = by_name["lightgbm"].params
    for key in (
        "num_boost_round",
        "early_stopping_rounds",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "max_bin",
    ):
        _require_positive(lightgbm, "lightgbm", key)
    for key in ("feature_fraction", "bagging_fraction"):
        _require_fraction(lightgbm, "lightgbm", key)
    if int(lightgbm["early_stopping_rounds"]) >= int(lightgbm["num_boost_round"]):
        raise ValueError("lightgbm early stopping must be below num_boost_round")

    catboost = by_name["catboost"].params
    for key in (
        "iterations",
        "early_stopping_rounds",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
    ):
        _require_positive(catboost, "catboost", key)
    for key in ("rsm", "subsample"):
        _require_fraction(catboost, "catboost", key)
    if int(catboost["early_stopping_rounds"]) >= int(catboost["iterations"]):
        raise ValueError("catboost early stopping must be below iterations")
    random_strength = catboost.get("random_strength")
    if not isinstance(random_strength, (int, float)) or isinstance(random_strength, bool):
        raise ValueError("catboost.random_strength must be numeric")
    if float(random_strength) < 0:
        raise ValueError("catboost.random_strength must be non-negative")

    xgboost = by_name["xgboost"].params
    for key in (
        "num_boost_round",
        "early_stopping_rounds",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "reg_lambda",
        "max_bin",
    ):
        _require_positive(xgboost, "xgboost", key)
    for key in ("subsample", "colsample_bytree"):
        _require_fraction(xgboost, "xgboost", key)
    if int(xgboost["early_stopping_rounds"]) >= int(xgboost["num_boost_round"]):
        raise ValueError("xgboost early stopping must be below num_boost_round")
    reg_alpha = xgboost.get("reg_alpha")
    if not isinstance(reg_alpha, (int, float)) or isinstance(reg_alpha, bool):
        raise ValueError("xgboost.reg_alpha must be numeric")
    if float(reg_alpha) < 0:
        raise ValueError("xgboost.reg_alpha must be non-negative")
