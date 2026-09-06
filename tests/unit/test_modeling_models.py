from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from home_credit.modeling.models import (
    fit_catboost,
    fit_lightgbm,
    fit_linear_logistic,
    fit_xgboost,
)
from home_credit.observability.logging import RunLogger


def _numeric_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260905)
    x_train = rng.normal(size=(600, 8)).astype(np.float32)
    x_validation = rng.normal(size=(240, 8)).astype(np.float32)
    y_train = (x_train[:, 0] + 0.5 * x_train[:, 1] > 0.0).astype(np.int8)
    y_validation = (x_validation[:, 0] + 0.5 * x_validation[:, 1] > 0.0).astype(np.int8)
    return x_train, y_train, x_validation, y_validation


def _assert_predictions(prediction: np.ndarray, expected_rows: int) -> None:
    assert prediction.shape == (expected_rows,)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0.0) & (prediction <= 1.0)).all()


def test_linear_logistic_wrapper_persists_preprocessing(tmp_path: Path) -> None:
    x_train, y_train, x_validation, _ = _numeric_data()
    medians = np.zeros(x_train.shape[1], dtype=np.float32)
    means = np.zeros(x_train.shape[1], dtype=np.float32)
    scales = np.ones(x_train.shape[1], dtype=np.float32)
    artifact = tmp_path / "linear.npz"

    result = fit_linear_logistic(
        x_train,
        y_train,
        x_validation,
        params={
            "alpha": 1e-5,
            "max_iter": 200,
            "tol": 1e-2,
        },
        seed=7,
        artifact_path=artifact,
        feature_names=tuple(f"f{i}" for i in range(x_train.shape[1])),
        medians=medians,
        means=means,
        scales=scales,
        logger=RunLogger("linear-test", tmp_path / "logs"),
    )

    _assert_predictions(result.prediction, len(x_validation))
    assert result.best_iteration >= 1
    assert artifact.is_file()
    with np.load(artifact) as payload:
        assert payload["coef"].shape[1] == x_train.shape[1]
        assert payload["feature_names"].shape == (x_train.shape[1],)


def test_lightgbm_wrapper_fits_and_saves(tmp_path: Path) -> None:
    x_train, y_train, x_validation, y_validation = _numeric_data()
    artifact = tmp_path / "lightgbm.txt"

    result = fit_lightgbm(
        x_train,
        y_train,
        x_validation,
        y_validation,
        params={
            "learning_rate": 0.08,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "max_bin": 63,
            "num_boost_round": 40,
            "early_stopping_rounds": 8,
        },
        seed=7,
        threads=2,
        artifact_path=artifact,
        feature_names=tuple(f"f{i}" for i in range(x_train.shape[1])),
        logger=RunLogger("lightgbm-test", tmp_path / "logs"),
    )

    _assert_predictions(result.prediction, len(x_validation))
    assert result.best_iteration >= 1
    assert artifact.is_file()


def test_xgboost_wrapper_fits_quantile_histogram_model(tmp_path: Path) -> None:
    x_train, y_train, x_validation, y_validation = _numeric_data()
    artifact = tmp_path / "xgboost.ubj"

    result = fit_xgboost(
        x_train,
        y_train,
        x_validation,
        y_validation,
        params={
            "learning_rate": 0.08,
            "max_depth": 4,
            "min_child_weight": 2.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "max_bin": 64,
            "num_boost_round": 40,
            "early_stopping_rounds": 8,
        },
        seed=7,
        threads=2,
        artifact_path=artifact,
        feature_names=tuple(f"f{i}" for i in range(x_train.shape[1])),
        logger=RunLogger("xgboost-test", tmp_path / "logs"),
    )

    _assert_predictions(result.prediction, len(x_validation))
    assert result.best_iteration >= 1
    assert artifact.is_file()


def test_catboost_wrapper_uses_native_categorical_semantics(tmp_path: Path) -> None:
    x_train, y_train, x_validation, y_validation = _numeric_data()
    train = pd.DataFrame(x_train[:, :6], columns=[f"f{i}" for i in range(6)])
    validation = pd.DataFrame(
        x_validation[:, :6],
        columns=[f"f{i}" for i in range(6)],
    )
    train["category"] = np.arange(len(train), dtype=np.int32) % 4
    validation["category"] = np.arange(len(validation), dtype=np.int32) % 5
    artifact = tmp_path / "catboost.cbm"

    result = fit_catboost(
        train,
        y_train,
        validation,
        y_validation,
        categorical_indices=(6,),
        params={
            "iterations": 40,
            "early_stopping_rounds": 8,
            "learning_rate": 0.08,
            "depth": 5,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.25,
            "rsm": 0.9,
            "subsample": 0.9,
        },
        seed=7,
        threads=2,
        artifact_path=artifact,
        logger=RunLogger("catboost-test", tmp_path / "logs"),
    )

    _assert_predictions(result.prediction, len(validation))
    assert result.best_iteration >= 1
    assert artifact.is_file()
    assert not artifact.with_suffix(".json").exists()
