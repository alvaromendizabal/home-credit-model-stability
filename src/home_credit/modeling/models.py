"""Deterministic model-family training wrappers for temporal benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier

from home_credit.observability.logging import RunLogger


@dataclass(frozen=True, slots=True)
class ModelFitResult:
    """Predictions and artifact metadata for one fold fit."""

    prediction: NDArray[np.float64]
    best_iteration: int
    artifact_path: Path


def fit_linear_logistic(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int8],
    x_validation: NDArray[np.float32],
    *,
    params: dict[str, Any],
    seed: int,
    artifact_path: Path,
    feature_names: tuple[str, ...],
    medians: NDArray[np.float32],
    means: NDArray[np.float32],
    scales: NDArray[np.float32],
    logger: RunLogger,
) -> ModelFitResult:
    """Fit an averaged-SGD logistic baseline on standardized dense features."""
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(params["alpha"]),
        max_iter=int(params["max_iter"]),
        tol=float(params["tol"]),
        shuffle=True,
        random_state=seed,
        average=True,
        class_weight=None,
        n_jobs=1,
    )
    logger.event(
        "model_fit_started",
        model="linear_logistic",
        train_rows=x_train.shape[0],
        validation_rows=x_validation.shape[0],
        features=x_train.shape[1],
    )
    model.fit(x_train, y_train)
    prediction = model.predict_proba(x_validation)[:, 1].astype(np.float64)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        classes=np.asarray(model.classes_),
        feature_names=np.asarray(feature_names),
        medians=medians,
        means=means,
        scales=scales,
    )
    logger.event(
        "model_fit_completed",
        model="linear_logistic",
        iterations=int(model.n_iter_),
    )
    return ModelFitResult(
        prediction=prediction,
        best_iteration=int(model.n_iter_),
        artifact_path=artifact_path,
    )


def fit_lightgbm(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int8],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int8],
    *,
    params: dict[str, Any],
    seed: int,
    threads: int,
    artifact_path: Path,
    feature_names: tuple[str, ...],
    logger: RunLogger,
) -> ModelFitResult:
    """Fit one deterministic LightGBM histogram booster."""
    training_params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "feature_fraction": float(params["feature_fraction"]),
        "bagging_fraction": float(params["bagging_fraction"]),
        "bagging_freq": int(params["bagging_freq"]),
        "lambda_l1": float(params["lambda_l1"]),
        "lambda_l2": float(params["lambda_l2"]),
        "max_bin": int(params["max_bin"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        x_validation,
        label=y_validation,
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    logger.event(
        "model_fit_started",
        model="lightgbm",
        train_rows=x_train.shape[0],
        validation_rows=x_validation.shape[0],
        features=x_train.shape[1],
    )
    booster = lgb.train(
        training_params,
        train_set,
        num_boost_round=int(params["num_boost_round"]),
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(
                int(params["early_stopping_rounds"]),
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(booster.best_iteration or params["num_boost_round"])
    prediction = np.asarray(
        booster.predict(x_validation, num_iteration=best_iteration),
        dtype=np.float64,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(artifact_path), num_iteration=best_iteration)
    logger.event(
        "model_fit_completed",
        model="lightgbm",
        best_iteration=best_iteration,
    )
    return ModelFitResult(
        prediction=prediction,
        best_iteration=best_iteration,
        artifact_path=artifact_path,
    )


def fit_xgboost(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int8],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int8],
    *,
    params: dict[str, Any],
    seed: int,
    threads: int,
    artifact_path: Path,
    feature_names: tuple[str, ...],
    logger: RunLogger,
) -> ModelFitResult:
    """Fit one CPU histogram XGBoost model through QuantileDMatrix."""
    max_bin = int(params["max_bin"])
    dtrain = xgb.QuantileDMatrix(
        x_train,
        label=y_train,
        feature_names=list(feature_names),
        max_bin=max_bin,
    )
    dvalidation = xgb.QuantileDMatrix(
        x_validation,
        label=y_validation,
        feature_names=list(feature_names),
        max_bin=max_bin,
        ref=dtrain,
    )
    training_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "max_bin": max_bin,
        "seed": seed,
        "nthread": threads,
    }
    logger.event(
        "model_fit_started",
        model="xgboost",
        train_rows=x_train.shape[0],
        validation_rows=x_validation.shape[0],
        features=x_train.shape[1],
    )
    booster = xgb.train(
        training_params,
        dtrain,
        num_boost_round=int(params["num_boost_round"]),
        evals=[(dvalidation, "validation")],
        early_stopping_rounds=int(params["early_stopping_rounds"]),
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", 0)) + 1
    prediction = np.asarray(
        booster.predict(dvalidation, iteration_range=(0, best_iteration)),
        dtype=np.float64,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(artifact_path))
    logger.event(
        "model_fit_completed",
        model="xgboost",
        best_iteration=best_iteration,
    )
    return ModelFitResult(
        prediction=prediction,
        best_iteration=best_iteration,
        artifact_path=artifact_path,
    )


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: NDArray[np.int8],
    x_validation: pd.DataFrame,
    y_validation: NDArray[np.int8],
    *,
    categorical_indices: tuple[int, ...],
    params: dict[str, Any],
    seed: int,
    threads: int,
    artifact_path: Path,
    logger: RunLogger,
) -> ModelFitResult:
    """Fit CatBoost with train-only categorical codes and native cat semantics."""
    model = cb.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(params["iterations"]),
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_strength=float(params["random_strength"]),
        rsm=float(params["rsm"]),
        bootstrap_type="Bernoulli",
        subsample=float(params["subsample"]),
        random_seed=seed,
        thread_count=threads,
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
    )
    train_pool = cb.Pool(
        x_train,
        label=y_train,
        cat_features=list(categorical_indices),
        feature_names=list(x_train.columns),
    )
    validation_pool = cb.Pool(
        x_validation,
        label=y_validation,
        cat_features=list(categorical_indices),
        feature_names=list(x_validation.columns),
    )
    logger.event(
        "model_fit_started",
        model="catboost",
        train_rows=len(x_train),
        validation_rows=len(x_validation),
        features=x_train.shape[1],
        categorical_features=len(categorical_indices),
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=int(params["early_stopping_rounds"]),
    )
    best_iteration = int(model.get_best_iteration()) + 1
    prediction = np.asarray(
        model.predict_proba(validation_pool)[:, 1],
        dtype=np.float64,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_path), format="cbm")
    logger.event(
        "model_fit_completed",
        model="catboost",
        best_iteration=best_iteration,
    )
    return ModelFitResult(
        prediction=prediction,
        best_iteration=best_iteration,
        artifact_path=artifact_path,
    )
