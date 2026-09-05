"""Fold-fitted categorical encoders and numeric matrix preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from home_credit.modeling.data import FeatureRef

_MISSING_TOKEN = "__HC_MISSING__"


@dataclass(frozen=True, slots=True)
class FrequencyEncoded:
    """Numeric matrices produced with train-only categorical frequencies."""

    train: NDArray[np.float32]
    validation: NDArray[np.float32]
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatBoostEncoded:
    """Memory-conscious mixed-dtype frames for CatBoost native categories."""

    train: pd.DataFrame
    validation: pd.DataFrame
    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]


def frequency_encode(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    features: tuple[FeatureRef, ...],
) -> FrequencyEncoded:
    """Encode categoricals from training frequencies and cast numerics to float32."""
    names = tuple(feature.name for feature in features)
    frequency_maps = _fit_frequency_maps(train, features)
    train_numeric = _frequency_frame(train, features, frequency_maps)
    validation_numeric = _frequency_frame(validation, features, frequency_maps)

    train_array = np.asarray(train_numeric.to_numpy(), dtype=np.float32)
    validation_array = np.asarray(validation_numeric.to_numpy(), dtype=np.float32)
    if train_array.ndim != 2 or validation_array.ndim != 2:
        raise ValueError("encoded feature matrices must be two-dimensional")
    if train_array.shape[1] != len(features) or validation_array.shape[1] != len(features):
        raise ValueError("encoded feature matrix width does not match feature plan")
    return FrequencyEncoded(
        train=train_array,
        validation=validation_array,
        feature_names=names,
    )


def catboost_encode(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    features: tuple[FeatureRef, ...],
) -> CatBoostEncoded:
    """Fit train-only category codes while retaining categorical semantics."""
    category_maps = _fit_category_maps(train, features)
    train_frame = _category_code_frame(train, features, category_maps)
    validation_frame = _category_code_frame(validation, features, category_maps)

    categorical_indices = tuple(
        index for index, feature in enumerate(features) if feature.categorical
    )
    names = tuple(feature.name for feature in features)

    train_pandas = train_frame.to_pandas(use_pyarrow_extension_array=False)
    validation_pandas = validation_frame.to_pandas(use_pyarrow_extension_array=False)

    for index in categorical_indices:
        column = names[index]
        train_pandas[column] = train_pandas[column].astype("int32")
        validation_pandas[column] = validation_pandas[column].astype("int32")

    return CatBoostEncoded(
        train=train_pandas,
        validation=validation_pandas,
        feature_names=names,
        categorical_indices=categorical_indices,
    )


def standardize_for_linear(
    train: NDArray[np.float32],
    validation: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Median-impute and standardize a dense linear-model matrix using train only."""
    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("linear matrices must be two-dimensional")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("linear train/validation widths must match")
    if train.shape[1] == 0:
        raise ValueError("linear matrix must contain at least one feature")

    train_work = np.asarray(train, dtype=np.float32).copy()
    validation_work = np.asarray(validation, dtype=np.float32).copy()
    train_work[~np.isfinite(train_work)] = np.nan
    validation_work[~np.isfinite(validation_work)] = np.nan

    medians = np.nanmedian(train_work, axis=0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    _fill_nan_columns(train_work, medians)
    _fill_nan_columns(validation_work, medians)

    means = train_work.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = train_work.std(axis=0, dtype=np.float64).astype(np.float32)
    scales = np.where(scales > 1e-6, scales, 1.0).astype(np.float32)

    train_work = ((train_work - means) / scales).astype(np.float32, copy=False)
    validation_work = ((validation_work - means) / scales).astype(np.float32, copy=False)
    return train_work, validation_work, medians, means, scales


def _fit_frequency_maps(
    train: pl.DataFrame,
    features: tuple[FeatureRef, ...],
) -> dict[str, dict[str, float]]:
    row_count = train.height
    if row_count == 0:
        raise ValueError("cannot fit frequency maps on an empty training frame")

    result: dict[str, dict[str, float]] = {}
    for feature in features:
        if not feature.categorical:
            continue
        values = train.select(
            pl.col(feature.name)
            .cast(pl.String, strict=False)
            .fill_null(_MISSING_TOKEN)
            .alias(feature.name)
        )
        counts = values.group_by(feature.name).len()
        categories = counts.get_column(feature.name).to_list()
        frequencies = counts.get_column("len").cast(pl.Float64) / float(row_count)
        result[feature.name] = {
            str(category): float(frequency)
            for category, frequency in zip(
                categories,
                frequencies.to_list(),
                strict=True,
            )
        }
    return result


def _fit_category_maps(
    train: pl.DataFrame,
    features: tuple[FeatureRef, ...],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for feature in features:
        if not feature.categorical:
            continue
        values = (
            train.get_column(feature.name)
            .cast(pl.String, strict=False)
            .fill_null(_MISSING_TOKEN)
            .unique()
            .sort()
            .to_list()
        )
        result[feature.name] = {str(value): index for index, value in enumerate(values, start=1)}
    return result


def _frequency_frame(
    frame: pl.DataFrame,
    features: tuple[FeatureRef, ...],
    frequency_maps: Mapping[str, Mapping[str, float]],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for feature in features:
        if feature.categorical:
            mapping = frequency_maps[feature.name]
            expressions.append(
                pl.col(feature.name)
                .cast(pl.String, strict=False)
                .fill_null(_MISSING_TOKEN)
                .replace_strict(
                    mapping,
                    default=0.0,
                    return_dtype=pl.Float32,
                )
                .alias(feature.name)
            )
        else:
            numeric = pl.col(feature.name).cast(pl.Float64, strict=False)
            expressions.append(
                pl.when(numeric.is_finite())
                .then(numeric)
                .otherwise(None)
                .cast(pl.Float32)
                .alias(feature.name)
            )
    return frame.select(expressions)


def _category_code_frame(
    frame: pl.DataFrame,
    features: tuple[FeatureRef, ...],
    category_maps: Mapping[str, Mapping[str, int]],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for feature in features:
        if feature.categorical:
            mapping = category_maps[feature.name]
            expressions.append(
                pl.col(feature.name)
                .cast(pl.String, strict=False)
                .fill_null(_MISSING_TOKEN)
                .replace_strict(
                    mapping,
                    default=0,
                    return_dtype=pl.Int32,
                )
                .alias(feature.name)
            )
        else:
            numeric = pl.col(feature.name).cast(pl.Float64, strict=False)
            expressions.append(
                pl.when(numeric.is_finite())
                .then(numeric)
                .otherwise(None)
                .cast(pl.Float32)
                .alias(feature.name)
            )
    return frame.select(expressions)


def _fill_nan_columns(matrix: NDArray[np.float32], values: NDArray[np.float32]) -> None:
    rows, columns = np.where(np.isnan(matrix))
    if rows.size:
        matrix[rows, columns] = values[columns]
