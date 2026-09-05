from __future__ import annotations

import numpy as np
import polars as pl

from home_credit.modeling.data import FeatureRef
from home_credit.modeling.encoding import (
    catboost_encode,
    frequency_encode,
    standardize_for_linear,
)


def _features() -> tuple[FeatureRef, ...]:
    return (
        FeatureRef(
            name="category",
            block="demo_depth0",
            family="demo",
            depth=0,
            dtype="string",
            categorical=True,
        ),
        FeatureRef(
            name="numeric",
            block="demo_depth0",
            family="demo",
            depth=0,
            dtype="double",
            categorical=False,
        ),
    )


def test_frequency_encoding_is_train_only_and_handles_unseen_categories() -> None:
    train = pl.DataFrame(
        {
            "category": ["a", "a", "b", None],
            "numeric": [1.0, 2.0, None, 4.0],
        }
    )
    validation = pl.DataFrame(
        {
            "category": ["a", "c", None],
            "numeric": [5.0, float("inf"), 7.0],
        }
    )

    encoded = frequency_encode(train, validation, _features())

    assert encoded.train.shape == (4, 2)
    assert encoded.validation.shape == (3, 2)
    assert np.isclose(encoded.validation[0, 0], 0.5)
    assert encoded.validation[1, 0] == 0.0
    assert np.isclose(encoded.validation[2, 0], 0.25)
    assert np.isnan(encoded.validation[1, 1])


def test_catboost_category_codes_are_fit_on_train_only() -> None:
    train = pl.DataFrame(
        {
            "category": ["b", "a", None],
            "numeric": [1.0, 2.0, 3.0],
        }
    )
    validation = pl.DataFrame(
        {
            "category": ["a", "never_seen", None],
            "numeric": [4.0, 5.0, 6.0],
        }
    )

    encoded = catboost_encode(train, validation, _features())

    assert encoded.categorical_indices == (0,)
    assert encoded.validation["category"].tolist()[0] > 0
    assert encoded.validation["category"].tolist()[1] == 0
    assert encoded.validation["category"].tolist()[2] > 0


def test_linear_standardization_is_finite_and_train_fitted() -> None:
    train = np.asarray(
        [
            [1.0, np.nan, 3.0],
            [2.0, 5.0, 3.0],
            [3.0, 7.0, 3.0],
        ],
        dtype=np.float32,
    )
    validation = np.asarray(
        [[4.0, np.inf, 3.0]],
        dtype=np.float32,
    )

    train_scaled, validation_scaled, medians, means, scales = standardize_for_linear(
        train,
        validation,
    )

    assert np.isfinite(train_scaled).all()
    assert np.isfinite(validation_scaled).all()
    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert medians.shape == means.shape == scales.shape == (3,)
    assert scales[2] == 1.0
