"""Release contracts: temporal safety, portable encoding and native-model round trips."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.data import FeatureRef
from home_credit.modeling.release import (
    checked_member,
    evaluation_report,
    fit_encoder,
    fit_frozen_model,
    frozen_iterations,
    predict_components,
    read_object,
    save_json,
    transform,
    validate_features,
    validate_probabilities,
    verify_file,
)
from home_credit.observability.logging import RunLogger


@pytest.fixture
def features():
    return (
        FeatureRef("amount", "static_depth0", "static", 0, "double", False),
        FeatureRef("category", "static_depth0", "static", 0, "string", True),
    )


def test_encoder_is_training_only_and_roundtrips(tmp_path, features):
    train = pl.DataFrame({"amount": [1.0, None, 3.0, 4.0], "category": ["a", "a", "b", None]})
    encoder = fit_encoder(train, features)
    path = tmp_path / "encoder.json"
    save_json(path, encoder)
    verify_file(path, sha256_file(path))
    restored = read_object(path)
    probe = pl.DataFrame({"category": ["b", "unseen", None], "amount": [8.0, 9.0, 10.0]})
    actual = transform(probe, features, restored)
    np.testing.assert_array_equal(actual[:, 1], [0.25, 0, 0.25])
    assert restored["fit_rows"] == 4
    assert restored == encoder
    assert actual.dtype == np.float32
    with pytest.raises(ValueError, match="order"):
        transform(probe, features[::-1], restored)


@pytest.mark.parametrize("mutation", ["missing_map", "bad_mass", "nan", "schema", "order"])
def test_corrupt_encoder_rejected(features, mutation):
    frame = pl.DataFrame({"amount": [1, 2], "category": ["a", "b"]})
    encoder = fit_encoder(frame, features)
    if mutation == "missing_map":
        encoder["frequency_maps"] = {}
    elif mutation == "bad_mass":
        encoder["frequency_maps"]["category"]["a"] = 0.1
    elif mutation == "nan":
        encoder["frequency_maps"]["category"]["a"] = float("nan")
    elif mutation == "schema":
        encoder["schema_version"] = 12
    else:
        encoder["feature_names"].reverse()
    with pytest.raises(ValueError):
        transform(frame, features, encoder)


@pytest.mark.parametrize(
    "name", ["target", "case_id", "WEEK_NUM", "MONTH", "score", "base__decision_day"]
)
def test_forbidden_predictors_rejected(name):
    with pytest.raises(ValueError):
        validate_features((FeatureRef(name, "base_depth0", "base", 0, "int", False),))


def test_missing_schema_empty_fit_and_missing_token_fail(features):
    with pytest.raises(ValueError, match="empty"):
        fit_encoder(pl.DataFrame(schema={"amount": pl.Float32, "category": pl.String}), features)
    with pytest.raises(ValueError, match="schema"):
        fit_encoder(pl.DataFrame({"amount": [1]}), features)
    with pytest.raises(ValueError, match="reserved"):
        fit_encoder(pl.DataFrame({"amount": [1], "category": ["__HC_MISSING__"]}), features)
    with pytest.raises(ValueError, match="duplicate"):
        validate_features(features + features[:1])


def test_fixed_iterations_derive_from_all_saved_development_folds():
    state = {
        "schema_version": 1,
        "identity": {"smoke": False},
        "folds": {
            f"lightgbm:fold_{i}": {"fold": i, "metrics": {"best_iteration": n}}
            for i, n in enumerate([1024, 1435, 1995, 2199, 1852], 1)
        },
    }
    assert frozen_iterations(state) == 1852
    state["identity"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke"):
        frozen_iterations(state)
    state["identity"]["smoke"] = False
    state["folds"]["lightgbm:fold_3"]["metrics"]["best_iteration"] = True
    with pytest.raises(ValueError, match="rounds"):
        frozen_iterations(state)


@pytest.mark.parametrize(
    "values", [[float("nan")], [float("inf")], [-0.1], [1.1], [[0.1]], [0.1, 0.2]]
)
def test_probability_contract(values):
    with pytest.raises(ValueError):
        validate_probabilities(np.asarray(values), 1)


def test_safe_artifacts_and_nonportable_json(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("native model")
    verify_file(path, sha256_file(path))
    with pytest.raises(ValueError, match="digest"):
        verify_file(path, "0" * 64)
    for name in ("../other", "/etc/passwd"):
        with pytest.raises(ValueError, match="unsafe"):
            checked_member(tmp_path, name)
    path = tmp_path / "metadata.json"
    path.write_text("[1]")
    with pytest.raises(ValueError):
        read_object(path)
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        read_object(path)


def test_evaluation_requires_every_declared_week_and_both_classes():
    frame = pl.DataFrame({"WEEK_NUM": [73, 73, 74, 74], "target": [0, 1, 0, 1]})
    prediction = np.asarray([0.1, 0.9, 0.2, 0.8])
    result = evaluation_report(frame, prediction, expected_weeks=[73, 74])
    assert result["metrics"]["stability_score"] == pytest.approx(1.0)
    assert len(result["weekly"]) == 2
    with pytest.raises(ValueError, match="coverage"):
        evaluation_report(frame, prediction, expected_weeks=[73, 74, 75])
    with pytest.raises(ValueError, match="single-class"):
        evaluation_report(
            frame.with_columns(pl.Series("target", [0, 0, 0, 1])),
            prediction,
            expected_weeks=[73, 74],
        )
    constant = evaluation_report(frame, np.full(4, 0.5), expected_weeks=[73, 74])
    assert constant["reliability"] == [
        {"rows": 4, "mean_prediction": 0.5, "observed_default_rate": 0.5}
    ]


def test_native_model_fixed_rounds_reload_and_predict(tmp_path, features):
    rng = np.random.default_rng(509)
    frame = pl.DataFrame({"amount": rng.normal(size=500), "category": ["a", "b"] * 250})
    encoder = fit_encoder(frame, features)
    matrix = transform(frame, features, encoder)
    target = (matrix[:, 0] + rng.normal(size=500) > 0).astype(np.int8)
    parameters = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/model_benchmark.json").read_text()
    )["models"]["lightgbm"]
    parameters = copy.deepcopy(parameters)
    parameters.update(min_data_in_leaf=10, num_leaves=7)
    path = tmp_path / "lightgbm.txt"
    events = []
    receipt = fit_frozen_model(
        matrix,
        target,
        features,
        params=parameters,
        rounds=12,
        seed=9,
        threads=1,
        path=path,
        logger=RunLogger("release-test", tmp_path / "logs"),
        check_lease=lambda: events.append("checked"),
    )
    assert receipt["actual_rounds"] == 12
    assert receipt["reload_max_absolute_error"] == 0
    assert receipt["early_stopping_used"] is False
    prediction = predict_components(matrix, features, {"model": path}, {"model": 1.0})
    validate_probabilities(prediction, len(target))
    assert len(events) >= 12
    with pytest.raises(ValueError, match="weight"):
        predict_components(matrix, features, {"model": path}, {"model": 0.5})
    with pytest.raises(ValueError, match="order"):
        predict_components(matrix, features[::-1], {"model": path}, {"model": 1.0})
