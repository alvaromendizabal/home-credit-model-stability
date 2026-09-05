from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.runner import (
    _evaluate_predictions,
    _load_protocol,
    _protocol_folds,
    _smoke_config,
)


def _protocol() -> dict[str, object]:
    return {
        "protocol_sha256": "protocol",
        "outer_holdout": {
            "locked": True,
            "validation_week_min": 73,
            "validation_week_max": 91,
        },
        "inner_temporal_cv": {
            "folds": [
                {
                    "fold": 1,
                    "train_week_min": 0,
                    "train_week_max": 32,
                    "validation_week_min": 33,
                    "validation_week_max": 40,
                },
                {
                    "fold": 2,
                    "train_week_min": 0,
                    "train_week_max": 40,
                    "validation_week_min": 41,
                    "validation_week_max": 48,
                },
            ]
        },
    }


def test_protocol_parser_matches_frozen_schema(tmp_path: Path) -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))
    path = tmp_path / "validation_protocol.json"
    path.write_text(json.dumps(_protocol()) + "\n", encoding="utf-8")

    protocol = _load_protocol(
        path,
        expected_sha256="protocol",
        config=config,
    )
    folds = _protocol_folds(protocol)

    assert folds[0] == {
        "fold": 1,
        "train_week_min": 0,
        "train_week_max": 32,
        "validation_week_min": 33,
        "validation_week_max": 40,
    }
    assert folds[1]["validation_week_min"] == 41


def test_protocol_guard_rejects_screening_that_reaches_inner_validation(
    tmp_path: Path,
) -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))
    unsafe = replace(
        config,
        screening=replace(
            config.screening,
            validation_week_max=33,
        ),
    )
    path = tmp_path / "validation_protocol.json"
    path.write_text(json.dumps(_protocol()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="screening overlaps"):
        _load_protocol(
            path,
            expected_sha256="protocol",
            config=unsafe,
        )


def test_prediction_metrics_include_stability_and_probability_metrics() -> None:
    y_true = np.asarray(
        [0, 1, 0, 1, 0, 1, 0, 1],
        dtype=np.int8,
    )
    prediction = np.asarray(
        [0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.25, 0.85],
        dtype=np.float64,
    )
    weeks = np.asarray(
        [33, 33, 34, 34, 35, 35, 36, 36],
        dtype=np.int32,
    )

    metrics = _evaluate_predictions(y_true, prediction, weeks)

    assert metrics["auc"] == 1.0
    assert metrics["mean_gini"] == 1.0
    assert metrics["stability_score"] == pytest.approx(1.0)
    assert metrics["brier_score"] > 0.0
    assert metrics["log_loss"] > 0.0


def test_smoke_config_keeps_temporal_guards_and_caps_work() -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))

    smoke = _smoke_config(config)

    assert smoke.outer_holdout_guard_week_min == 73
    assert smoke.screening.validation_week_max == 32
    assert smoke.screening.max_features <= 128
    assert smoke.screening.max_train_rows <= 50_000
    assert smoke.screening.max_validation_rows <= 25_000
