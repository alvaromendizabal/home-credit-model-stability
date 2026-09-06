from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.runner import (
    _evaluate_predictions,
    _limit_pending_models,
    _load_protocol,
    _protocol_folds,
    _remaining_checkpoint_budget,
    _smoke_config,
    _validate_checkpoint_budget,
    _verified_model_fold_count,
)
from home_credit.modeling.state import (
    BenchmarkIdentity,
    BenchmarkStateStore,
    FoldReceipt,
    sha256_file,
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


def test_checkpoint_budget_validation() -> None:
    assert _validate_checkpoint_budget(None) is None
    assert _validate_checkpoint_budget(1) == 1

    with pytest.raises(ValueError, match="must be positive"):
        _validate_checkpoint_budget(0)


def test_checkpoint_budget_limits_pending_models() -> None:
    pending = ["linear_logistic", "lightgbm", "xgboost", "catboost"]

    assert _limit_pending_models(pending, max_new_checkpoints=None) == pending
    assert _limit_pending_models(pending, max_new_checkpoints=1) == ["linear_logistic"]
    assert _limit_pending_models(pending, max_new_checkpoints=2) == [
        "linear_logistic",
        "lightgbm",
    ]


def test_remaining_checkpoint_budget_never_goes_negative() -> None:
    assert _remaining_checkpoint_budget(None, completed=10) is None
    assert _remaining_checkpoint_budget(3, completed=0) == 3
    assert _remaining_checkpoint_budget(3, completed=2) == 1
    assert _remaining_checkpoint_budget(3, completed=3) == 0
    assert _remaining_checkpoint_budget(3, completed=5) == 0

    with pytest.raises(ValueError, match="cannot be negative"):
        _remaining_checkpoint_budget(3, completed=-1)


def test_verified_model_fold_count_ignores_missing_artifacts(tmp_path: Path) -> None:
    identity = BenchmarkIdentity(
        git_commit="abc",
        feature_manifest_sha256="manifest",
        validation_protocol_sha256="protocol",
        benchmark_config_sha256="config",
        feature_screen_sha256="screen",
        smoke=False,
    )
    state = BenchmarkStateStore(
        tmp_path / "benchmark_state.json",
        identity=identity,
    )

    model_path = tmp_path / "models" / "lightgbm" / "fold_1.txt"
    prediction_path = tmp_path / "predictions" / "lightgbm" / "fold_1.parquet"
    model_path.parent.mkdir(parents=True)
    prediction_path.parent.mkdir(parents=True)
    model_path.write_text("model\n", encoding="utf-8")
    prediction_path.write_bytes(b"prediction")

    state.record(
        FoldReceipt(
            model="lightgbm",
            fold=1,
            prediction_path="predictions/lightgbm/fold_1.parquet",
            prediction_sha256=sha256_file(prediction_path),
            model_path="models/lightgbm/fold_1.txt",
            model_sha256=sha256_file(model_path),
            metrics={"stability_score": 0.5},
        )
    )
    state.record(
        FoldReceipt(
            model="xgboost",
            fold=1,
            prediction_path="predictions/xgboost/fold_1.parquet",
            prediction_sha256="missing",
            model_path="models/xgboost/fold_1.ubj",
            model_sha256="missing",
            metrics={"stability_score": 0.4},
        )
    )

    count = _verified_model_fold_count(
        state,
        folds=(
            {
                "fold": 1,
                "train_week_min": 0,
                "train_week_max": 32,
                "validation_week_min": 33,
                "validation_week_max": 40,
            },
        ),
        model_names=("lightgbm", "xgboost"),
        output_root=tmp_path,
    )

    assert count == 1
