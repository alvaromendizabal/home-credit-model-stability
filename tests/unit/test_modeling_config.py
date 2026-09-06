from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from home_credit.modeling.config import BenchmarkConfig


def test_benchmark_config_is_safe_and_complete() -> None:
    path = Path("configs/model_benchmark.json")
    config, digest = BenchmarkConfig.load(path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "2182868d53525fca56106a5d7c9f344438adb56f599ce507888f9cc5862746fa"
    assert config.outer_holdout_guard_week_min == 73
    assert config.screening.train_week_min == 0
    assert config.screening.validation_week_max == 32
    assert config.enabled_model_names == (
        "linear_logistic",
        "lightgbm",
        "xgboost",
        "catboost",
    )


def test_direct_time_predictors_are_excluded() -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))

    required = {
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

    assert required.issubset(config.excluded_predictors)


def test_screening_rejects_outer_holdout_contact() -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))
    unsafe = replace(
        config,
        screening=replace(
            config.screening,
            validation_week_max=73,
        ),
    )

    with pytest.raises(ValueError, match="outer holdout"):
        unsafe.validate()


def test_config_round_trip_has_no_hidden_model_keys(tmp_path: Path) -> None:
    config, _ = BenchmarkConfig.load(Path("configs/model_benchmark.json"))
    payload = json.loads(Path("configs/model_benchmark.json").read_text(encoding="utf-8"))
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded, _ = BenchmarkConfig.load(path)

    assert asdict(loaded) == asdict(config)
