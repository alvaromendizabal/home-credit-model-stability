from __future__ import annotations

from pathlib import Path

import pytest

from home_credit.modeling.state import (
    BenchmarkIdentity,
    BenchmarkStateStore,
    FoldReceipt,
    sha256_file,
)


def _identity() -> BenchmarkIdentity:
    return BenchmarkIdentity(
        git_commit="abc",
        feature_manifest_sha256="manifest",
        validation_protocol_sha256="protocol",
        benchmark_config_sha256="config",
        feature_screen_sha256="screen",
        smoke=False,
    )


def test_state_round_trip_verifies_artifact_hashes(tmp_path: Path) -> None:
    model = tmp_path / "models" / "model.txt"
    prediction = tmp_path / "predictions" / "pred.parquet"
    model.parent.mkdir(parents=True)
    prediction.parent.mkdir(parents=True)
    model.write_text("model\n", encoding="utf-8")
    prediction.write_bytes(b"prediction")

    state = BenchmarkStateStore(
        tmp_path / "benchmark_state.json",
        identity=_identity(),
    )
    state.record(
        FoldReceipt(
            model="lightgbm",
            fold=1,
            prediction_path="predictions/pred.parquet",
            prediction_sha256=sha256_file(prediction),
            model_path="models/model.txt",
            model_sha256=sha256_file(model),
            metrics={"stability_score": 0.5},
        )
    )

    receipt = state.receipt("lightgbm", 1, output_root=tmp_path)

    assert receipt is not None
    assert receipt.metrics["stability_score"] == 0.5


def test_state_detects_artifact_corruption(tmp_path: Path) -> None:
    model = tmp_path / "model.txt"
    prediction = tmp_path / "prediction.parquet"
    model.write_text("model\n", encoding="utf-8")
    prediction.write_bytes(b"prediction")

    state = BenchmarkStateStore(
        tmp_path / "benchmark_state.json",
        identity=_identity(),
    )
    state.record(
        FoldReceipt(
            model="xgboost",
            fold=1,
            prediction_path="prediction.parquet",
            prediction_sha256=sha256_file(prediction),
            model_path="model.txt",
            model_sha256=sha256_file(model),
            metrics={"auc": 0.7},
        )
    )
    prediction.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="hash mismatch"):
        state.receipt("xgboost", 1, output_root=tmp_path)
