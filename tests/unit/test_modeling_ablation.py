from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from home_credit.modeling.ablation import compare_predictions, frozen_features, write_report
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import FeatureSnapshot
from home_credit.modeling.screening import feature_refs_from_payload


def snapshot() -> FeatureSnapshot:
    source = json.loads(Path("configs/benchmark_features.json").read_text())
    return FeatureSnapshot(
        root=Path("."),
        manifest_sha256=source["identity"]["feature_manifest_sha256"],
        protocol_sha256=source["identity"]["validation_protocol_sha256"],
        recipe_sha256="recipe",
        execution_sha256="execution",
        feature_git_commit="source",
        train_blocks=(),
        test_blocks=(),
        features=feature_refs_from_payload(source),
    )


@pytest.mark.parametrize(
    "name,count",
    [
        ("control", 700),
        ("without_credit_bureau_a", 427),
        ("without_previous_applications", 516),
        ("without_depth2", 651),
    ],
)
def test_frozen_ablation_retains_original_order_and_exact_counts(name, count):
    config, _ = BenchmarkConfig.load(Path(f"configs/ablations/{name}.json"))
    original = snapshot()
    selected = frozen_features(config, original)
    assert len(selected) == count
    assert selected == tuple(
        f for f in original.features if f.block not in config.feature_selection["exclude_blocks"]
    )
    assert config.enabled_model_names == ("lightgbm",)
    assert (
        config.model("lightgbm").params
        == BenchmarkConfig.load(Path("configs/model_benchmark.json"))[0].model("lightgbm").params
    )


def test_frozen_features_reject_changed_snapshot_and_source_hash():
    config, _ = BenchmarkConfig.load(Path("configs/ablations/control.json"))
    with pytest.raises(ValueError, match="feature identity"):
        frozen_features(config, replace(snapshot(), manifest_sha256="different"))
    with pytest.raises(ValueError, match="hash mismatch"):
        frozen_features(
            replace(config, feature_selection={**config.feature_selection, "sha256": "0" * 64}),
            snapshot(),
        )


@pytest.mark.parametrize("blocks", [["unknown"], ["static_depth0", "static_depth0"], [5]])
def test_ablation_rejects_invalid_or_unknown_blocks(blocks):
    config, _ = BenchmarkConfig.load(Path("configs/ablations/control.json"))
    config = replace(
        config, feature_selection={**config.feature_selection, "exclude_blocks": blocks}
    )
    with pytest.raises(ValueError):
        config.validate()
        frozen_features(config, snapshot())


def predictions():
    return pl.DataFrame(
        {
            "case_id": list(range(16)),
            "WEEK_NUM": np.repeat([33, 34, 35, 36], 4),
            "target": [0, 1, 0, 1] * 4,
            "prediction": [0.1, 0.9, 0.2, 0.8] * 4,
        }
    )


def folds():
    return [{"fold": 1, "validation_week_min": 33, "validation_week_max": 36}]


def test_comparison_recomputes_and_accepts_row_reordering(tmp_path):
    frame = predictions()
    result = compare_predictions({"control": frame, "same": frame.reverse()}, folds())
    assert result["rows"][0]["mean_fold_stability"] == pytest.approx(1)
    assert all(row["delta_vs_control"] == 0 for row in result["rows"])
    assert result["outer_holdout_touched"] is False
    write_report(result, tmp_path / "report.html")
    page = (tmp_path / "report.html").read_text()
    assert "plotly.js" in page and "oof brier score" in page


@pytest.mark.parametrize("kind", ["cases", "labels", "holdout", "missing_week", "nonfinite"])
def test_comparison_rejects_invalid_evidence(kind):
    frame = predictions()
    bad = frame
    if kind == "cases":
        bad = frame.with_columns(pl.col("case_id") + 100)
    if kind == "labels":
        bad = frame.with_columns((1 - pl.col("target")).alias("target"))
    if kind == "holdout":
        bad = frame.with_columns(pl.lit(73).alias("WEEK_NUM"))
    if kind == "missing_week":
        bad = frame.filter(pl.col("WEEK_NUM") != 36)
    if kind == "nonfinite":
        bad = frame.with_columns(pl.lit(float("nan")).alias("prediction"))
    with pytest.raises(ValueError):
        compare_predictions({"control": frame, "bad": bad}, folds())


def test_native_lightgbm_ablation_checkpoint_resume(tmp_path, monkeypatch):
    """Fit a real CPU model, then prove a resumed runner never calls fit again."""
    import home_credit.modeling.runner as module
    from home_credit.modeling.runner import BenchmarkRunner
    from home_credit.modeling.state import sha256_file

    original = snapshot()
    monkeypatch.setattr(module.FeatureSnapshot, "load", lambda *args, **kwargs: original)
    monkeypatch.setattr(module, "_git_commit", lambda: "test-commit")

    def load_frames(_snapshot, selected_features, **kwargs):
        assert kwargs["validation_week_max"] == 40
        rng = np.random.default_rng(10)

        def data(weeks, start):
            n = len(weeks)
            target = np.arange(n) % 2
            values = {f.name: rng.normal(size=n) + target for f in selected_features}
            # Synthetic numeric data exercises real encoding, fit, metrics and checkpoint writing.
            return pl.DataFrame(
                {
                    "case_id": np.arange(start, start + n),
                    "WEEK_NUM": weeks,
                    "target": target,
                    **values,
                }
            )

        return data(np.repeat(np.arange(0, 33), 20), 0), data(
            np.repeat(np.arange(33, 41), 20), 10000
        )

    monkeypatch.setattr(module, "load_fold_frames", load_frames)
    config = Path("configs/ablations/without_credit_bureau_a.json")
    runner = BenchmarkRunner(
        feature_dir=tmp_path / "features",
        expected_feature_manifest_sha256=original.manifest_sha256,
        protocol_path=Path("configs/validation_protocol.json"),
        expected_protocol_sha256=original.protocol_sha256,
        config_path=config,
        expected_config_sha256=sha256_file(config),
        output_dir=tmp_path / "run",
        logs_dir=tmp_path / "logs",
        smoke=True,
        max_new_checkpoints=1,
    )
    summary = runner.run()
    assert summary["models"][0]["model"] == "lightgbm"
    assert summary["selected_features"] == 427
    state = (tmp_path / "run/benchmark_state.json").read_bytes()

    def no_fit(*args, **kwargs):
        raise AssertionError("completed fold was retrained")

    monkeypatch.setattr(module, "fit_lightgbm", no_fit)
    runner.run()
    assert (tmp_path / "run/benchmark_state.json").read_bytes() == state
    assert "benchmark_fold_resumed" in runner.logger.jsonl_path.read_text()
