"""Feature boundary tests with adversarial values and independent model replay."""

import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from home_credit.features.research import (
    Hypothesis,
    apply_peer_references,
    case_features,
    catalog,
    peer_features,
    prune_candidates,
)
from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.config import BenchmarkConfig, ModelConfig
from home_credit.modeling.data import FeatureBlockRef, FeatureRef, FeatureSnapshot
from home_credit.modeling.experiment_store import ExperimentStore
from home_credit.modeling.feature_interpretation import interpret_fold
from home_credit.modeling.feature_research import (
    add_hypotheses,
    diagnostics,
    run_fold,
    validate_plan,
)
from home_credit.modeling.release import fit_encoder, transform
from home_credit.modeling.release_workflow import ReleaseLogger

ROOT = Path(__file__).resolve().parents[2]


def hypothesis(name: str, operation: str, *sources: str) -> Hypothesis:
    family = "peer_statistics" if operation.startswith("peer_") or operation == "rank" else "test"
    return Hypothesis(name, family, operation, sources, "Test economic hypothesis")


def test_ratio_and_dispersion_are_finite_or_null() -> None:
    frame = pl.DataFrame(
        {"a": [4.0, 2.0, None, float("inf"), 1e300], "b": [2.0, 0.0, -1.0, 3.0, 1.0]}
    )
    specs = (
        hypothesis("ratio", "positive_ratio", "a", "b"),
        hypothesis("relative", "relative_std", "a", "b"),
    )
    actual = case_features(frame, specs)
    assert actual["ratio"].to_list() == [2.0, None, None, None, None]
    assert actual["relative"].to_list() == [2.0, None, None, None, None]


def test_category_interactions_have_no_separator_or_null_collision() -> None:
    frame = pl.DataFrame({"a": ["x|y", "x", None, "null"], "b": ["z", "y|z", "a", "a"]})
    actual = case_features(frame, (hypothesis("pair", "category_pair", "a", "b"),))
    assert actual["pair"].n_unique() == 4


def test_catalog_is_deterministic_and_covers_original_sources() -> None:
    scores = json.loads((ROOT / "reports/feature_ablation/feature_screen.json").read_text())[
        "result"
    ]["scores"]
    a, b = catalog(scores), catalog(list(reversed(scores)))
    assert a == b
    assert len(a) > 4000
    assert {h.family for h in a} == {
        "dispersion",
        "source_order",
        "recency",
        "amount_ratios",
        "missingness",
        "household",
        "category_interactions",
        "peer_statistics",
    }
    assert all(not {"target", "case_id", "WEEK_NUM", "MONTH"} & set(h.sources) for h in a)


def test_training_only_peer_references_ignore_validation_labels_and_distribution() -> None:
    train = pl.DataFrame(
        {"amount": np.arange(1.0, 101.0), "group": ["known"] * 100, "target": [0, 1] * 50}
    )
    valid = pl.DataFrame(
        {"amount": [20.0, 50.0, None], "group": ["known", "unknown", "known"], "target": [1, 0, 1]}
    )
    specs = (
        hypothesis("percentile", "rank", "amount"),
        hypothesis("median_ratio", "peer_median_ratio", "amount", "group"),
        hypothesis("position", "peer_iqr_position", "amount", "group"),
    )
    a, b, state = peer_features(train, valid, specs)
    changed = valid.with_columns(
        pl.lit(1e12).alias("amount"), (1 - pl.col("target")).alias("target")
    )
    c, _, changed_state = peer_features(train, changed, specs)
    assert a.equals(c)
    assert state == changed_state
    assert b["percentile"].to_list() == pytest.approx([0.2, 0.5, None], nan_ok=True)
    assert b["median_ratio"][0] == pytest.approx(20 / 50.5)
    assert b["median_ratio"][1] is None
    assert b["median_ratio"][2] is None
    assert state["position"]["reference"][0]["count"] == 100
    replayed = apply_peer_references(valid.drop("target"), specs, json.loads(json.dumps(state)))
    assert replayed.equals(b)


def test_exact_duplicates_and_near_constants_are_recorded() -> None:
    frame = pl.DataFrame(
        {
            "a": list(range(2000)),
            "b": list(range(2000)),
            "c": [0.0] * 2000,
            "d": [0.0] * 1999 + [1.0],
        }
    )
    specs = tuple(hypothesis(n, "missing", n) for n in frame.columns)
    retained, records = prune_candidates(frame, specs)
    assert [s.name for s in retained] == ["a"]
    assert [r["structural_rejection"] for r in records] == [
        None,
        "exact_duplicate",
        "constant",
        "near_constant",
    ]


def test_no_additional_features_preserves_frames() -> None:
    frame = pl.DataFrame({"case_id": [1, 2], "target": [0, 1]})
    a, b, state = add_hypotheses(frame, frame, ())
    assert a.equals(frame) and b.equals(frame) and not state


@pytest.mark.parametrize(
    "change",
    [{"development_week_max": 91}, {"new_model_fit_budget": 200}, {"experiments": ["engineered"]}],
)
def test_research_rejects_unplanned_scopes(change: dict[str, object]) -> None:
    plan = json.loads((ROOT / "configs/feature_research.json").read_text())
    protocol = json.loads((ROOT / "configs/validation_protocol.json").read_text())
    plan.update(change)
    with pytest.raises(ValueError):
        validate_plan(plan, protocol)


def test_saved_encoder_model_and_shap_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    specs = (
        Hypothesis("research__signal", "amount_ratios", "positive_ratio", ("a", "b"), "Exposure"),
    )
    raw = pl.DataFrame({"a": rng.normal(2, 1, 1200), "b": np.ones(1200)})
    frame = raw.hstack(case_features(raw, specs)).with_columns(
        pl.Series("target", (raw["a"].to_numpy() + rng.normal(0, 1, 1200) > 2).astype(np.int8)),
        pl.Series("WEEK_NUM", np.repeat(np.arange(12), 100)),
    )
    refs = tuple(s.ref for s in specs)
    encoder = json.loads(json.dumps(fit_encoder(frame[:800], refs)))
    x = transform(frame, refs, encoder)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 10},
        lgb.Dataset(x[:800], label=frame["target"][:800].to_numpy()),
        num_boost_round=10,
    )
    path = tmp_path / "model.txt"
    booster.save_model(path)
    restored = lgb.Booster(model_file=str(path))
    assert np.array_equal(booster.predict(x[800:]), restored.predict(x[800:]))
    result = diagnostics(
        restored,
        x[800:],
        frame[800:],
        x[:800],
        refs,
        {"permutation_rows": 400, "permutation_repeats": 2, "shap_rows": 50, "threads": 1},
        42,
    )
    assert result["shap_rows"] == 50
    assert result["shap_sampling"] == "uniform_without_replacement_from_full_validation_fold"
    assert {row["WEEK_NUM"] for row in result["shap_week_counts"]} == {8, 9, 10, 11}
    assert sum(row["len"] for row in result["shap_week_counts"]) == 50
    assert result["importance"][0]["mean_abs_shap"] > 0
    assert all(r["auc_decrease"] > 0 for r in result["permutation"])


@pytest.mark.parametrize("fault", ["missing", "rank_source", "unordered", "unsupported_group"])
def test_saved_peer_references_fail_closed(fault: str) -> None:
    train = pl.DataFrame({"amount": np.arange(100.0), "group": ["a"] * 100})
    specs = (
        hypothesis("rank", "rank", "amount"),
        hypothesis("median", "peer_median_ratio", "amount", "group"),
    )
    _, expected, state = peer_features(train, train, specs)
    assert apply_peer_references(train, specs, state).equals(expected)
    if fault == "missing":
        del state["rank"]
    elif fault == "rank_source":
        state["rank"]["source"] = "target"
    elif fault == "unordered":
        state["rank"]["values"].reverse()
    else:
        state["median"]["reference"][0]["count"] = 49
    with pytest.raises(ValueError):
        apply_peer_references(train, specs, state)


def test_fold_orchestration_writes_replayable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual feature join, learner, evaluation and checkpoint path."""
    rng = np.random.default_rng(8)
    frame = pl.DataFrame(
        {
            "case_id": np.arange(4000),
            "WEEK_NUM": np.repeat(np.arange(40), 100),
            "target": np.tile([0, 1], 2000),
            "a": rng.normal(2, 1, 4000),
            "b": np.ones(4000),
            "group": ["known"] * 4000,
        }
    )
    path = tmp_path / "base.parquet"
    frame.write_parquet(path)
    refs = tuple(
        FeatureRef(
            n, "base_depth0", "base", 0, "String" if n == "group" else "Float64", n == "group"
        )
        for n in ("a", "b", "group")
    )
    block = FeatureBlockRef("train", "base", 0, path, sha256_file(path), 4000, 3)
    snapshot = FeatureSnapshot(
        tmp_path, "manifest", "protocol", "recipe", "execution", "commit", (block,), (), refs
    )
    config, _ = BenchmarkConfig.load(ROOT / "configs/model_benchmark.json")
    params = next(m.params for m in config.models if m.name == "lightgbm").copy()
    params.update(num_boost_round=20, early_stopping_rounds=5)
    config = replace(config, models=(ModelConfig("lightgbm", True, params),), threads=1)
    logger = ReleaseLogger("research-unit", tmp_path / "logs")
    store = ExperimentStore(None, "test", "test", tmp_path / "study", logger)
    specs = (
        Hypothesis("research__amount", "amount_ratios", "positive_ratio", ("a", "b"), "Exposure"),
        Hypothesis("research__rank", "peer_statistics", "rank", ("a",), "Relative position"),
    )

    def publish(_store: ExperimentStore, path: Path, relative: str) -> dict[str, object]:
        return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}

    monkeypatch.setattr("home_credit.modeling.feature_research.publish_verified", publish)
    fold = {
        "fold": 1,
        "train_week_min": 0,
        "train_week_max": 31,
        "validation_week_min": 32,
        "validation_week_max": 39,
    }
    plan = {"permutation_rows": 800, "permutation_repeats": 1, "shap_rows": 64, "threads": 1}
    result = run_fold(
        snapshot,
        (refs[0], refs[2]),
        specs,
        fold,
        "engineered",
        config,
        plan,
        store,
        logger,
    )
    assert result["train_rows"] == 3200 and result["validation_rows"] == 800
    assert result["features"] == 4
    assert set(result["artifacts"]) == {
        "model",
        "predictions",
        "encoder",
        "peer_references",
        "features",
        "diagnostics",
    }
    saved = pl.read_parquet(store.root / result["artifacts"]["predictions"]["path"])
    assert set(saved["WEEK_NUM"]) == set(range(32, 40))
    assert np.isfinite(saved["prediction"].to_numpy()).all()

    def restore(_store: ExperimentStore, member: dict[str, object]) -> Path:
        path = store.root / str(member["path"])
        assert sha256_file(path) == member["sha256"]
        return path

    monkeypatch.setattr("home_credit.modeling.feature_interpretation.restore_member", restore)
    replay = interpret_fold(result, fold, snapshot, config, plan, store)
    assert replay["new_model_fits"] == 0 and replay["peer_references_refitted"] is False
    assert replay["prediction_maximum_absolute_error"] <= 1e-12
    assert replay["replayed_predictions"] == 800
    assert replay["native_model_sha256"] == result["artifacts"]["model"]["sha256"]
