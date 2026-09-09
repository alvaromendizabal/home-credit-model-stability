"""Independent numeric oracles, temporal boundaries, real fits and checkpoint recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from scipy.stats import skew

from home_credit.features.distributions import (
    STATISTICS,
    HistoryFeature,
    aggregate_history,
    inspect_source,
    scan_numeric_history,
)
from home_credit.modeling.checkpoints import sha256_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig, ModelConfig
from home_credit.modeling.data import FeatureBlockRef, FeatureRef, FeatureSnapshot
from home_credit.modeling.experiment_store import ExperimentStore
from home_credit.modeling.feature_research import run_fold
from home_credit.modeling.history_research import (
    build_history_snapshot,
    checkpointed_fold,
    validate_plan,
)
from home_credit.modeling.release_workflow import ReleaseLogger

ROOT = Path(__file__).resolve().parents[2]


def test_history_quantiles_pool_shards_and_match_numpy_scipy(tmp_path):
    raw = pl.DataFrame(
        {
            "case_id": [1] * 6 + [2] * 3 + [3] * 5 + [99],
            "amountA": [
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
                float("inf"),
                4.0,
                None,
                float("nan"),
                *([3.0] * 5),
                1e30,
            ],
            "num_group1": list(range(15)),
            "eventD": ["2020-01-01"] * 15,
        }
    )
    paths = (tmp_path / "a.parquet", tmp_path / "b.parquet")
    raw[:4].write_parquet(paths[0])
    raw[4:].write_parquet(paths[1])
    specs, audit = inspect_source(paths, "applprev", 1)
    assert len(specs) == 4 and all(s.column == "amountA" for s in specs)
    assert audit["date_fields"][0]["availability_time_verified"] is False
    cases = pl.DataFrame({"case_id": [1, 2, 3, 4], "WEEK_NUM": [0, 32, 72, 40]})
    history = scan_numeric_history(paths, ("amountA",))
    actual = aggregate_history(history, cases, specs)
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    expected = [
        np.median(values),
        np.quantile(values, 0.75) - np.quantile(values, 0.25),
        np.quantile(values, 0.9),
        skew(values, bias=False),
    ]
    np.testing.assert_allclose(actual.row(0)[1:], expected, rtol=1e-6)
    assert actual.row(1)[1:] == (4.0, None, None, None)
    assert actual.row(2)[1:] == (3.0, 0.0, 3.0, None)
    assert actual.row(3)[1:] == (None,) * 4
    shuffled = aggregate_history(history.reverse(), cases.reverse(), specs)
    assert actual.equals(shuffled)
    partitioned = pl.concat(
        [aggregate_history(history, part, specs) for part in cases.iter_slices(2)]
    )
    assert actual.equals(partitioned.sort("case_id"))


@pytest.mark.parametrize("fault", ["holdout", "duplicate", "missing_week", "date", "index"])
def test_history_boundaries_fail_closed(fault):
    cases = pl.DataFrame({"case_id": [1, 2], "WEEK_NUM": [0, 72]})
    column = "amountA"
    if fault == "holdout":
        cases = cases.with_columns(pl.lit(73).alias("WEEK_NUM"))
    elif fault == "duplicate":
        cases = cases.with_columns(pl.lit(1).alias("case_id"))
    elif fault == "missing_week":
        cases = cases.drop("WEEK_NUM")
    else:
        column = "eventD" if fault == "date" else "num_group1"
    specs = (HistoryFeature("bureau", 1, column, "median"),)
    with pytest.raises(ValueError):
        aggregate_history(pl.DataFrame({"case_id": [1], column: [3.0]}).lazy(), cases, specs)


@pytest.mark.parametrize(
    "change",
    [
        {"development_week_max": 91},
        {"new_model_fit_budget": 50},
        {"release_promotion_allowed": True},
        {"additional_features": 700},
        {"screen_weeks": {"train": [0, 40], "validation": [41, 48]}},
    ],
)
def test_history_plan_rejects_adaptive_expansion(change):
    plan = json.loads((ROOT / "configs/history_research.json").read_text())
    protocol = json.loads((ROOT / "configs/validation_protocol.json").read_text())
    validate_plan(plan, protocol)
    plan.update(change)
    with pytest.raises(ValueError):
        validate_plan(plan, protocol)


def test_completed_trial_survives_coordinator_interruption(tmp_path, monkeypatch):
    """A saved fit receipt prevents another fit even before the parent ledger advances."""
    config, _ = BenchmarkConfig.load(ROOT / "configs/model_benchmark.json")
    view = FeatureSnapshot(tmp_path, "view", "protocol", "recipe", "exec", "commit", (), (), ())
    store = ExperimentStore(Mock(), "bucket", "study", tmp_path, Mock())
    objects = {}

    def put(key, payload, **conditions):
        assert conditions == {"IfNoneMatch": "*"} and key not in objects
        objects[key] = payload

    store.put = put
    store.read = lambda key: (objects[key], "etag") if key in objects else None
    fit = Mock(return_value={"experiment": "history_shape", "fold": 1, "artifacts": {}})
    monkeypatch.setattr("home_credit.modeling.history_research.run_fold", fit)
    fold = {"fold": 1}
    first = checkpointed_fold(view, (), fold, "history_shape", config, {}, store, Mock())
    again = checkpointed_fold(view, (), fold, "history_shape", config, {}, store, Mock())
    assert first == again and fit.call_count == 1
    with pytest.raises(ValueError, match="specification"):
        checkpointed_fold(
            view, (), fold, "history_shape", replace(config, seed=42), {}, store, Mock()
        )


def test_history_snapshot_resume_and_actual_model_fit(tmp_path, monkeypatch):
    """Exercise shard download, aggregation, verified reuse, feature joins and native fit."""
    rng = np.random.default_rng(81)
    base = pl.DataFrame(
        {
            "case_id": np.arange(4000),
            "WEEK_NUM": np.repeat(np.arange(40), 100),
            "target": np.tile([0, 1], 2000),
            "original": rng.normal(size=4000),
        }
    )
    base_path = tmp_path / "base.parquet"
    base.write_parquet(base_path)
    ref = FeatureRef("original", "base_depth0", "base", 0, "float", False)
    snapshot = FeatureSnapshot(
        tmp_path,
        "a" * 64,
        "protocol",
        "recipe",
        "exec",
        "commit",
        (FeatureBlockRef("train", "base", 0, base_path, sha256_file(base_path), 4000, 1),),
        (),
        (ref,),
    )
    raw = pl.DataFrame(
        {"case_id": np.repeat(np.arange(4000), 6), "amountA": rng.lognormal(size=24000)}
    )
    raw_path = tmp_path / "train_applprev_1_0.parquet"
    raw.write_parquet(raw_path)
    record = {
        "file": raw_path.name,
        "s3_key": "raw/history",
        "bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
    }
    manifest = (json.dumps(record) + "\n").encode()
    objects = {"manifest": manifest, "raw/history": raw_path.read_bytes()}
    logger = ReleaseLogger("history-test", tmp_path / "logs")
    client = Mock()
    client.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(
        objects[key]
    )
    store = ExperimentStore(client, "bucket", "study", tmp_path / "study", logger)
    store.root.mkdir()
    store.commit = Mock()

    def publish(_store, path, relative):
        key = f"verified/{sha256_file(path)}"
        objects[key] = path.read_bytes()
        return {
            "key": key,
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    monkeypatch.setattr("home_credit.modeling.history_research.publish_verified", publish)
    monkeypatch.setattr("home_credit.modeling.feature_research.publish_verified", publish)
    plan = {
        "source_depths": [1, 2],
        "raw_manifest_sha256": sha256_bytes(manifest),
        "case_partition_rows": 1000,
    }
    protocol = {
        "data_lock": {"manifest_uri": "s3://bucket/manifest"},
        "outer_holdout": {"development_rows": 4000},
    }
    state = {
        "identity": {"source_commit": "commit", "plan_sha256": "p"},
        "stages": {},
        "revision": 0,
    }
    view, specs, completed = build_history_snapshot(snapshot, plan, protocol, store, Mock(), state)
    assert len(specs) == len(STATISTICS)
    assert view.manifest_sha256 != snapshot.manifest_sha256 and view.test_blocks == ()
    assert snapshot.train_blocks == (view.train_blocks[0],)
    assert len(completed["stages"]) == 6
    assert not state["stages"]
    view.train_blocks[-1].path.write_bytes(b"corrupt local checkpoint")
    monkeypatch.setattr(
        "home_credit.modeling.history_research.aggregate_history",
        lambda *a: pytest.fail("valid partitions were rebuilt"),
    )
    resumed, again, _ = build_history_snapshot(snapshot, plan, protocol, store, Mock(), completed)
    assert again == specs and resumed.manifest_sha256 == view.manifest_sha256
    assert sha256_file(view.train_blocks[-1].path) == view.train_blocks[-1].output_sha256
    config, _ = BenchmarkConfig.load(ROOT / "configs/model_benchmark.json")
    params = next(m.params for m in config.models if m.name == "lightgbm").copy()
    params.update(num_boost_round=12, early_stopping_rounds=4)
    config = replace(config, threads=1, models=(ModelConfig("lightgbm", True, params),))
    trial = run_fold(
        view,
        (ref, *(s.ref for s in specs)),
        (),
        {
            "fold": 1,
            "train_week_min": 0,
            "train_week_max": 31,
            "validation_week_min": 32,
            "validation_week_max": 39,
        },
        "history_shape",
        config,
        {},
        store,
        logger,
    )
    assert trial["features"] == 5 and trial["validation_rows"] == 800
    predictions = pl.read_parquet(store.root / trial["artifacts"]["predictions"]["path"])
    assert predictions["case_id"].n_unique() == 800
    assert set(predictions["WEEK_NUM"]) == set(range(32, 40))
    assert np.isfinite(predictions["prediction"].to_numpy()).all()
