"""Raw feature parity, resumable native scoring and owner-only synthetic CSV tests."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import home_credit.modeling.inference as inference
from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.data import FeatureRef
from home_credit.modeling.release import fit_encoder, save_json, transform
from home_credit.observability.logging import RunLogger


@pytest.fixture
def raw_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pl.DataFrame(
        {
            "case_id": [3, 1, 2],
            "date_decision": [date(2020, 1, 31)] * 3,
            "WEEK_NUM": [100] * 3,
            "MONTH": [1] * 3,
        }
    ).write_parquet(raw / "test_base.parquet")
    pl.DataFrame(
        {
            "case_id": [3, 1],
            "amountA": [30.0, 10.0],
            "statusM": ["b", "a"],
            "openedD": [date(2020, 1, 30), date(2020, 1, 1)],
        }
    ).write_parquet(raw / "test_static_0_0.parquet")
    pl.DataFrame(
        {"case_id": [2], "amountA": [20.0], "statusM": [None], "openedD": [date(2020, 1, 21)]}
    ).write_parquet(raw / "test_static_0_1.parquet")
    pl.DataFrame(
        {"case_id": [1, 1, 3], "num_group1": [0, 1, 0], "balanceA": [2.0, 4.0, 7.0]}
    ).write_parquet(raw / "test_other_1.parquet")
    features = (
        FeatureRef("static__d0__amountA", "static_depth0", "static", 0, "double", False),
        FeatureRef("static__d0__statusM", "static_depth0", "static", 0, "string", True),
        FeatureRef("static__d0__openedD", "static_depth0", "static", 0, "float", False),
        FeatureRef("other__d1__balanceA__mean", "other_depth1", "other", 1, "double", False),
    )
    recipe = Path(__file__).resolve().parents[2] / "configs/feature_recipe.json"
    schema = {
        "schema_version": 1,
        "recipe_sha256": sha256_file(recipe),
        "blocks": {
            "static_depth0": {
                "semantic": {
                    "numeric": ["amountA"],
                    "categorical": ["statusM"],
                    "date": ["openedD"],
                    "unsupported": [],
                },
                "allowed_absent_columns": [],
            },
            "other_depth1": {
                "semantic": {
                    "numeric": ["balanceA"],
                    "categorical": [],
                    "date": [],
                    "unsupported": [],
                },
                "allowed_absent_columns": [],
            },
        },
    }
    return raw, features, schema, recipe


def frame_from_fixture(fixture, **kwargs):
    raw, features, schema, recipe = fixture
    return inference.raw_test_frame(
        raw, inference.local_test_records(raw), features, schema, recipe, **kwargs
    )


def test_raw_multishard_dates_and_missing_history_match_batches(raw_fixture):
    frame = frame_from_fixture(raw_fixture)
    assert frame["case_id"].to_list() == [1, 2, 3]
    assert frame["static__d0__amountA"].to_list() == [10.0, 20.0, 30.0]
    assert frame["static__d0__openedD"].to_list() == [-30.0, -10.0, -1.0]
    assert frame["other__d1__balanceA__mean"].to_list() == [3.0, None, 7.0]
    small = pl.concat(
        [
            frame_from_fixture(raw_fixture, case_ids=[1, 2]),
            frame_from_fixture(raw_fixture, case_ids=[3]),
        ]
    )
    assert_frame_equal(frame, small)


def test_text_decision_dates_preserve_raw_features(raw_fixture, capfd):
    expected = frame_from_fixture(raw_fixture)
    raw, _, _, _ = raw_fixture
    path = raw / "test_base.parquet"
    pl.read_parquet(path).with_columns(pl.col("date_decision").cast(pl.String)).write_parquet(path)
    assert_frame_equal(frame_from_fixture(raw_fixture), expected)
    assert "DeprecationWarning" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_table",
        "missing_column",
        "target",
        "duplicate_static",
        "duplicate_base",
        "null_id",
        "fractional_id",
    ],
)
def test_raw_contract_failures(raw_fixture, mutation):
    raw, _, _, _ = raw_fixture
    if mutation == "missing_table":
        (raw / "test_other_1.parquet").unlink()
    elif mutation == "missing_column":
        for path in raw.glob("test_static*"):
            pl.read_parquet(path).drop("amountA").write_parquet(path)
    elif mutation == "target":
        path = raw / "test_other_1.parquet"
        pl.read_parquet(path).with_columns(pl.lit(1).alias("target")).write_parquet(path)
    elif mutation == "duplicate_static":
        path = raw / "test_static_0_0.parquet"
        table = pl.read_parquet(path)
        pl.concat([table, table.head(1)]).write_parquet(path)
    else:
        path = raw / "test_base.parquet"
        table = pl.read_parquet(path)
        if mutation == "duplicate_base":
            table = pl.concat([table, table.head(1)])
        elif mutation == "null_id":
            table = table.with_columns(pl.Series("case_id", [None, 1, 2], dtype=pl.Int64))
        else:
            table = table.with_columns(pl.Series("case_id", [3.5, 1.0, 2.0]))
        table.write_parquet(path)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        frame_from_fixture(raw_fixture)


def test_empty_history_is_not_missing_table(raw_fixture):
    raw, _, _, _ = raw_fixture
    path = raw / "test_other_1.parquet"
    pl.read_parquet(path).head(0).write_parquet(path)
    result = frame_from_fixture(raw_fixture)
    assert result.height == 3
    assert result["other__d1__balanceA__mean"].null_count() == 3


def test_changed_recipe_and_bad_case_batch_rejected(raw_fixture):
    _, _, schema, _ = raw_fixture
    for cases in ([1, 1], [4], []):
        with pytest.raises(ValueError):
            frame_from_fixture(raw_fixture, case_ids=cases)
    schema["recipe_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recipe"):
        frame_from_fixture(raw_fixture)


@pytest.fixture
def native_bundle(raw_fixture, tmp_path):
    raw, features, schema, recipe = raw_fixture
    frame = frame_from_fixture(raw_fixture)
    train = pl.concat([frame] * 30)
    encoder = fit_encoder(train, features)
    matrix = transform(train, features, encoder)
    target = np.tile([0, 1, 1], 30)
    model = lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "num_threads": 1,
            "min_data_in_leaf": 2,
            "num_leaves": 3,
            "deterministic": True,
            "force_col_wise": True,
        },
        lgb.Dataset(matrix, label=target, feature_name=[f.name for f in features]),
        num_boost_round=5,
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "feature_recipe.json").write_bytes(recipe.read_bytes())
    save_json(bundle / "features.json", {"features": [asdict(f) for f in features]})
    save_json(bundle / "raw_schema.json", schema)
    save_json(bundle / "encoder.json", encoder)
    model.save_model(str(bundle / "lightgbm.txt"))
    manifest = {
        "schema_version": 1,
        "phase": "all_labels",
        "fit_weeks": [0, 91],
        "calibration": "none",
        "feature_count": len(features),
        "inference_code_sha256": inference.inference_code_identity(),
        "models": {
            "model": {"path": "lightgbm.txt", "sha256": sha256_file(bundle / "lightgbm.txt")}
        },
        "weights": {"model": 1.0},
        "files": [
            {"path": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(bundle.iterdir())
        ],
    }
    save_json(bundle / "bundle.json", manifest)
    return bundle, raw, sha256_file(bundle / "bundle.json")


def test_native_raw_inference_reuses_completed_batches(native_bundle, tmp_path, monkeypatch):
    bundle, raw, digest = native_bundle
    logger = RunLogger("scoring-test", tmp_path / "logs")
    result = inference.predict_raw(
        bundle, raw, tmp_path / "cache", expected_bundle_sha256=digest, logger=logger, batch_rows=2
    )
    assert result["case_id"].tolist() == [1, 2, 3]
    assert result["score"].between(0, 1).all()
    assert not list(tmp_path.rglob("*.csv"))
    monkeypatch.setattr(
        inference, "raw_test_frame", lambda *a, **k: pytest.fail("a completed batch was rescored")
    )
    restored = inference.predict_raw(
        bundle, raw, tmp_path / "cache", expected_bundle_sha256=digest, logger=logger, batch_rows=2
    )
    pd.testing.assert_frame_equal(result, restored)
    paths = list((tmp_path / "cache").rglob("part_000001.parquet"))
    assert len(paths) == 1
    paths[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        inference.predict_raw(
            bundle,
            raw,
            tmp_path / "cache",
            expected_bundle_sha256=digest,
            logger=logger,
            batch_rows=2,
        )


@pytest.mark.parametrize(
    "mutation", ["model", "implementation", "phase", "unlisted_encoder", "duplicate_member"]
)
def test_bundle_corruption_and_missing_members_rejected(native_bundle, mutation):
    directory, _, digest = native_bundle
    path = directory / "bundle.json"
    manifest = json.loads(path.read_text())
    if mutation == "model":
        (directory / "lightgbm.txt").write_text("corrupt")
    elif mutation == "implementation":
        manifest["inference_code_sha256"] = "0" * 64
    elif mutation == "phase":
        manifest["phase"] = "development"
    elif mutation == "unlisted_encoder":
        manifest["files"] = [r for r in manifest["files"] if r["path"] != "encoder.json"]
    else:
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    if mutation != "model":
        save_json(path, manifest)
        digest = sha256_file(path)
    with pytest.raises(ValueError):
        inference.load_bundle(directory, expected_sha256=digest)


def test_owner_export_aligns_and_roundtrips_synthetic_only(tmp_path):
    sample = pd.DataFrame({"case_id": [3, 1, 2], "score": [0.0] * 3})
    predictions = pd.DataFrame({"case_id": [1, 2, 3], "score": [0.1, 0.9, 0.12345678901234567]})
    path = tmp_path / "submission.csv"
    lineage = {"bundle_sha256": "a" * 64, "input_sha256": "b" * 64, "synthetic": True}
    with pytest.raises(ValueError, match="owner"):
        inference.export_submission(sample, predictions, path, lineage=lineage)
    assert not path.exists()
    inference.export_submission(sample, predictions, path, lineage=lineage, owner_confirmed=True)
    exported = pd.read_csv(path, float_precision="round_trip")
    assert exported["case_id"].tolist() == [3, 1, 2]
    np.testing.assert_allclose(exported["score"], [0.12345678901234567, 0.1, 0.9], atol=1e-15)
    evidence = json.loads(path.with_suffix(".json").read_text())
    assert evidence["sha256"] == sha256_file(path)
    assert evidence["kaggle_submitted"] is False
    before = path.read_bytes()
    inference.export_submission(sample, predictions, path, lineage=lineage, owner_confirmed=True)
    assert path.read_bytes() == before
    predictions.loc[0, "score"] = 0.3
    with pytest.raises(ValueError, match="overwrite"):
        inference.export_submission(
            sample, predictions, path, lineage=lineage, owner_confirmed=True
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "extra",
        "replaced",
        "nan",
        "infinity",
        "negative",
        "too_large",
        "float_id",
        "columns",
    ],
)
def test_submission_contract_rejects_invalid_predictions(mutation):
    sample = pd.DataFrame({"case_id": [1, 2], "score": [0.0, 0.0]})
    predictions = pd.DataFrame({"case_id": [1, 2], "score": [0.2, 0.3]})
    if mutation == "duplicate":
        predictions["case_id"] = [1, 1]
    elif mutation == "missing":
        predictions = predictions.head(1)
    elif mutation == "extra":
        predictions = pd.concat([predictions, pd.DataFrame({"case_id": [3], "score": [0.1]})])
    elif mutation == "replaced":
        predictions["case_id"] = [1, 3]
    elif mutation == "float_id":
        predictions["case_id"] = predictions["case_id"].astype(float)
    elif mutation == "columns":
        predictions = predictions.rename(columns={"score": "prediction"})
    else:
        predictions.loc[0, "score"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "negative": -0.1,
            "too_large": 1.1,
        }[mutation]
    with pytest.raises(ValueError):
        inference.submission_frame(sample, predictions)
