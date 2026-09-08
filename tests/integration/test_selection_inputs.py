"""Exercise real Parquet input recovery without AWS or native model training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="Parquet integration requires the locked environment")

from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_bytes
from home_credit.modeling.selection_workflow import load_inputs
from home_credit.observability.logging import RunLogger


class Downloads:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def download(self, key, path, digest):
        payload = self.objects[key]
        assert sha256_bytes(payload) == digest
        self.calls.append(key)
        atomic_write(path, payload)


def inputs_fixture(tmp_path, fault=None):
    root = Path(__file__).resolve().parents[2]
    protocol = (root / "configs/validation_protocol.json").read_bytes()
    atomic_write(tmp_path / "configs/validation_protocol.json", protocol)
    plan = json.loads((root / "configs/model_selection.json").read_text())
    plan["expected_rows"] = 800
    objects = {}
    manifests = {}
    for name, policy in plan["manifests"].items():
        manifests[name] = {
            "schema_version": 1,
            "smoke": False,
            "feature_manifest_sha256": plan["feature_manifest_sha256"],
            "validation_protocol_sha256": plan["protocol_sha256"],
            "completed_model_folds": policy["completed_model_folds"],
            "files": [],
        }
    weeks = np.repeat(np.arange(33, 73), 20)
    target = np.tile([0] * 15 + [1] * 5, 40)
    for index, source in enumerate(plan["sources"]):
        frame = pd.DataFrame(
            {
                "case_id": np.arange(800),
                "target": target,
                "WEEK_NUM": weeks,
                "fold": (weeks - 33) // 8 + 1,
                "prediction": 0.1 + target * 0.7 + index * 0.001,
            }
        )
        if index == 2:
            if fault == "missing_case":
                frame = frame.iloc[:-1]
            elif fault == "target":
                frame.loc[0, "target"] = 1
            elif fault == "holdout":
                frame.loc[0, "WEEK_NUM"] = 73
        payload = frame.to_parquet(index=False)
        key = f"fixture/{source['name']}.parquet"
        digest = sha256_bytes(payload)
        objects[key] = payload
        source["sha256"] = digest
        manifests[source["manifest"]]["files"].append(
            {
                "path": source["path"],
                "object_key": key,
                "sha256": digest,
                "bytes": len(payload),
            }
        )
    if fault == "feature_identity":
        manifests["tuned"]["feature_manifest_sha256"] = "changed"
    elif fault == "incomplete_manifest":
        manifests["tuned"]["completed_model_folds"] = 4
    elif fault == "source_digest":
        plan["sources"][0]["sha256"] = "f" * 64
    for name, manifest in manifests.items():
        payload = canonical_json_bytes(manifest)
        key = f"fixture/{name}.json"
        objects[key] = payload
        plan["manifests"][name]["key"] = key
        plan["manifests"][name]["sha256"] = sha256_bytes(payload)
    metrics = {
        "mean_fold_stability": 0.5,
        "worst_fold_stability": 0.4,
        "mean_weekly_gini": 0.6,
        "mean_temporal_slope": 0.0,
        "mean_residual_std": 0.01,
        "mean_brier_score": 0.1,
    }
    trials = []
    for index in range(9):
        name = "control" if index == 0 else f"trial_{index:03d}"
        row = {"name": name, "state": "complete", "metrics": copy.deepcopy(metrics)}
        row["metrics"]["mean_fold_stability"] += 0.1 * (index == 6)
        trials.append(row)
    study = {
        "identity": {"smoke": fault == "smoke"},
        "complete": True,
        "outer_holdout_touched": False,
        "trials": trials,
        "selected_trial": "trial_001" if fault == "winner" else "trial_006",
    }
    payload = canonical_json_bytes(study)
    objects["fixture/study.json"] = payload
    plan["study"]["key"] = "fixture/study.json"
    plan["study"]["sha256"] = sha256_bytes(payload)
    return plan, Downloads(objects)


def test_frozen_parquet_inputs_round_trip_and_align(tmp_path):
    plan, store = inputs_fixture(tmp_path)
    data, study, protocol = load_inputs(
        tmp_path, plan, store, RunLogger("inputs", tmp_path / "logs")
    )
    assert len(data.target) == 800
    assert len(data.predictions) == 4
    assert data.week.max() == 72
    assert len(store.calls) == 7
    assert study["selected_trial"] == "trial_006"
    assert protocol["outer_holdout"]["locked"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "feature_identity",
        "incomplete_manifest",
        "source_digest",
        "smoke",
        "winner",
        "missing_case",
        "target",
        "holdout",
    ],
)
def test_invalid_frozen_inputs_never_reach_candidate_scoring(tmp_path, fault):
    plan, store = inputs_fixture(tmp_path, fault)
    with pytest.raises(ValueError):
        load_inputs(tmp_path, plan, store, RunLogger("bad-inputs", tmp_path / "logs"))
