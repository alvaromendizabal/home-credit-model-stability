"""Bound shard ownership and preserve the original feature-recipe selection."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import FeatureRef

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "feature_shards", ROOT / "scripts/run_feature_research_shards.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def policy():
    return json.loads((ROOT / "configs/feature_research_shards.json").read_text())


def test_late_folds_have_one_owner_and_resumable_json_identity():
    plan = policy()
    tasks = set()
    for shard in plan["shards"]:
        pairs = MODULE.assignment(plan, shard)
        assert not pairs & tasks
        tasks |= pairs
        identity = MODULE.worker_identity({"training_source_commit": "fixed"}, shard, pairs)
        assert json.loads(json.dumps(identity)) == identity
    assert len(tasks) == 8
    assert {f for _, f in tasks} == {4, 5}


@pytest.mark.parametrize("fault", ["overlap", "holdout", "budget", "missing"])
def test_invalid_worker_grid_is_rejected(fault):
    plan = copy.deepcopy(policy())
    if fault == "overlap":
        plan["shards"]["wider"]["experiments"].append("engineered")
    elif fault == "holdout":
        plan["shards"]["wider"]["folds"] = [4, 6]
    elif fault == "budget":
        plan["maximum_complete_fits"] = 21
    else:
        del plan["shards"]["engineered"]
    with pytest.raises(ValueError):
        MODULE.assignment(plan, "wider")


def test_shard_feature_plan_retains_originals_and_removes_only_named_families():
    source = json.loads((ROOT / "reports/feature_ablation/feature_screen.json").read_text())[
        "result"
    ]
    keys = ["name", "block", "family", "depth", "dtype", "categorical"]
    snapshot = SimpleNamespace(
        features=tuple(FeatureRef(**{k: row[k] for k in keys}) for row in source["scores"])
    )
    screen = {
        "selected": [
            {
                "name": f"research__{family}",
                "family": family,
                "operation": "missing",
                "sources": [source["scores"][0]["name"]],
                "rationale": "Unit fixture",
            }
            for family in ("amount_ratios", "peer_statistics", "missingness")
        ]
    }
    config, _ = BenchmarkConfig.load(ROOT / "configs/model_benchmark.json")
    plan = json.loads((ROOT / "configs/feature_research.json").read_text())
    variants = MODULE.conditions(ROOT, snapshot, screen, config, plan)
    original, additions = variants["engineered"]
    assert len(original) == 700 and len(additions) == 3
    assert len(variants["wider_original"][0]) == 1400
    assert variants["wider_original"][0][:700] == original
    for name, excluded in (
        ("without_amount_ratios", "amount_ratios"),
        ("without_peer_statistics", "peer_statistics"),
    ):
        retained, hypotheses = variants[name]
        assert retained == original
        assert {h.family for h in hypotheses} == {
            "amount_ratios",
            "peer_statistics",
            "missingness",
        } - {excluded}


@pytest.mark.parametrize(
    "status,advanced", [("InProgress", False), ("Failed", False), ("InProgress", True)]
)
def test_guard_stops_only_the_named_source_job_after_durable_early_folds(
    tmp_path, monkeypatch, status, advanced
):
    plan = json.loads((ROOT / "configs/feature_research.json").read_text())
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/feature_research.json").write_text(json.dumps(plan))
    records = [
        {"experiment": name, "fold": fold} for name in plan["experiments"] for fold in (1, 2, 3)
    ]
    if advanced:
        records.append({"experiment": "wider_original", "fold": 4})
    managed = Mock()
    managed.describe_processing_job.return_value = {"ProcessingJobStatus": status}
    monkeypatch.setattr(MODULE, "source_identity", lambda *args: {})
    monkeypatch.setattr(MODULE, "parent_state", lambda *args: (None, {"trials": records}))
    monkeypatch.setattr(
        MODULE.boto3,
        "client",
        lambda service, **kwargs: managed if service == "sagemaker" else Mock(),
    )
    if advanced:
        with pytest.raises(ValueError, match="advanced"):
            MODULE.guard_original(ROOT, tmp_path, "test-bucket")
        managed.stop_processing_job.assert_not_called()
    else:
        MODULE.guard_original(ROOT, tmp_path, "test-bucket")
        if status == "InProgress":
            managed.stop_processing_job.assert_called_once_with(
                ProcessingJobName=policy()["original_job"]
            )
        else:
            managed.stop_processing_job.assert_not_called()
