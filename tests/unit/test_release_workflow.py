"""Fault injection proves release stages do not outrun verified durable state."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import home_credit.modeling.release_workflow as workflow
from home_credit.modeling.checkpoints import sha256_bytes
from home_credit.observability.runtime import StageTimer


def test_stage_state_only_advances_after_durable_commit():
    original = {"revision": 0, "stages": {}}
    store = Mock()
    lease = Mock()
    store.commit.side_effect = OSError("upload failed")
    with pytest.raises(OSError):
        workflow.commit_stage(store, lease, original, "model", {"sha256": "a" * 64})
    assert original == {"revision": 0, "stages": {}}
    store.commit.side_effect = None
    result = workflow.commit_stage(store, lease, original, "model", {"sha256": "a" * 64})
    assert result["revision"] == 1 and "model" in result["stages"]
    assert original["revision"] == 0
    lease.check.assert_called()


def test_full_readback_is_required_after_upload(tmp_path):
    path = tmp_path / "native.txt"
    path.write_bytes(b"native model")
    store = Mock()
    store.publish.return_value = "s3-key"
    store.read.return_value = (path.read_bytes(), "etag")
    member = workflow.publish_verified(store, path, "development/model.txt")
    assert member["bytes"] == len(path.read_bytes())
    assert member["sha256"] == sha256_bytes(path.read_bytes())
    store.read.return_value = (b"corrupted", "etag")
    with pytest.raises(ValueError, match="read-back"):
        workflow.publish_verified(store, path, "development/model.txt")


def test_complete_models_resume_without_materializing_or_fitting(tmp_path, monkeypatch):
    model = {"fit_weeks": [0, 72], "fit_rows": 7, "model": {"path": "model.txt"}, "encoder": {"path": "encoder.json"}}
    original = {"stages": {"development/a": copy.deepcopy(model), "development/b": copy.deepcopy(model)}}
    store = SimpleNamespace(logger=Mock())
    monkeypatch.setattr(workflow, "restore_member", lambda *args: tmp_path / "verified")
    monkeypatch.setattr(workflow, "load_feature_frame", lambda *a, **k: pytest.fail("valid work was rematerialized"))
    monkeypatch.setattr(workflow, "fit_frozen_model", lambda *a, **k: pytest.fail("valid fit was repeated"))
    plan = {"components": {"a": {"num_boost_round": 5}, "b": {"num_boost_round": 5}}, "holdout_fit_weeks": [0, 72], "seed": 9, "threads": 1}
    parameters = {"a": {"num_leaves": 7}, "b": {"num_leaves": 7}}
    for name in plan["components"]:
        original["stages"][f"development/{name}"]["fit_spec_sha256"] = sha256_bytes(workflow.canonical_json_bytes({"phase": "development", "weeks": [0, 72], "rows": 7, "features": [], "parameters": parameters[name], "rounds": 5, "seed": 9, "threads": 1}))
    actual = workflow.fit_phase("development", [0, 72], 7, None, (), plan, parameters, original, store, Mock())
    assert actual is original
    with pytest.raises(ValueError, match="specification"):
        workflow.fit_phase("development", [0, 72], 8, None, (), plan, parameters, original, store, Mock())


def test_holdout_cannot_change_frozen_models_after_intent_exists():
    state = {"stages": {"development/a": {"model": {"sha256": "a" * 64}, "encoder": {"sha256": "b" * 64}}}}
    plan = {"components": {"a": {}}, "selection": {"sha256": "c" * 64, "weights": {"a": 1.0}}, "feature_manifest_sha256": "d" * 64, "holdout_fit_weeks": [0, 72], "holdout_evaluation_weeks": [73, 91], "protocol_sha256": "e" * 64}
    store = Mock()
    store.read.return_value = (b'{"models": "different"}', "etag")
    with pytest.raises(ValueError, match="already bound"):
        workflow.evaluate_holdout(None, (), plan, {}, state, store, Mock())
    store.put.assert_not_called()


def test_completed_holdout_reuses_verified_prediction_receipt(monkeypatch):
    report = {"sha256": "f" * 64}
    state = {"stages": {"holdout": {"report": report, "predictions": {}}}}
    restored = Mock(return_value=Path("verified"))
    monkeypatch.setattr(workflow, "restore_member", restored)
    monkeypatch.setattr(workflow, "load_feature_frame", lambda *a, **k: pytest.fail("holdout reread"))
    assert workflow.evaluate_holdout(None, (), {}, {}, state, Mock(), Mock()) is state
    assert restored.call_count == 2


def test_logging_includes_utc_nested_stage_total_and_heartbeat(tmp_path):
    logger = workflow.ReleaseLogger("test-release", tmp_path)
    with StageTimer(logger, "outer", heartbeat_seconds=0.01):
        with StageTimer(logger, "inner"):
            logger.event("progress", completed=1, total=2)
        time.sleep(0.025)
    rows = [json.loads(line) for line in logger.jsonl_path.read_text().splitlines()]
    assert any(row["event"] == "heartbeat" for row in rows)
    for row in rows:
        assert row["timestamp"].endswith("Z")
        assert row["stage_elapsed_seconds"] >= 0
        assert row["total_elapsed_seconds"] >= 0
    assert next(row for row in rows if row["event"] == "progress")["stage"] == "inner"
    assert [row for row in rows if row["event"] == "stage_completed"][-1]["stage"] == "outer"
