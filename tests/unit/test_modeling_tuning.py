"""Test proposal replay, frozen inputs, temporal selection, and native fold recovery."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.tuning import (
    evaluate_trial,
    load_plan,
    propose,
    rank_records,
    search_space,
    trial_config,
)

ROOT = Path(__file__).resolve().parents[2]


def frame(weeks):
    values = np.repeat(weeks, 20)
    target = np.tile([0, 1], len(values) // 2)
    return pl.DataFrame(
        {
            "case_id": values * 1000 + np.tile(np.arange(20), len(weeks)),
            "WEEK_NUM": values,
            "target": target,
            "prediction": 0.2 + 0.6 * target,
        }
    )


def study_fixture():
    plan = load_plan(ROOT)
    base = json.loads((ROOT / "configs/ablations/control.json").read_text())
    folds = json.loads((ROOT / "configs/validation_protocol.json").read_text())[
        "inner_temporal_cv"
    ]["folds"]
    control = frame(np.arange(33, 73))
    baseline = {
        "slot": 0,
        "name": "control",
        "state": "complete",
        "params": {k: base["models"]["lightgbm"][k] for k in search_space(plan)},
        **evaluate_trial(control, control, folds, "baseline"),
    }
    baseline["metrics"]["experiment"] = "control"
    for row in baseline["folds"]:
        row["experiment"] = "control"
    return plan, base, baseline


def test_proposals_replay_exactly_after_json_round_trip_and_tpe_startup(tmp_path):
    plan, base, baseline = study_fixture()
    history = [baseline]
    for slot in range(1, 9):
        first = propose(plan, history, slot)
        second = propose(plan, json.loads(json.dumps(history)), slot)
        assert first == second
        assert first["sampler_seed"] == plan["seed"] + slot
        config = trial_config(base, first)
        path = tmp_path / "trial.json"
        path.write_bytes(canonical_json_bytes(config))
        loaded, _ = BenchmarkConfig.load(path)
        assert loaded.enabled_model_names == ("lightgbm",)
        assert loaded.threads == 6 and loaded.feature_selection["exclude_blocks"] == []
        assert loaded.model("lightgbm").params["num_boost_round"] == 2200
        assert loaded.model("lightgbm").params["learning_rate"] == 0.03
        history.append({**first, "state": "complete", "value": 0.5 + slot / 1000})


def test_sampler_rejects_sampling_past_pending_or_nonfinite_history():
    plan, _, baseline = study_fixture()
    with pytest.raises(ValueError, match="incomplete"):
        propose(plan, [{**baseline, "state": "proposed"}], 1)
    with pytest.raises(ValueError, match="nonfinite"):
        propose(plan, [{**baseline, "value": float("nan")}], 1)
    with pytest.raises(ValueError, match="contiguous"):
        propose(plan, [baseline], 2)


def test_selection_obeys_all_frozen_tie_breakers_and_retains_control():
    _, _, baseline = study_fixture()
    candidate = copy.deepcopy(baseline)
    candidate.update(slot=1, name="trial_001")
    assert rank_records([candidate, baseline])[0]["slot"] == 0
    for key, direction in [
        ("worst_fold_stability", 1),
        ("mean_weekly_gini", 1),
        ("mean_temporal_slope", 1),
        ("mean_residual_std", -1),
        ("mean_brier_score", -1),
    ]:
        alternative = copy.deepcopy(candidate)
        alternative["metrics"][key] += direction * 0.001
        assert rank_records([baseline, alternative])[0]["slot"] == 1
    candidate["metrics"]["mean_fold_stability"] -= 0.001
    candidate["metrics"]["oof_auc"] = 1
    assert rank_records([candidate, baseline])[0]["slot"] == 0


def test_plan_rejects_modified_source_before_training(tmp_path):
    plan, _, _ = study_fixture()
    for name in [*plan["input_sha256"], "configs/model_tuning.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    path = tmp_path / "src/home_credit/modeling/models.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        load_plan(tmp_path)


def test_native_study_resumes_same_trial_after_completed_fold(tmp_path, monkeypatch):
    """Real LightGBM fits survive a simulated process failure between model-folds."""
    from test_experiment_store import FakeS3

    import home_credit.modeling.runner as engine
    from home_credit.modeling.data import FeatureSnapshot
    from home_credit.modeling.experiment_store import ExperimentStore
    from home_credit.modeling.screening import feature_refs_from_payload
    from home_credit.observability.logging import RunLogger

    spec = importlib.util.spec_from_file_location(
        "tuning_cli", ROOT / "scripts/run_model_tuning.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    plan, base, _ = study_fixture()
    plan["new_trials"] = 2
    base["models"]["lightgbm"].update(num_boost_round=20, early_stopping_rounds=5)
    for name, value in [
        ("configs/ablations/control.json", base),
        ("configs/model_tuning.json", plan),
        ("reports/feature_ablation/comparison.json", {"rows": [{"mean_fold_stability": 1.0}]}),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    protocol = tmp_path / "configs/validation_protocol.json"
    protocol.write_bytes((ROOT / "configs/validation_protocol.json").read_bytes())
    source = json.loads((ROOT / "configs/benchmark_features.json").read_text())
    snapshot = FeatureSnapshot(
        root=tmp_path,
        manifest_sha256=plan["feature_manifest_sha256"],
        protocol_sha256=plan["protocol_sha256"],
        recipe_sha256="recipe",
        execution_sha256="execution",
        feature_git_commit="source",
        train_blocks=(),
        test_blocks=(),
        features=feature_refs_from_payload(source),
    )
    monkeypatch.setattr(engine.FeatureSnapshot, "load", lambda *a, **k: snapshot)
    monkeypatch.setattr(engine, "_git_commit", lambda: "native-test")

    def load_frames(_snapshot, selected_features, **kwargs):
        def data(start, end):
            df = frame(np.arange(start, end + 1)).drop("prediction")
            return df.with_columns(
                [
                    ((pl.col("target") + pl.col("case_id") % 3) / 3).alias(f.name)
                    for f in selected_features
                ]
            )

        return data(kwargs["train_week_min"], kwargs["train_week_max"]), data(
            kwargs["validation_week_min"], kwargs["validation_week_max"]
        )

    monkeypatch.setattr(engine, "load_fold_frames", load_frames)
    monkeypatch.setattr(cli, "baseline_predictions", lambda *a: frame(np.arange(33, 73)))
    monkeypatch.setattr(cli, "publish_report", lambda *a, **k: None)
    interrupted = True
    fit_calls = []
    native_fit = engine.fit_lightgbm

    def counted_fit(*args, **kwargs):
        fit_calls.append(str(kwargs["artifact_path"]))
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(engine, "fit_lightgbm", counted_fit)

    def run_process(command, output, *args):
        nonlocal interrupted
        config = Path(command[command.index("--config") + 1])
        runner = engine.BenchmarkRunner(
            feature_dir=tmp_path,
            expected_feature_manifest_sha256=plan["feature_manifest_sha256"],
            protocol_path=protocol,
            expected_protocol_sha256=plan["protocol_sha256"],
            config_path=config,
            expected_config_sha256=sha256_file(config),
            output_dir=output,
            logs_dir=output / "logs",
            smoke=False,
            max_new_checkpoints=1 if interrupted else None,
        )
        runner.run()
        if interrupted:
            interrupted = False
            raise RuntimeError("injected worker interruption")

    monkeypatch.setattr(cli, "run_process", run_process)
    work = tmp_path / "work"
    logger = RunLogger("native-tuning", tmp_path / "logs")
    store = ExperimentStore(FakeS3(), "bucket", "native-study", work, logger)
    monkeypatch.setattr(store, "publish", lambda *a, **k: "uploaded")
    lease = SimpleNamespace(check=lambda: None)
    args = (tmp_path, work, plan, "native-test", False, store, lease, logger)
    with pytest.raises(RuntimeError, match="injected worker"):
        cli.run_study(*args)
    pending = json.loads((work / "study.json").read_text())["trials"][-1]
    assert pending["state"] == "proposed" and len(fit_calls) == 1
    saved = sha256_file(work / "trial_001/models/lightgbm/fold_1.txt")
    cli.run_study(*args)
    result = json.loads((work / "study.json").read_text())
    assert result["complete"] and len(fit_calls) == 10
    assert result["trials"][1]["params"] == pending["params"]
    assert sha256_file(work / "trial_001/models/lightgbm/fold_1.txt") == saved
    before = (work / "study.json").read_bytes()
    cli.run_study(*args)
    assert len(fit_calls) == 10 and (work / "study.json").read_bytes() == before
    assert result["outer_holdout_touched"] is False


def test_progress_counts_the_whole_study_and_stops_child_on_lease_loss(tmp_path, monkeypatch):
    import signal
    import subprocess
    import time

    from home_credit.observability.logging import RunLogger

    spec = importlib.util.spec_from_file_location(
        "tuning_progress_cli", ROOT / "scripts/run_model_tuning.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "trial_004"
    output.mkdir()
    (output / "benchmark_state.json").write_text(json.dumps({"folds": {"one": {}, "two": {}}}))

    class Process:
        pid = 123

        def __init__(self):
            self.killed = False

        def wait(self, timeout):
            if self.killed:
                return -15
            raise subprocess.TimeoutExpired("trial", timeout)

        def poll(self):
            return -15 if self.killed else None

    process = Process()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: process)
    signals = []

    def kill(pid, sig):
        signals.append((pid, sig))
        process.killed = True

    monkeypatch.setattr(cli.os, "killpg", kill)
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("lease lost")

    logger = RunLogger("progress-test", tmp_path / "logs")
    with pytest.raises(RuntimeError, match="lease lost"):
        cli.run_process(
            ["trial"],
            output,
            4,
            8,
            5,
            time.monotonic(),
            SimpleNamespace(root=tmp_path),
            SimpleNamespace(check=check),
            logger,
        )
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed_new_fits"] == 17
    assert progress["total_new_fits"] == 40
    assert progress["fit_progress_percent"] == 42.5
    assert progress["timestamp"].endswith("Z") and progress["total_elapsed_seconds"] >= 0
    assert signals == [(123, signal.SIGTERM)]
    with pytest.raises(InterruptedError, match="resumable"):
        cli.handle_termination(signal.SIGTERM, None)
