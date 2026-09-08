"""Contracts for the bounded, development-only OOF ensemble stage."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from home_credit.modeling.selection import (
    align_predictions,
    blend,
    diagnostics,
    evaluate_candidate,
    fixed_candidates,
    prequential_choices,
    rank_records,
    validate_record,
    validate_windows,
)

NAMES = ["tuned_lightgbm", "lightgbm", "xgboost", "catboost"]
LEADER = NAMES[0]


def fixture_data():
    windows = [
        {
            "fold": f,
            "train_week_min": 0,
            "train_week_max": 30 + 2 * f,
            "validation_week_min": 31 + 2 * f,
            "validation_week_max": 32 + 2 * f,
        }
        for f in range(1, 6)
    ]
    week = np.repeat(np.arange(33, 43), 20)
    target = np.tile(np.array([0] * 15 + [1] * 5), 10)
    rng = np.random.default_rng(41)
    frames = {}
    for name in NAMES:
        frames[name] = pd.DataFrame(
            {
                "case_id": np.arange(200),
                "target": target,
                "WEEK_NUM": week,
                "fold": (week - 33) // 2 + 1,
                "prediction": np.clip(0.1 + 0.4 * target + rng.normal(0, 0.18, 200), 0, 1),
            }
        )
    return frames, windows


def test_fixed_grid_has_exact_budget_and_weights():
    candidates = fixed_candidates(NAMES, LEADER)
    assert len(candidates) == 15
    assert candidates[LEADER] == {LEADER: 1.0}
    for weights in candidates.values():
        assert sum(weights.values()) == pytest.approx(1)
        assert all(0 < value <= 1 for value in weights.values())


@pytest.mark.parametrize("names", [NAMES[:3], [LEADER] * 4, [*NAMES[:3], "bad/name"]])
def test_invalid_candidate_names(names):
    with pytest.raises(ValueError):
        fixed_candidates(names, LEADER)


def test_alignment_sorts_without_joining_or_dropping_cases():
    frames, windows = fixture_data()
    expected = frames[LEADER]["prediction"].to_numpy()
    frames[LEADER] = frames[LEADER].sample(frac=1, random_state=42)
    aligned = align_predictions(frames, windows, expected_rows=200)
    np.testing.assert_array_equal(aligned.case_id, np.arange(200))
    np.testing.assert_array_equal(aligned.predictions[LEADER], expected)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "target",
        "week",
        "fold",
        "nan",
        "inf",
        "range",
        "float_id",
        "single_class",
        "negative_id",
        "column",
    ],
)
def test_bad_predictions_are_rejected(fault):
    frames, windows = fixture_data()
    frame = frames["xgboost"]
    if fault == "duplicate":
        frame.loc[1, "case_id"] = 0
    elif fault == "missing":
        frames["xgboost"] = frame.iloc[:-1]
    elif fault == "target":
        frame.loc[0, "target"] = 1
    elif fault == "week":
        frame.loc[0, "WEEK_NUM"] = 73
    elif fault == "fold":
        frame.loc[0, "fold"] = 2
    elif fault in {"nan", "inf", "range"}:
        frame.loc[0, "prediction"] = {"nan": np.nan, "inf": np.inf, "range": -0.1}[fault]
    elif fault == "float_id":
        frame["case_id"] = frame["case_id"].astype(float)
    elif fault == "single_class":
        frame.loc[frame["WEEK_NUM"] == 33, "target"] = 0
    elif fault == "negative_id":
        frame.loc[0, "case_id"] = -1
    else:
        frames["xgboost"] = frame.drop(columns="target")
    with pytest.raises(ValueError):
        align_predictions(frames, windows, expected_rows=200)


@pytest.mark.parametrize(
    "key,value",
    [("validation_week_max", 73), ("train_week_max", 34), ("fold", 9), ("validation_week_min", 1)],
)
def test_invalid_windows_rejected(key, value):
    _, windows = fixture_data()
    windows[0][key] = value
    with pytest.raises(ValueError):
        validate_windows(windows)


@pytest.mark.parametrize(
    "weights", [{LEADER: 0.8}, {LEADER: -1}, {LEADER: np.nan}, {"unknown": 1}, {}]
)
def test_invalid_weights_rejected(weights):
    frames, windows = fixture_data()
    aligned = align_predictions(frames, windows, expected_rows=200)
    with pytest.raises(ValueError):
        blend(aligned, weights)


def test_metrics_recomputed_and_cache_contract_checked():
    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    weights = {LEADER: 0.75, "xgboost": 0.25}
    record = evaluate_candidate(data, "candidate", weights)
    validate_record(record, "candidate", weights, 5)
    assert record["metrics"]["mean_fold_stability"] == pytest.approx(
        np.mean([row["stability_score"] for row in record["folds"]])
    )
    assert record["metrics"]["oof_brier_score"] == pytest.approx(
        np.mean((blend(data, weights) - data.target) ** 2)
    )
    damaged = copy.deepcopy(record)
    damaged["metrics"]["mean_fold_stability"] += 0.1
    with pytest.raises(ValueError, match="cached summary"):
        validate_record(damaged, "candidate", weights, 5)


def test_identical_metrics_retain_single_model_incumbent():
    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    first = evaluate_candidate(data, LEADER, {LEADER: 1})
    duplicate = copy.deepcopy(first)
    duplicate["name"] = "aaa_blend"
    assert rank_records([duplicate, first], LEADER)[0]["name"] == LEADER


def test_future_folds_cannot_change_earlier_weight_choice():
    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    records = [evaluate_candidate(data, name, {name: 1}) for name in NAMES]
    before = prequential_choices(records, LEADER)
    damaged = copy.deepcopy(records)
    for record in damaged:
        record["folds"][4]["stability_score"] = 1000 if record["name"] == "catboost" else -1000
    after = prequential_choices(damaged, LEADER)
    assert [r["candidate"] for r in before] == [r["candidate"] for r in after]
    assert all(max(r["selected_using_folds"]) < r["fold"] for r in after)


def test_aggregate_diagnostics_and_constant_prediction_handling():
    frames, windows = fixture_data()
    frames["catboost"]["prediction"] = 0.2
    data = align_predictions(frames, windows, expected_rows=200)
    result = diagnostics(data, {"catboost": 1})
    assert sum(row["rows"] for row in result["weekly"]) == 200
    assert sum(row["rows"] for row in result["reliability"]) == 200
    assert len(result["reliability"]) == 1
    assert all("case_id" not in row for rows in result.values() for row in rows)
    assert all(
        row["pearson_prediction_correlation"] is None
        for row in result["prediction_correlations"]
        if "catboost" in {row["first"], row["second"]}
    )


class MemoryStore:
    def __init__(self, fail_after=None):
        self.state = None
        self.commits = 0
        self.fail_after = fail_after

    def commit(self, state):
        if self.commits == self.fail_after:
            raise OSError("simulated upload interruption")
        self.state = copy.deepcopy(state)
        self.commits += 1


class MemoryLease:
    def check(self):
        pass


def test_interrupted_upload_reuses_only_durably_completed_candidates(tmp_path, monkeypatch):
    import home_credit.modeling.selection_workflow as workflow
    from home_credit.observability.logging import RunLogger

    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    candidates = fixed_candidates(NAMES, LEADER)
    store = MemoryStore(fail_after=1)
    logger = RunLogger("selection-test", tmp_path)
    state = {"trials": [], "revision": 0}
    with pytest.raises(OSError, match="upload interruption"):
        workflow.run_candidates(data, candidates, state, store, MemoryLease(), logger)
    assert state == {"trials": [], "revision": 0}
    assert len(store.state["trials"]) == 1
    assert store.state["trials"][0]["name"] == LEADER
    scored = []
    evaluate = workflow.evaluate_candidate

    def spy(data, name, weights):
        scored.append(name)
        return evaluate(data, name, weights)

    monkeypatch.setattr(workflow, "evaluate_candidate", spy)
    store.fail_after = None
    result = workflow.run_candidates(data, candidates, store.state, store, MemoryLease(), logger)
    assert len(result["trials"]) == 15
    assert len(scored) == 14 and LEADER not in scored
    scored.clear()
    workflow.run_candidates(data, candidates, result, store, MemoryLease(), logger)
    assert scored == []


def test_lease_loss_cannot_publish_a_new_candidate(tmp_path):
    from home_credit.modeling.selection_workflow import run_candidates
    from home_credit.observability.logging import RunLogger

    class LostLease:
        def check(self):
            raise RuntimeError("lease lost")

    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    store = MemoryStore()
    with pytest.raises(RuntimeError, match="lease lost"):
        run_candidates(
            data,
            fixed_candidates(NAMES, LEADER),
            {"trials": [], "revision": 0},
            store,
            LostLease(),
            RunLogger("lease-test", tmp_path),
        )
    assert store.commits == 0


def test_corrupt_cached_candidate_is_rejected_before_advance(tmp_path):
    from home_credit.modeling.selection_workflow import run_candidates
    from home_credit.observability.logging import RunLogger

    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    record = evaluate_candidate(data, LEADER, {LEADER: 1})
    record["weights"] = {"xgboost": 1}
    store = MemoryStore()
    with pytest.raises(ValueError, match="cached candidate changed"):
        run_candidates(
            data,
            fixed_candidates(NAMES, LEADER),
            {"trials": [record], "revision": 1},
            store,
            MemoryLease(),
            RunLogger("cache-test", tmp_path),
        )
    assert store.commits == 0


def test_tuning_reference_recomputed_metrics_must_match():
    from home_credit.modeling.selection_workflow import verify_tuning_reference

    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    record = evaluate_candidate(data, LEADER, {LEADER: 1})
    study = {"trials": [copy.deepcopy(record)], "selected_trial": LEADER}
    verify_tuning_reference(record, study)
    study["trials"][0]["metrics"]["oof_auc"] += 0.01
    with pytest.raises(ValueError, match="tuned metric mismatch"):
        verify_tuning_reference(record, study)


def make_selection_result():
    from home_credit.modeling.selection_workflow import selection_result

    frames, windows = fixture_data()
    data = align_predictions(frames, windows, expected_rows=200)
    trials = [
        evaluate_candidate(data, name, weights)
        for name, weights in fixed_candidates(NAMES, LEADER).items()
    ]
    plan = {
        "leader": LEADER,
        "study": {"sha256": "synthetic-integration"},
        "selection_scope": "SYNTHETIC INTEGRATION TEST; not competition results",
        "prequential_scope": "Not unbiased nested validation",
    }
    return selection_result(data, {"trials": trials, "identity": {"smoke": True}}, plan)


def test_report_preserves_scope_and_unchanged_notebook_outputs(tmp_path):
    import json

    import nbformat

    from home_credit.modeling.selection_report import (
        write_selection_notebook,
        write_selection_report,
    )

    result = make_selection_result()
    paths = write_selection_report(result, tmp_path)
    assert len(paths) == 4 and all(p.is_file() for p in paths)
    page = (tmp_path / "report.html").read_text()
    assert "no new training" in page and "no calibrator" in page.lower()
    assert "<svg" in page and "case_id" not in page
    (tmp_path / "selection.json").write_text(json.dumps(result, allow_nan=False))
    path = tmp_path / "08_model_selection.ipynb"
    write_selection_notebook(path)
    notebook = nbformat.read(path, as_version=4)
    assert len(notebook.cells) == 8
    assert not any("competitions_submit" in c.source for c in notebook.cells)
    notebook.metadata["test_marker"] = "preserve completed work"
    nbformat.write(notebook, path)
    original = path.read_bytes()
    write_selection_notebook(path)
    assert path.read_bytes() == original
    assert json.dumps(result, allow_nan=False)


def test_selection_plan_is_explicitly_development_only():
    import json
    from pathlib import Path

    plan = json.loads(Path("configs/model_selection.json").read_text())
    assert plan["candidate_budget"] == 15 and plan["new_model_fits"] == 0
    assert plan["outer_holdout_touched"] is False
    assert plan["expected_rows"] == 727187
    assert [s["name"] for s in plan["sources"]] == NAMES
    assert len(plan["study"]["sha256"]) == 64
    for policy in plan["manifests"].values():
        assert policy["sha256"] in policy["key"]
