"""Durable OOF selection orchestration; no native model fit or Kaggle action."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.selection import (
    AlignedPredictions,
    align_predictions,
    diagnostics,
    evaluate_candidate,
    fixed_candidates,
    prequential_choices,
    rank_records,
    require,
    validate_record,
)
from home_credit.observability.runtime import StageTimer
from home_credit.validation.protocol import verify_protocol_sha256

if TYPE_CHECKING:
    from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
    from home_credit.observability.logging import RunLogger


def load_inputs(
    root: Path, plan: dict[str, Any], store: ExperimentStore, logger: RunLogger
) -> tuple[AlignedPredictions, dict[str, Any], dict[str, Any]]:
    """Read only four hash-pinned development prediction files and small manifests."""
    protocol = json.loads((root / "configs/validation_protocol.json").read_text())
    require(verify_protocol_sha256(protocol), "protocol content changed")
    require(protocol["protocol_sha256"] == plan["protocol_sha256"], "protocol identity changed")
    require(protocol["outer_holdout"]["locked"] is True, "holdout is not locked")
    windows = protocol["inner_temporal_cv"]["folds"]
    require(len(windows) == 5 and windows[-1]["validation_week_max"] == 72, "fold policy changed")
    require(plan["new_model_fits"] == 0 and plan["candidate_budget"] == 15, "budget changed")
    require(plan["outer_holdout_touched"] is False, "selection cannot access the holdout")
    cache = root / "artifacts/model_selection/inputs"
    manifests = {}
    for name, policy in plan["manifests"].items():
        path = cache / f"{policy['sha256']}.json"
        store.download(policy["key"], path, policy["sha256"])
        manifest = json.loads(path.read_text())
        require(manifest["schema_version"] == 1 and manifest["smoke"] is False, "invalid manifest")
        require(
            manifest["feature_manifest_sha256"] == plan["feature_manifest_sha256"],
            "feature lineage changed",
        )
        require(
            manifest["validation_protocol_sha256"] == plan["protocol_sha256"],
            "manifest protocol changed",
        )
        require(
            manifest["completed_model_folds"] == policy["completed_model_folds"],
            "incomplete source",
        )
        manifests[name] = manifest
    policy = plan["study"]
    study_path = cache / f"{policy['sha256']}.json"
    store.download(policy["key"], study_path, policy["sha256"])
    study = json.loads(study_path.read_text())
    require(study["complete"] is True and study["identity"]["smoke"] is False, "tuning incomplete")
    require(study["outer_holdout_touched"] is False, "tuning used the holdout")
    require(
        len(study["trials"]) == 9 and all(r["state"] == "complete" for r in study["trials"]),
        "incomplete tuning candidates",
    )
    require(study["selected_trial"] == policy["selected_trial"], "tuning winner changed")
    require(
        rank_records(study["trials"], "control")[0]["name"] == policy["selected_trial"],
        "tuning selection mismatch",
    )
    frames = {}
    for source in plan["sources"]:
        name = source["name"]
        require(name not in frames, "duplicate input model")
        matches = [f for f in manifests[source["manifest"]]["files"] if f["path"] == source["path"]]
        require(len(matches) == 1, "prediction object absent or duplicated")
        item = matches[0]
        require(item["sha256"] == source["sha256"], "prediction digest differs from plan")
        require(item["path"].startswith("oof/"), "only OOF prediction inputs are allowed")
        path = cache / f"{item['sha256']}.parquet"
        store.download(item["object_key"], path, item["sha256"])
        require(path.stat().st_size == item["bytes"], "prediction length mismatch")
        frames[name] = pd.read_parquet(path)
    with StageTimer(logger, "verify_prediction_alignment", heartbeat_seconds=15):
        data = align_predictions(frames, windows, expected_rows=plan["expected_rows"])
    return data, study, protocol


def run_candidates(
    data: AlignedPredictions,
    candidates: dict[str, dict[str, float]],
    state: dict[str, Any],
    store: ExperimentStore,
    lease: WriterLease,
    logger: RunLogger,
) -> dict[str, Any]:
    """Commit each completed candidate before advancing; resume without rescoring it."""
    existing = {record["name"]: record for record in state["trials"]}
    require(len(existing) == len(state["trials"]), "duplicate cached candidate")
    require(set(existing) <= set(candidates), "unexpected cached candidate")
    for name, weights in candidates.items():
        lease.check()
        if name in existing:
            validate_record(existing[name], name, weights, len(set(data.fold.tolist())))
            logger.event(
                "selection_candidate_reused",
                candidate=name,
                completed=len(state["trials"]),
                total=len(candidates),
            )
            continue
        with StageTimer(logger, f"evaluate_{name}", heartbeat_seconds=15):
            record = evaluate_candidate(data, name, weights)
        lease.check()
        updated = copy.deepcopy(state)
        updated["trials"].append(record)
        updated["revision"] += 1
        store.commit(updated)
        state = updated
        logger.event(
            "selection_candidate_completed",
            candidate=name,
            completed=len(state["trials"]),
            total=len(candidates),
            mean_fold_stability=record["metrics"]["mean_fold_stability"],
        )
    return state


def verify_tuning_reference(record: dict[str, Any], study: dict[str, Any]) -> None:
    """Independently match the saved winner to newly recomputed OOF metrics."""
    reference = next(r for r in study["trials"] if r["name"] == study["selected_trial"])
    for name, value in record["metrics"].items():
        require(
            math.isclose(value, reference["metrics"][name], rel_tol=1e-8, abs_tol=1e-10),
            f"tuned metric mismatch: {name}",
        )
    for actual, expected in zip(record["folds"], reference["folds"], strict=True):
        require(actual["fold"] == expected["fold"], "tuned fold order differs")
        for key, value in actual.items():
            require(
                math.isclose(value, expected[key], rel_tol=1e-8, abs_tol=1e-10),
                f"tuned fold metric mismatch: {key}",
            )


def selection_result(
    data: AlignedPredictions, state: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Build aggregate-only evidence with the exact selected weights and caveats."""
    leader = plan["leader"]
    ranked = rank_records(state["trials"], leader)
    incumbent = next(row for row in ranked if row["name"] == leader)
    selected = ranked[0]
    rows = []
    for record in ranked:
        rows.append(
            {
                "candidate": record["name"],
                **record["metrics"],
                "delta_vs_tuned_lightgbm": record["metrics"]["mean_fold_stability"]
                - incumbent["metrics"]["mean_fold_stability"],
            }
        )
    return {
        "schema_version": 1,
        "scope": plan["selection_scope"],
        "smoke": bool(state["identity"].get("smoke", False)),
        "outer_holdout_touched": False,
        "rows_evaluated": len(data.target),
        "new_model_fits": 0,
        "candidate_count": len(ranked),
        "selected_candidate": selected["name"],
        "selected_weights": selected["weights"],
        "identity": state["identity"],
        "source_study_sha256": plan["study"]["sha256"],
        "rows": rows,
        "folds": [{"candidate": row["name"], **fold} for row in ranked for fold in row["folds"]],
        "prequential_scope": plan["prequential_scope"],
        "prequential_weight_choices": prequential_choices(ranked, leader),
        "diagnostics": diagnostics(data, selected["weights"]),
        "calibration_fitted": False,
        "kaggle_submitted": False,
    }


def execute_selection(
    root: Path,
    plan: dict[str, Any],
    identity: dict[str, Any],
    store: ExperimentStore,
    lease: WriterLease,
    logger: RunLogger,
) -> dict[str, Any]:
    """Recover verified inputs and candidate receipts before producing the report."""
    state = store.restore(identity)
    if state is None:
        state = {
            "schema_version": 1,
            "identity": identity,
            "revision": 0,
            "trials": [],
            "complete": False,
        }
        store.commit(state)
    with StageTimer(logger, "restore_development_predictions", heartbeat_seconds=15):
        data, study, _ = load_inputs(root, plan, store, logger)
    candidates = fixed_candidates([source["name"] for source in plan["sources"]], plan["leader"])
    state = run_candidates(data, candidates, state, store, lease, logger)
    incumbent = next(row for row in state["trials"] if row["name"] == plan["leader"])
    verify_tuning_reference(incumbent, study)
    with StageTimer(logger, "selection_diagnostics", heartbeat_seconds=15):
        result = selection_result(data, state, plan)
    lease.check()
    path = store.root / "report/selection.json"
    atomic_write(path, canonical_json_bytes(result))
    state["selection_sha256"] = sha256_file(path)
    state["selected_candidate"] = result["selected_candidate"]
    return state
