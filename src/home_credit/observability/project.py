"""Read-only project status from validated evidence and optional live AWS reads."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from home_credit.modeling.calibration_report import load_evidence as load_calibration
from home_credit.modeling.feature_research_report import load_evidence as load_research
from home_credit.modeling.history_report import load_evidence as load_history
from home_credit.modeling.portfolio import load_portfolio
from home_credit.modeling.release import checked_member, read_object, verify_file
from home_credit.modeling.selection import require

PROJECT_PREFIX = "home-credit-model-stability/"
ACTIVE_STATES = {"InProgress", "Stopping"}


def fold_sensitivity(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Paired fold sensitivity, without treating five overlapping fits as independent."""
    folds = {(r["experiment"], r["fold"]): r["stability_score"] for r in result["folds"]}
    rows = []
    for row in result["rows"]:
        name = row["experiment"]
        if name == "control":
            continue
        deltas = [folds[name, f] - folds["control", f] for f in range(1, 6)]
        omitted = [(sum(deltas) - value) / 4 for value in deltas]
        rows.append(
            {
                "experiment": name,
                "mean_delta": sum(deltas) / 5,
                "improved_folds": sum(value > 0 for value in deltas),
                "folds": 5,
                "leave_one_fold_out_delta_min": min(omitted),
                "leave_one_fold_out_delta_max": max(omitted),
            }
        )
    return rows


def published_status(root: Path) -> dict[str, Any]:
    """Reject stale evidence before returning results; never infer live cloud state."""
    portfolio = load_portfolio(root)
    research = load_research(root)["result"]
    history_evidence = load_history(root)
    history = history_evidence["result"]
    calibration = load_calibration(root)
    original = portfolio["features"]
    gate = read_object(root / "configs/research_gate.json")
    require(gate["schema_version"] == 1, "unsupported research gate")
    requirements = gate["requirements"]
    require(
        len({r["id"] for r in requirements}) == len(requirements) and bool(requirements),
        "missing or duplicate research requirement",
    )
    require(all(r["status"] in {"open", "addressed"} for r in requirements), "invalid gate state")
    for requirement in requirements:
        if requirement["status"] == "addressed":
            # A prose status flag cannot close an unexecuted research requirement.
            require(bool(requirement.get("evidence")), "addressed gate requires evidence")
            for member in requirement["evidence"]:
                verify_file(checked_member(root, member["path"]), member["sha256"])
    open_requirements = [r for r in requirements if r["status"] == "open"]
    notebooks = []
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    paths.append(root / "reports/feature_ablation/06_feature_ablation.ipynb")
    for path in paths:
        notebook = read_object(path)
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        errors = sum(o.get("output_type") == "error" for c in cells for o in c["outputs"])
        executed = sum(c["execution_count"] is not None for c in cells)
        notebooks.append(
            {
                "path": path.relative_to(root).as_posix(),
                "code_cells": len(cells),
                "executed_cells": executed,
                "errors": errors,
                "has_complete_saved_outputs": bool(cells) and executed == len(cells) and not errors,
            }
        )
    return {
        "schema_version": 1,
        "published_evidence_verified": True,
        "release_complete": portfolio["state"]["complete"],
        "expanded_study_complete": True,
        "raw_history_study_complete": True,
        "feature_completion_gate_passed": not open_requirements,
        "remaining_requirements": open_requirements,
        "holdout": {"weeks": [73, 91], "observed": True, "available_for_new_selection": False},
        "features": {
            "original_candidates": original["candidate_count"],
            "original_eligible": original["eligible_count"],
            "release_retained": original["retained_count"],
            "original_rejected": original["rejected_count"],
            "additional_candidates": research["additional_candidates"],
            "additional_retained_for_experiment": research["additional_retained"],
            "additional_rejected": research["additional_rejected"],
            "raw_history_candidates": history["additional_candidates"],
            "raw_history_retained_for_experiment": history["additional_retained"],
            "raw_history_rejected": history["additional_rejected"],
            "combined_hypotheses": original["candidate_count"]
            + research["additional_candidates"]
            + history["additional_candidates"],
            "additions_promoted_to_release": 0,
        },
        "expanded_study_fits": research["new_model_fits"],
        "raw_history_study_fits": history["new_model_fits"],
        "calibrator_fits": calibration["new_calibrator_fits"],
        "frozen_evaluation": portfolio["evaluation"]["metrics"],
        "feature_comparisons": research["rows"],
        "raw_history_comparisons": history["rows"],
        "raw_history_fold_sensitivity": fold_sensitivity(history),
        "raw_history_verification": history_evidence["verification"],
        "fold_sensitivity": fold_sensitivity(research),
        "sensitivity_scope": "Descriptive fold-omission sensitivity; not a confidence interval.",
        "notebooks": notebooks,
        "notebook_scope": "Saved output inventory; current-source execution is checked by CI.",
        "cloud": {"checked": False},
        "new_model_fits": 0,
    }


def cloud_jobs(client: Any, region: str) -> dict[str, Any]:
    """Paginate this project's jobs and describe each; access errors propagate."""
    jobs = []
    for kind in ("processing", "training"):
        operation = f"list_{kind}_jobs"
        summary_key = "ProcessingJobSummaries" if kind == "processing" else "TrainingJobSummaries"
        name_key = "ProcessingJobName" if kind == "processing" else "TrainingJobName"
        status_key = "ProcessingJobStatus" if kind == "processing" else "TrainingJobStatus"
        for page in client.get_paginator(operation).paginate(
            NameContains="home-credit", SortBy="CreationTime", SortOrder="Descending"
        ):
            for summary in page.get(summary_key, []):
                name = summary[name_key]
                if not name.startswith("home-credit-"):
                    continue
                detail = getattr(client, f"describe_{kind}_job")(**{name_key: name})
                start_key = "ProcessingStartTime" if kind == "processing" else "TrainingStartTime"
                end_key = "ProcessingEndTime" if kind == "processing" else "TrainingEndTime"
                start, end = detail.get(start_key), detail.get(end_key)
                elapsed = None
                if start is not None:
                    elapsed = max(0.0, ((end or datetime.now(UTC)) - start).total_seconds())
                jobs.append(
                    {
                        "name": name,
                        "kind": kind,
                        "status": detail[status_key],
                        "started_utc": start.isoformat() if start else None,
                        "ended_utc": end.isoformat() if end else None,
                        "elapsed_seconds": elapsed,
                        "failure_reason": detail.get("FailureReason"),
                    }
                )
    jobs.sort(key=lambda row: (row["started_utc"] or "", row["name"]), reverse=True)
    return {
        "checked": True,
        "checked_utc": datetime.now(UTC).isoformat(),
        "region": region,
        "jobs": jobs,
        "active_jobs": [r["name"] for r in jobs if r["status"] in ACTIVE_STATES],
        "scope": "Project processing/training jobs in this region; excludes Studio/other services.",
    }


def artifact_members(value: Any) -> Iterable[dict[str, Any]]:
    """Walk saved ledgers without reading or printing the underlying borrower data."""
    if isinstance(value, dict):
        if {"key", "sha256", "bytes"} <= value.keys():
            yield {key: value[key] for key in ("key", "sha256", "bytes")}
        for child in value.values():
            yield from artifact_members(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_members(child)


def cloud_artifacts(root: Path) -> list[dict[str, Any]]:
    """Select immutable release/research/calibration checkpoints from validated reports."""
    # Verify the ledgers through their published identities before following their keys.
    portfolio = load_portfolio(root)
    research = load_research(root)
    calibration = load_calibration(root)
    history = load_history(root)
    members: dict[str, dict[str, Any]] = {}
    for data in (portfolio["state"], research["result"], calibration, history["study"]):
        for member in artifact_members(data):
            key = member["key"]
            require(key.startswith(PROJECT_PREFIX), "checkpoint belongs to another project")
            require(key not in members or members[key] == member, "conflicting checkpoint identity")
            members[key] = member
    require(bool(members), "no checkpoint identities available")
    return [members[key] for key in sorted(members)]


def verify_cloud_artifacts(
    client: Any, bucket: str, members: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stream full hashes, closing every body even on failure; no writes or retraining."""
    total = 0
    for member in members:
        response = client.get_object(Bucket=bucket, Key=member["key"])
        digest, count = hashlib.sha256(), 0
        with response["Body"] as body:
            require(response["ContentLength"] == member["bytes"], "cloud checkpoint size changed")
            while chunk := body.read(1024 * 1024):
                count += len(chunk)
                digest.update(chunk)
        require(count == member["bytes"], "cloud checkpoint truncated")
        require(digest.hexdigest() == member["sha256"], "cloud checkpoint digest changed")
        total += count
    require(math.isfinite(total), "invalid checkpoint byte count")
    return {"verified_objects": len(members), "verified_bytes": total, "method": "full_sha256"}
