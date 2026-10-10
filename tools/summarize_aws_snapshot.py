"""Publish a bounded snapshot from previously downloaded, authenticated AWS receipts.

Regenerate: python tools/summarize_aws_snapshot.py --receipts /path/to/private-receipts
Check public snapshot: python tools/summarize_aws_snapshot.py --check
No network requests, remote execution, configuration export or model replay occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_execution import accepted_scores, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/latest_execution/aws_snapshot.json"
RUN_ID = "20261010T030650590522Z-438c10fa"
PARENT_RUN_ID = "20261010T022535072312Z-ced3dd22"
OBSERVED_UTC = "2026-10-10T04:09:38.221796+00:00"
SOURCES: dict[str, dict[str, Any]] = {
    "evidence_commit": {
        "file": "evidence.json",
        "bytes": 4478,
        "sha256": "8ecb0478004beba69204bf47f0048a6165e4426e1357c9da55ca38d5880ddeb8",
        "last_modified": "2026-10-10T03:56:26+00:00",
        "kind": "remote_json",
    },
    "ledger_commit": {
        "file": "ledger.json",
        "bytes": 639,
        "sha256": "e6a9cd2797aba5d5953c537c0049544cba7f772d16305a5be3c7554526388f1d",
        "last_modified": "2026-10-10T03:56:26+00:00",
        "kind": "remote_json",
    },
    "return_commit": {
        "file": "return_evidence.json",
        "bytes": 3791,
        "sha256": "cc1c8b54e02996a222a90d1fbbe5291d712571b9bcf368bfc7ef180a8d63659c",
        "last_modified": "2026-10-10T03:56:27+00:00",
        "kind": "remote_json",
    },
    "conditional_release": {
        "file": "stage39_result.json",
        "bytes": 210,
        "sha256": "4407dae56bc46da5571146ec6f84e1f694f1cbfa82cd45628e662b2ed6360b8e",
        "last_modified": "2026-10-10T03:56:19+00:00",
        "kind": "remote_json",
    },
    "diagnostic": {
        "file": "diagnostic_receipt.json",
        "bytes": 1736,
        "sha256": "9aa11bc4c90fc95f6b103174acfdc5c3bd9e4fa52fca1b156de0eaa2409b6945",
        "last_modified": "2026-10-10T03:56:27+00:00",
        "kind": "remote_json",
    },
    "execution_ledger": {
        "file": "execution_ledger.json",
        "bytes": 40235,
        "sha256": "482ed162b03f8a52f45fe306dabe853c1cd9c4c6bbae446ad85ee65b17776940",
        "last_modified": "2026-10-10T03:56:25+00:00",
        "kind": "remote_json",
    },
    "selftests": {
        "file": "run_selftests.json",
        "bytes": 119,
        "sha256": "846e4de6feb33514f48c7de9cd860e6c3c04d9ceed0d9ea3f850480048480e46",
        "last_modified": "2026-10-10T03:09:06+00:00",
        "kind": "remote_json",
    },
    "selection_commit": {
        "file": "stage28.json",
        "bytes": 1167,
        "sha256": "5ae42c086afcd5d9fdeb9e9aad3f68c7f8ee3acd757311f9390783b07b65fde6",
        "last_modified": "2026-10-10T03:38:28+00:00",
        "kind": "remote_json",
    },
    "confirmation_commit": {
        "file": "stage33.json",
        "bytes": 1166,
        "sha256": "f1b0a1812107a951a419cfc680c9140bb1f4fcfe4fcd5fa846e264c727bb9889",
        "last_modified": "2026-10-10T03:56:17+00:00",
        "kind": "remote_json",
    },
    "selection": {
        "file": "stage28_result.json",
        "bytes": 1672,
        "sha256": "7ec34e2f4ebff3a34e99a3242c43ff8b01c11fd5d2b2cfc1c3741e0024503140",
        "last_modified": "2026-10-10T03:38:28+00:00",
        "kind": "remote_json",
    },
    "confirmation": {
        "file": "stage33_result.json",
        "bytes": 959,
        "sha256": "2f0073ff7fa3e03f234944612eef548c511a818f9fa8085012d987d75311f400",
        "last_modified": "2026-10-10T03:56:17+00:00",
        "kind": "remote_json",
    },
    "conditional_release_commit": {
        "file": "stage39.json",
        "bytes": 1166,
        "sha256": "c48a50ee99b8fc2b28d8ea08d3bd9db0622b89555e78b7ad8bc866db0eb31ae4",
        "last_modified": "2026-10-10T03:56:19+00:00",
        "kind": "remote_json",
    },
    "inspection_receipt": {
        "file": "aws_read_observation.json",
        "bytes": 1050,
        "sha256": "616f11866433b6347936946c344e9a3f3dd025350fbb45bc17eff0c2548af0ab",
        "last_modified": None,
        "kind": "local_inspection_receipt",
    },
}


def read_verified(directory: Path) -> dict[str, dict[str, Any]]:
    """Read only fixed, size-bounded JSON files; no private paths reach the output."""
    result = {}
    for role, pin in SOURCES.items():
        path = directory / pin["file"]
        require(not path.is_symlink() and path.is_file(), "missing or linked receipt")
        require(path.stat().st_size == pin["bytes"] <= 100_000, "receipt size mismatch")
        raw = path.read_bytes()
        require(sha256(raw) == pin["sha256"], "receipt SHA-256 mismatch")
        result[role] = read_json(raw)
    return result


def bind(
    documents: dict[str, dict[str, Any]], commit_role: str, target_role: str, member: str
) -> None:
    """Corroborate target identity against its authenticated immutable commit."""
    found = [item for item in documents[commit_role]["files"] if item["path"] == member]
    require(len(found) == 1, "commit member coverage mismatch")
    receipt = found[0]["object"]
    pin = SOURCES[target_role]
    require(
        receipt["bytes"] == pin["bytes"] and receipt["sha256"] == pin["sha256"],
        "commit target identity mismatch",
    )


def provenance() -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "bytes": pin["bytes"],
            "sha256": pin["sha256"],
            "source_last_modified_utc": pin["last_modified"],
            "kind": pin["kind"],
        }
        for role, pin in SOURCES.items()
    ]


def public_result(
    current: dict[str, Any],
    selected: dict[str, Any],
    confirmation: dict[str, Any],
    tests: dict[str, Any],
) -> dict[str, Any]:
    """Use explicit fields only; candidate identities, weights and cloud locators stay private."""
    return {
        "schema_version": 1,
        "scope": "inspected_aws_execution_snapshot",
        "version": "E98",
        "run_id": current["run_id"],
        "parent_run_id": current["parent_run_id"],
        "observed_at_utc": OBSERVED_UTC,
        "execution_ended_utc": None,
        "execution_status": current["status"],
        "completed_tasks_recorded_in_ledger": current["completed_tasks"],
        "new_completed_fits": current["new_completed_fits"],
        "scientific_outcome": confirmation["status"],
        "selection": {
            "reported_mean_stability_delta": selected["mean_stability_delta"],
            "reported_stability_wins": selected["wins"],
        },
        "confirmation": {
            "passed": confirmation["confirmation_passed"],
            "reported_mean_stability_delta": confirmation["mean_stability_delta"],
            "reported_worst_stability_delta": confirmation["worst_stability_delta"],
            "reported_mean_auc_delta": confirmation["mean_auc_delta"],
            "reported_stability_wins": confirmation["wins"],
            "selection_reopened": confirmation["selection_reopened"],
            "weeks_73_91_accessed_in_this_run": confirmation["holdout_weeks_73_91_accessed"],
        },
        "release_allowed": confirmation["release_allowed"],
        "submission_ready": confirmation["submission_ready"],
        "submission_attempted": False,
        "score_improvement_claim": False,
        "existing_external_scores": accepted_scores(),
        "reported_selftests": {
            "status": tests["status"],
            "tests": tests["tests"],
            "rerun_here": False,
        },
        "verification": {
            "remote_json_bodies_verified": 12,
            "immutable_commit_target_bindings_verified": 5,
            "read_only_inspection": True,
            "private_models_or_predictions_downloaded": False,
            "private_configuration_downloaded": False,
            "private_predictions_replayed": False,
            "live_studio_filesystem_read": False,
            "complete_remote_inventory_claim": False,
        },
        "provenance": provenance(),
        "limitations": [
            "Execution success and scientific rejection are separate outcomes.",
            "These metrics are receipt-derived aggregates, not independent prediction replay.",
            "Object modification and inspection timestamps are not execution end timestamps.",
            "This run did not reopen weeks 73-91; those weeks were already observed historically.",
            "External scores retain their earlier source; no new submission was attempted.",
            "This snapshot covers selected inspected records, not every cloud object or process.",
        ],
    }


def summarize(directory: Path) -> dict[str, Any]:
    documents = read_verified(directory)
    bind(documents, "ledger_commit", "execution_ledger", "ledger.json")
    bind(documents, "evidence_commit", "selftests", "run/selftests.json")
    for stage, role in ((28, "selection"), (33, "confirmation"), (39, "conditional_release")):
        bind(documents, role + "_commit", role, "result.json")
        require(
            documents[role + "_commit"]["metadata"]["stage"] == stage, "stage identity mismatch"
        )
        require(
            documents[role + "_commit"]["metadata"]["result_sha256"] == SOURCES[role]["sha256"],
            "stage result identity mismatch",
        )
        require(documents[role]["status"] == "PASS", "incomplete stage receipt")
    current = documents["execution_ledger"]["runs"][-1]
    require(
        current["run_id"] == RUN_ID
        and current["parent_run_id"] == PARENT_RUN_ID
        and current["status"] == "SUCCESS"
        and current["completed_tasks"] == 42
        and current["new_completed_fits"] == current["scientific_fits"] == 5,
        "execution ledger disagreement",
    )
    require(
        documents["ledger_commit"]["metadata"]["run_id"] == RUN_ID
        and documents["ledger_commit"]["metadata"]["status"] == "SUCCESS"
        and documents["evidence_commit"]["metadata"]["status"] == "SUCCESS"
        and documents["return_commit"]["metadata"]["run_id"] == RUN_ID,
        "final commit disagreement",
    )
    selected = documents["selection"]["decision"]
    confirmation = documents["confirmation"]["decision"]
    require(
        selected["status"] == "SELECTION_PASS"
        and selected["selection_passed"] is True
        and selected["confirmation_labels_accessed"] is False
        and selected["selected"] == confirmation["selected"],
        "frozen selection disagreement",
    )
    require(
        confirmation["status"] == "CONFIRMATION_REJECTED"
        and all(
            confirmation[key] is False
            for key in (
                "confirmation_passed",
                "selection_reopened",
                "holdout_weeks_73_91_accessed",
                "release_allowed",
                "submission_ready",
            )
        ),
        "confirmation outcome disagreement",
    )
    release = documents["conditional_release"]
    require(
        release["conditional_not_applicable"] is True
        and release["submission_attempted"] is False
        and release["scientific_model_fits"] == 0,
        "conditional release disagreement",
    )
    tests = documents["selftests"]
    require(
        tests["status"] == "PASS"
        and tests["tests"] == 692
        and tests["failure_count"] == tests["error_count"] == tests["skipped"] == 0,
        "selftest receipt disagreement",
    )
    observation = documents["inspection_receipt"]
    require(
        observation["observed_at_utc"] == OBSERVED_UTC
        and observation["mutations"] == 0
        and all(
            observation[key] is False
            for key in (
                "compute_started",
                "remote_shell_executed",
                "private_model_config_or_prediction_bytes_read",
                "studio_filesystem_read",
            )
        ),
        "inspection scope disagreement",
    )
    return public_result(current, selected["selected"], confirmation, tests)


def expected_snapshot() -> dict[str, Any]:
    return public_result(
        {
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "status": "SUCCESS",
            "completed_tasks": 42,
            "new_completed_fits": 5,
        },
        {"mean_stability_delta": 0.013320499345895964, "wins": 3},
        {
            "status": "CONFIRMATION_REJECTED",
            "confirmation_passed": False,
            "mean_stability_delta": -0.0032024321890856133,
            "worst_stability_delta": -0.004158623513201065,
            "mean_auc_delta": 0.0013394978106975874,
            "wins": 0,
            "selection_reopened": False,
            "holdout_weeks_73_91_accessed": False,
            "release_allowed": False,
            "submission_ready": False,
        },
        {"status": "PASS", "tests": 692},
    )


def validate_snapshot(value: dict[str, Any]) -> None:
    require(
        json.dumps(value, sort_keys=True, allow_nan=False)
        == json.dumps(expected_snapshot(), sort_keys=True, allow_nan=False),
        "AWS snapshot contract mismatch",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--receipts", type=Path)
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.check:
        validate_snapshot(read_json(args.output.read_bytes()))
        print("PASS: E98 aggregate snapshot, evidence boundaries and prior score consistency")
    else:
        result = summarize(args.receipts)
        validate_snapshot(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            "PASS: verified inspected receipt bytes and commit bindings; published aggregates only"
        )


if __name__ == "__main__":
    main()
