"""Offline provenance, publication-boundary and scientific-outcome regression tests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import summarize_aws_snapshot as snapshot
from summarize_execution import read_json, sha256


def documents() -> dict[str, Any]:
    selected = {
        "mean_stability_delta": 0.013320499345895964,
        "wins": 3,
        "private_recipe": "PRIVATE_SELECTION_DO_NOT_PUBLISH",
    }
    decision = {
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
        "selected": selected,
    }
    current = {
        "run_id": snapshot.RUN_ID,
        "parent_run_id": snapshot.PARENT_RUN_ID,
        "status": "SUCCESS",
        "completed_tasks": 42,
        "new_completed_fits": 5,
        "scientific_fits": 5,
    }
    result: dict[str, Any] = {
        "execution_ledger": {"runs": [current]},
        "selection": {
            "status": "PASS",
            "decision": {
                "status": "SELECTION_PASS",
                "selection_passed": True,
                "confirmation_labels_accessed": False,
                "selected": selected,
            },
        },
        "confirmation": {"status": "PASS", "decision": decision},
        "conditional_release": {
            "status": "PASS",
            "conditional_not_applicable": True,
            "submission_attempted": False,
            "scientific_model_fits": 0,
        },
        "selftests": {
            "status": "PASS",
            "tests": 692,
            "failure_count": 0,
            "error_count": 0,
            "skipped": 0,
        },
        "inspection_receipt": {
            "observed_at_utc": snapshot.OBSERVED_UTC,
            "mutations": 0,
            "compute_started": False,
            "remote_shell_executed": False,
            "private_model_config_or_prediction_bytes_read": False,
            "studio_filesystem_read": False,
        },
        "return_commit": {"metadata": {"run_id": snapshot.RUN_ID}},
    }
    for role, target, member in (
        ("ledger_commit", "execution_ledger", "ledger.json"),
        ("evidence_commit", "selftests", "run/selftests.json"),
        ("selection_commit", "selection", "result.json"),
        ("confirmation_commit", "confirmation", "result.json"),
        ("conditional_release_commit", "conditional_release", "result.json"),
    ):
        pin = snapshot.SOURCES[target]
        result[role] = {
            "metadata": {"run_id": snapshot.RUN_ID, "status": "SUCCESS"},
            "files": [
                {
                    "path": member,
                    "object": {
                        "bytes": pin["bytes"],
                        "sha256": pin["sha256"],
                        "key": "PRIVATE_CLOUD_LOCATOR",
                    },
                }
            ],
        }
    for stage, role in ((28, "selection"), (33, "confirmation"), (39, "conditional_release")):
        result[role + "_commit"]["metadata"].update(
            stage=stage, result_sha256=snapshot.SOURCES[role]["sha256"]
        )
    return result


class AwsSnapshotTests(unittest.TestCase):
    def test_actual_public_snapshot_and_exact_types(self) -> None:
        snapshot.validate_snapshot(read_json(snapshot.OUTPUT.read_bytes()))
        for field, value in (
            ("new_completed_fits", 5.0),
            ("release_allowed", True),
            ("private_recipe", "secret"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                snapshot.validate_snapshot(snapshot.expected_snapshot() | {field: value})

    def test_receipt_hash_size_and_symlink_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            raw = b'{"status":"SUCCESS"}'
            path.write_bytes(raw)
            pins = {"test": {"file": path.name, "bytes": len(raw), "sha256": sha256(raw)}}
            with patch.object(snapshot, "SOURCES", pins):
                self.assertEqual(
                    snapshot.read_verified(Path(directory))["test"]["status"], "SUCCESS"
                )
                path.write_bytes(raw.replace(b"SUCCESS", b"FAILURE"))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    snapshot.read_verified(Path(directory))
                path.write_bytes(raw + b"extra")
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    snapshot.read_verified(Path(directory))
                path.unlink()
                path.symlink_to(Path(directory) / "elsewhere")
                with self.assertRaisesRegex(ValueError, "linked receipt"):
                    snapshot.read_verified(Path(directory))

    def test_private_fields_never_enter_public_output(self) -> None:
        source = documents()
        source["execution_ledger"]["private_predictions"] = "PRIVATE_PREDICTIONS"
        with patch.object(snapshot, "read_verified", return_value=source):
            result = snapshot.summarize(Path("unused"))
        snapshot.validate_snapshot(result)
        self.assertNotIn("PRIVATE_", json.dumps(result))
        self.assertFalse(result["verification"]["private_predictions_replayed"])
        self.assertIsNone(result["execution_ended_utc"])
        self.assertEqual(result["execution_status"], "SUCCESS")
        self.assertEqual(result["scientific_outcome"], "CONFIRMATION_REJECTED")

    def test_commit_bindings_reject_wrong_target_and_stage(self) -> None:
        for kind in ("hash", "stage", "count"):
            source = documents()
            if kind == "hash":
                source["ledger_commit"]["files"][0]["object"]["sha256"] = "0" * 64
            elif kind == "stage":
                source["confirmation_commit"]["metadata"]["stage"] = 32
            else:
                source["evidence_commit"]["files"] *= 2
            with (
                self.subTest(kind=kind),
                patch.object(snapshot, "read_verified", return_value=source),
                self.assertRaises(ValueError),
            ):
                snapshot.summarize(Path("unused"))

    def test_promotion_submission_and_holdout_misclaims_rejected(self) -> None:
        for field in (
            "confirmation_passed",
            "selection_reopened",
            "release_allowed",
            "submission_ready",
            "holdout_weeks_73_91_accessed",
        ):
            source = documents()
            source["confirmation"]["decision"][field] = True
            with (
                self.subTest(field=field),
                patch.object(snapshot, "read_verified", return_value=source),
                self.assertRaisesRegex(ValueError, "confirmation outcome"),
            ):
                snapshot.summarize(Path("unused"))
        source = documents()
        source["conditional_release"]["submission_attempted"] = True
        with (
            patch.object(snapshot, "read_verified", return_value=source),
            self.assertRaisesRegex(ValueError, "conditional release"),
        ):
            snapshot.summarize(Path("unused"))

    def test_frozen_selection_and_inspection_scope_are_required(self) -> None:
        for kind in ("selection", "run", "mutation", "observation"):
            source = copy.deepcopy(documents())
            if kind == "selection":
                source["confirmation"]["decision"]["selected"] = {"wins": 0}
            elif kind == "run":
                source["execution_ledger"]["runs"][-1]["completed_tasks"] = 39
            elif kind == "mutation":
                source["inspection_receipt"]["mutations"] = 1
            else:
                source["inspection_receipt"]["observed_at_utc"] = "different"
            with (
                self.subTest(kind=kind),
                patch.object(snapshot, "read_verified", return_value=source),
                self.assertRaises(ValueError),
            ):
                snapshot.summarize(Path("unused"))


if __name__ == "__main__":
    unittest.main()
