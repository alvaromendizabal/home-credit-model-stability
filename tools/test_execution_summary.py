"""Offline integrity, scope and disclosure tests for the execution receipt."""

from __future__ import annotations

import copy
import io
import json
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import summarize_execution as summary


def encoded(value: object) -> bytes:
    return json.dumps(value).encode()


def fixtures() -> tuple[dict[str, Any], dict[str, bytes]]:
    totals = {"successful_runs": 58, "failed_runs": 36}
    manifest = {
        "run_id": summary.RUN_ID,
        "status": "STOPPED",
        "error": "MEMORY: configured available RAM or projected RSS guard",
        "new_completed_fits": 0,
        "new_training_attempts": 0,
        "reused_fits": 0,
        "scientific_fits": 0,
        "submission_ready": False,
        "leaderboard_score_changed": False,
        "submission_receipt": None,
        "scientific_outcome": None,
        "scientific_attempts": [],
        "completed_tasks": 13,
        "completed_stages": 13,
        "total_tasks": 42,
        "total_stages": 42,
        "remaining_tasks": 29,
        "ledger_totals": totals,
        "selftests": {
            "status": "PASS",
            "tests": 661,
            "failure_count": 0,
            "error_count": 0,
            "skipped": 0,
        },
        "full_evidence_remote": True,
        "unknown_private_input": "PRIVATE_MARKER_DO_NOT_PUBLISH",
    }
    stages: dict[str, Any] = {str(index): {"status": "PASS"} for index in range(1, 14)}
    stages["2"]["parity"] = {
        "status": "PASS",
        "rows": 10,
        "hidden_test_readiness_established": False,
    }
    payload = {
        "stage_results_summary.json": encoded({"stages": stages}),
        "execution_ledger.json": encoded(
            {
                "totals": totals,
                "runs": [
                    {
                        "run_id": summary.RUN_ID,
                        "status": "STOPPED",
                        "completed_tasks": 13,
                        "new_completed_fits": 0,
                    }
                ],
            }
        ),
        "complete_return_evidence_remote.json": encoded(
            {
                "complete_members_remote": True,
                "files": [{"private": "PRIVATE_REMOTE_LOCATOR"} for _ in range(9)],
            }
        ),
        # Intentionally not even JSON: private configuration must never be parsed.
        "config.json": b"PRIVATE_CONFIG_DO_NOT_PARSE",
    }
    return manifest, payload


def make_archive(
    path: Path,
    manifest: dict[str, Any],
    payload: dict[str, bytes],
    *,
    corrupt_receipt: bool = False,
    extra_tar: str | None = None,
    symlink: bool = False,
    missing_manifest: bool = False,
    duplicate_zip: bool = False,
) -> str:
    manifest = copy.deepcopy(manifest)
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:xz") as archive:
        for name, raw in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        if extra_tar:
            info = tarfile.TarInfo(extra_tar)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "config.json"
            archive.addfile(info, io.BytesIO(b""))
    manifest["evidence_artifacts"] = [
        {"path": name, "bytes": len(raw), "sha256": summary.sha256(raw)}
        for name, raw in payload.items()
    ]
    if corrupt_receipt:
        manifest["evidence_artifacts"][0]["sha256"] = "0" * 64
    raw_inner = inner.getvalue()
    manifest["return_artifacts"] = [
        {"path": "evidence.tar.xz", "bytes": len(raw_inner), "sha256": summary.sha256(raw_inner)}
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("evidence.tar.xz", raw_inner)
        if not missing_manifest:
            archive.writestr("manifest.json", encoded(manifest))
        if duplicate_zip:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("evidence.tar.xz", raw_inner)
    return summary.sha256(path.read_bytes())


class ExecutionSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "owner.zip"
        self.manifest, self.payload = fixtures()

    def test_allowlist_and_honest_evidence_scope(self) -> None:
        digest = make_archive(self.path, self.manifest, self.payload)
        with patch.object(tarfile.TarFile, "extract", side_effect=AssertionError("no extraction")):
            result = summary.summarize_archive(self.path, digest)
        text = json.dumps(result)
        self.assertNotIn("PRIVATE_", text)
        self.assertNotIn("config.json", text)
        self.assertEqual(
            result["progress"], {"completed_tasks": 13, "total_tasks": 42, "remaining_tasks": 29}
        )
        self.assertEqual(result["provenance"]["verified_inner_members"], 4)
        self.assertEqual(result["evidence_scope"]["remote_files_downloaded_by_reproducer"], 0)
        self.assertFalse(
            result["existing_external_scores"]["independently_reverified_by_this_return"]
        )
        self.assertIsNone(result["recorded_ended_utc"])

    def test_outer_and_nested_hash_tampering(self) -> None:
        digest = make_archive(self.path, self.manifest, self.payload)
        self.path.write_bytes(self.path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "archive SHA-256"):
            summary.inspect_archive(self.path, digest)
        digest = make_archive(self.path, self.manifest, self.payload, corrupt_receipt=True)
        with self.assertRaisesRegex(ValueError, "member SHA-256"):
            summary.inspect_archive(self.path, digest)

    def test_manifest_coverage_and_duplicates(self) -> None:
        for options in (
            {"missing_manifest": True},
            {"duplicate_zip": True},
            {"extra_tar": "extra.json"},
        ):
            with self.subTest(options=options):
                digest = make_archive(self.path, self.manifest, self.payload, **options)
                with self.assertRaises(ValueError):
                    summary.inspect_archive(self.path, digest)

    def test_unsafe_paths_and_link_members(self) -> None:
        for name in ("../outside", "/absolute", "a\\b", "a//b", "a/./b", "C:drive"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe member path"):
                summary.safe_path(name)
        digest = make_archive(
            self.path, self.manifest, self.payload, extra_tar="link", symlink=True
        )
        with self.assertRaisesRegex(ValueError, "linked or special"):
            summary.inspect_archive(self.path, digest)

    def test_archive_and_decompression_bounds(self) -> None:
        digest = make_archive(self.path, self.manifest, self.payload)
        with (
            patch.object(summary, "ARCHIVE_LIMIT", 1),
            self.assertRaisesRegex(ValueError, "size limit"),
        ):
            summary.inspect_archive(self.path, digest)
        with (
            patch.object(summary, "TOTAL_LIMIT", 1),
            self.assertRaisesRegex(ValueError, "decompressed"),
        ):
            summary.inspect_archive(self.path, digest)

    def test_unearned_promotions_and_activity_rejected(self) -> None:
        for key, value in (
            ("new_completed_fits", 1),
            ("submission_ready", True),
            ("scientific_outcome", "WIN"),
            ("completed_tasks", 14),
        ):
            with self.subTest(key=key):
                altered = self.manifest | {key: value}
                digest = make_archive(self.path, altered, self.payload)
                with self.assertRaises(ValueError):
                    summary.summarize_archive(self.path, digest)

    def test_public_snapshot_rejects_scope_and_private_fields(self) -> None:
        summary.validate_snapshot(summary.read_json(summary.OUTPUT.read_bytes()))
        for key, value in (
            ("private_config", "secret"),
            ("schema_version", True),
            ("submission_ready", True),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "snapshot contract"):
                summary.validate_snapshot(summary.expected_snapshot() | {key: value})
        altered = summary.expected_snapshot()
        altered["provenance"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "snapshot contract"):
            summary.validate_snapshot(altered)

    def test_duplicate_json_and_existing_score_disagreement(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            summary.read_json(b'{"status":"STOPPED","status":"PASS"}')
        path = Path(self.temp.name) / "score.json"
        path.write_bytes(
            encoded(
                {
                    "externally_measured_post_release_artifact": {
                        "kaggle_public": 0.99,
                        "kaggle_private": 0.47652,
                    }
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "public score mismatch"):
            summary.accepted_scores(path)


if __name__ == "__main__":
    unittest.main()
