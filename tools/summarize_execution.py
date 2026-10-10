"""Verify a bounded owner return and publish only an allowlisted execution receipt.

Regenerate: python tools/summarize_execution.py --archive /path/to/owner-return.zip
Public check: python tools/summarize_execution.py --check
Archive contents are read in memory; no code is run and no members are extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/latest_execution/summary.json"
SCORE_SOURCE = ROOT / "reports/post_release_frontier/october_2026_continuation_summary.json"
ARCHIVE_SHA = "12638274bd50bbc7d40d15e23d5a3a2cf3c25af4f7499b210e2c5b8be1dc7071"
MANIFEST_SHA = "913a6d37d185795d45125a476162dc16889c97eeca2fb0329ba7246c1c126e12"
EVIDENCE_SHA = "3f8eacc53aa0bae7817bb74dbdcdb310bbe2e443ac6f1cc90804cf148dff8905"
SUMMARY_SHA = "952f30497f7653558bee307a8387b73d367275a0fe70ea144eed82c84fc3fb4e"
POINTER_SHA = "20fec7aadee159358d2aeedb9ec162beea66ea10a59feb242d881b58e6448e8f"
LEDGER_SHA = "20484fcf24088b916e08c22ef747fe53052196403c9f1f1e31dde4460ada368e"
ARCHIVE_LIMIT = 2 * 1024 * 1024
MEMBER_LIMIT = 1024 * 1024
TOTAL_LIMIT = 2 * 1024 * 1024
RUN_ID = "20261010T022535072312Z-ced3dd22"
KEEP = {
    "stage_results_summary.json",
    "execution_ledger.json",
    "complete_return_evidence_remote.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("EXECUTION_SUMMARY: " + message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(value, dict), "expected a JSON object")
    return dict(value)


def safe_path(name: str) -> None:
    require(isinstance(name, str) and bool(name), "invalid member path")
    require(not PurePosixPath(name).is_absolute(), "unsafe member path")
    require(not any(char in name for char in ("\\", "\x00", ":")), "unsafe member path")
    require(all(part not in ("", ".", "..") for part in name.split("/")), "unsafe member path")


def verify_member(raw: bytes, receipt: dict[str, Any]) -> None:
    require(type(receipt["bytes"]) is int, "invalid member size")
    require(len(raw) == receipt["bytes"], "member size mismatch")
    require(sha256(raw) == receipt["sha256"], "member SHA-256 mismatch")


def inspect_archive(
    path: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Verify complete local coverage while discarding private configuration bytes."""
    require(path.stat().st_size <= ARCHIVE_LIMIT, "archive size limit exceeded")
    raw_archive = path.read_bytes()
    require(sha256(raw_archive) == expected_sha, "archive SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        members = archive.infolist()
        require(len(members) == 2, "unexpected ZIP member count")
        names = [member.filename for member in members]
        require(len({name.casefold() for name in names}) == len(names), "duplicate ZIP member")
        for member in members:
            safe_path(member.filename)
            require(member.orig_filename == member.filename, "unsafe original member path")
            require(not member.is_dir(), "directory ZIP member")
            require(not member.flag_bits & 1, "encrypted ZIP member")
            kind = stat.S_IFMT(member.external_attr >> 16)
            require(kind in (0, stat.S_IFREG), "linked or special ZIP member")
            require(member.file_size <= MEMBER_LIMIT, "ZIP member size limit exceeded")
        require(set(names) == {"manifest.json", "evidence.tar.xz"}, "missing manifest or evidence")
        manifest_raw = archive.read("manifest.json")
        evidence_raw = archive.read("evidence.tar.xz")
    manifest = read_json(manifest_raw)
    outer = manifest["return_artifacts"]
    require(isinstance(outer, list) and len(outer) == 1, "return coverage mismatch")
    require(outer[0]["path"] == "evidence.tar.xz", "unexpected return artifact")
    verify_member(evidence_raw, outer[0])
    receipts: dict[str, dict[str, Any]] = {}
    for item in manifest["evidence_artifacts"]:
        name = item["path"]
        safe_path(name)
        require(name not in receipts, "duplicate evidence receipt")
        receipts[name] = item
    require(set(receipts) == KEEP | {"config.json"}, "evidence coverage mismatch")
    kept: dict[str, bytes] = {}
    seen: set[str] = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(evidence_raw), mode="r|xz") as evidence:
        for tar_member in evidence:
            safe_path(tar_member.name)
            require(tar_member.name not in seen, "duplicate tar member")
            require(tar_member.isfile(), "linked or special tar member")
            require(tar_member.name in receipts, "unmanifested tar member")
            require(0 <= tar_member.size <= MEMBER_LIMIT, "tar member size limit exceeded")
            total += tar_member.size
            require(total <= TOTAL_LIMIT, "total decompressed size limit exceeded")
            stream = evidence.extractfile(tar_member)
            require(stream is not None, "unreadable tar member")
            assert stream is not None
            with stream:
                raw = stream.read(MEMBER_LIMIT + 1)
            require(len(raw) <= MEMBER_LIMIT, "decompressed member size limit exceeded")
            verify_member(raw, receipts[tar_member.name])
            seen.add(tar_member.name)
            if tar_member.name in KEEP:
                kept[tar_member.name] = raw
    require(seen == set(receipts), "missing tar member")
    provenance = {
        "source_zip_sha256": sha256(raw_archive),
        "manifest_sha256": sha256(manifest_raw),
        "evidence_tar_sha256": sha256(evidence_raw),
        "stage_summary_sha256": sha256(kept["stage_results_summary.json"]),
        "remote_pointer_sha256": sha256(kept["complete_return_evidence_remote.json"]),
        "execution_ledger_sha256": sha256(kept["execution_ledger.json"]),
        "verified_zip_members": len(names),
        "verified_inner_members": len(seen),
    }
    return manifest, kept, provenance


def accepted_scores(path: Path = SCORE_SOURCE) -> dict[str, Any]:
    result = read_json(path.read_bytes())["externally_measured_post_release_artifact"]
    require(result["kaggle_public"] == 0.56035, "existing public score mismatch")
    require(result["kaggle_private"] == 0.47652, "existing private score mismatch")
    return {
        "public": result["kaggle_public"],
        "private": result["kaggle_private"],
        "source": "reports/post_release_frontier/october_2026_continuation_summary.json",
        "independently_reverified_by_this_return": False,
    }


def summarize_archive(path: Path, expected_sha: str = ARCHIVE_SHA) -> dict[str, Any]:
    manifest, kept, provenance = inspect_archive(path, expected_sha)
    stages = read_json(kept["stage_results_summary.json"])["stages"]
    pointer = read_json(kept["complete_return_evidence_remote.json"])
    ledger = read_json(kept["execution_ledger.json"])
    require(manifest["run_id"] == RUN_ID and manifest["status"] == "STOPPED", "unexpected run")
    require(
        manifest["error"] == "MEMORY: configured available RAM or projected RSS guard",
        "unexpected stop reason",
    )
    for key in ("new_completed_fits", "new_training_attempts", "reused_fits", "scientific_fits"):
        require(type(manifest[key]) is int and manifest[key] == 0, "unexpected model activity")
    for key in ("submission_ready", "leaderboard_score_changed"):
        require(manifest[key] is False, "unexpected scientific promotion")
    require(manifest["submission_receipt"] is None, "unexpected submission receipt")
    require(manifest["scientific_outcome"] is None, "unexpected scientific outcome")
    require(manifest["scientific_attempts"] == [], "unexpected scientific attempt")
    require(
        manifest["completed_tasks"] == manifest["completed_stages"] == len(stages) == 13,
        "completed task count mismatch",
    )
    require(
        manifest["total_tasks"] == manifest["total_stages"] == 42
        and manifest["remaining_tasks"] == 29,
        "total task count mismatch",
    )
    require(set(stages) == {str(index) for index in range(1, 14)}, "stage identity mismatch")
    require(all(stage["status"] == "PASS" for stage in stages.values()), "incomplete saved stage")
    current = ledger["runs"][-1]
    require(
        current["run_id"] == RUN_ID
        and current["status"] == "STOPPED"
        and current["completed_tasks"] == 13
        and current["new_completed_fits"] == 0,
        "ledger disagreement",
    )
    require(manifest["ledger_totals"] == ledger["totals"], "ledger totals disagreement")
    tests = manifest["selftests"]
    require(
        tests["status"] == "PASS"
        and tests["tests"] == 661
        and tests["failure_count"] == tests["error_count"] == tests["skipped"] == 0,
        "selftest receipt disagreement",
    )
    require(
        pointer["complete_members_remote"] is True
        and len(pointer["files"]) == 9
        and manifest["full_evidence_remote"] is True,
        "remote evidence scope disagreement",
    )
    parity = stages["2"]["parity"]
    require(
        parity["status"] == "PASS"
        and parity["rows"] == 10
        and parity["hidden_test_readiness_established"] is False,
        "parity scope disagreement",
    )
    return {
        "schema_version": 1,
        "scope": "latest_inspected_owner_return",
        "version": "E97",
        "run_id": manifest["run_id"],
        "recorded_ended_utc": None,
        "status": manifest["status"],
        "stop_reason": "configured_available_memory_or_projected_rss_guard",
        "progress": {
            "completed_tasks": manifest["completed_tasks"],
            "total_tasks": manifest["total_tasks"],
            "remaining_tasks": manifest["remaining_tasks"],
        },
        "activity": {
            "new_training_attempts": manifest["new_training_attempts"],
            "new_completed_fits": manifest["new_completed_fits"],
            "reused_fits": manifest["reused_fits"],
            "new_submission_receipt": False,
        },
        "scientific_outcome": manifest["scientific_outcome"],
        "submission_ready": manifest["submission_ready"],
        "leaderboard_score_changed": manifest["leaderboard_score_changed"],
        "existing_external_scores": accepted_scores(),
        "reported_selftests": {
            "status": tests["status"],
            "tests": tests["tests"],
            "rerun_here": False,
        },
        "reported_incumbent_parity": {
            "status": parity["status"],
            "rows": parity["rows"],
            "hidden_test_readiness_established": parity["hidden_test_readiness_established"],
            "independently_replayed_here": False,
        },
        "provenance": provenance,
        "evidence_scope": {
            "local_archive_integrity_verified": True,
            "remote_files_referenced": len(pointer["files"]),
            "remote_files_downloaded_by_reproducer": 0,
            "private_predictions_independently_replayed": False,
            "current_live_execution_state_verified": False,
        },
        "limitations": [
            "This return includes a stage overview; fuller evidence is referenced remotely.",
            "Included bytes were verified. Reported execution was not independently replayed.",
            "No end timestamp is recorded. The run ID is not an end time.",
            "External scores retain their earlier evidence source. This run produced no new score.",
            "This is the latest inspected owner return, not a live or global cloud inventory.",
        ],
    }


def expected_snapshot() -> dict[str, Any]:
    """Exact public contract, independent of private archive availability in CI."""
    return {
        "schema_version": 1,
        "scope": "latest_inspected_owner_return",
        "version": "E97",
        "run_id": RUN_ID,
        "recorded_ended_utc": None,
        "status": "STOPPED",
        "stop_reason": "configured_available_memory_or_projected_rss_guard",
        "progress": {"completed_tasks": 13, "total_tasks": 42, "remaining_tasks": 29},
        "activity": {
            "new_training_attempts": 0,
            "new_completed_fits": 0,
            "reused_fits": 0,
            "new_submission_receipt": False,
        },
        "scientific_outcome": None,
        "submission_ready": False,
        "leaderboard_score_changed": False,
        "existing_external_scores": accepted_scores(),
        "reported_selftests": {"status": "PASS", "tests": 661, "rerun_here": False},
        "reported_incumbent_parity": {
            "status": "PASS",
            "rows": 10,
            "hidden_test_readiness_established": False,
            "independently_replayed_here": False,
        },
        "provenance": {
            "source_zip_sha256": ARCHIVE_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "evidence_tar_sha256": EVIDENCE_SHA,
            "stage_summary_sha256": SUMMARY_SHA,
            "remote_pointer_sha256": POINTER_SHA,
            "execution_ledger_sha256": LEDGER_SHA,
            "verified_zip_members": 2,
            "verified_inner_members": 4,
        },
        "evidence_scope": {
            "local_archive_integrity_verified": True,
            "remote_files_referenced": 9,
            "remote_files_downloaded_by_reproducer": 0,
            "private_predictions_independently_replayed": False,
            "current_live_execution_state_verified": False,
        },
        "limitations": [
            "This return includes a stage overview; fuller evidence is referenced remotely.",
            "Included bytes were verified. Reported execution was not independently replayed.",
            "No end timestamp is recorded. The run ID is not an end time.",
            "External scores retain their earlier evidence source. This run produced no new score.",
            "This is the latest inspected owner return, not a live or global cloud inventory.",
        ],
    }


def validate_snapshot(value: dict[str, Any]) -> None:
    require(
        json.dumps(value, sort_keys=True, allow_nan=False)
        == json.dumps(expected_snapshot(), sort_keys=True, allow_nan=False),
        "public snapshot contract mismatch",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--archive", type=Path)
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.check:
        validate_snapshot(read_json(args.output.read_bytes()))
        print("PASS: allowlisted E97 receipt, provenance and existing score consistency")
    else:
        result = summarize_archive(args.archive)
        validate_snapshot(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("PASS: verified local return; published only the aggregate execution receipt")


if __name__ == "__main__":
    main()
