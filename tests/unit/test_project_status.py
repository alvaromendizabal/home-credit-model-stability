"""Status cannot turn stale artifacts, denied cloud reads or stopped jobs into success."""

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from home_credit.observability.project import (
    cloud_artifacts,
    cloud_jobs,
    published_status,
    verify_cloud_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_status_separates_completed_study_from_expanded_research_gate():
    status = published_status(ROOT)
    assert status["release_complete"] and status["expanded_study_complete"]
    assert not status["feature_completion_gate_passed"]
    assert status["holdout"]["observed"] and not status["holdout"]["available_for_new_selection"]
    assert status["features"]["combined_hypotheses"] == 7125
    assert status["features"]["release_retained"] == 700
    assert status["features"]["additions_promoted_to_release"] == 0
    assert status["cloud"] == {"checked": False}
    assert all(r["has_complete_saved_outputs"] for r in status["notebooks"])
    assert len(status["notebooks"]) == 9
    rows = {r["experiment"]: r for r in status["fold_sensitivity"]}
    assert rows["wider_original"]["improved_folds"] == 2
    assert rows["wider_original"]["leave_one_fold_out_delta_min"] < 0
    assert rows["wider_original"]["leave_one_fold_out_delta_max"] > 0
    assert rows["engineered"]["leave_one_fold_out_delta_max"] < 0


def test_status_fails_before_reporting_a_corrupt_publication(tmp_path):
    with pytest.raises(FileNotFoundError):
        published_status(tmp_path)


def test_jobs_paginate_describe_and_keep_stopped_distinct_from_complete():
    client = Mock()
    processing, training = Mock(), Mock()
    client.get_paginator.side_effect = [processing, training]
    processing.paginate.return_value = [
        {"ProcessingJobSummaries": [{"ProcessingJobName": "home-credit-stopped"}]},
        {
            "ProcessingJobSummaries": [
                {"ProcessingJobName": "home-credit-running"},
                {"ProcessingJobName": "other-home-credit-copy"},
            ]
        },
    ]
    training.paginate.return_value = [
        {"TrainingJobSummaries": [{"TrainingJobName": "home-credit-trained"}]}
    ]
    start = datetime(2026, 9, 8, 1, tzinfo=UTC)
    end = datetime(2026, 9, 8, 2, tzinfo=UTC)
    client.describe_processing_job.side_effect = [
        {"ProcessingJobStatus": "Stopped", "ProcessingStartTime": start, "ProcessingEndTime": end},
        {"ProcessingJobStatus": "InProgress", "ProcessingStartTime": start},
    ]
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "TrainingStartTime": start,
        "TrainingEndTime": end,
    }
    status = cloud_jobs(client, "us-west-2")
    assert status["active_jobs"] == ["home-credit-running"]
    stopped = next(r for r in status["jobs"] if r["name"] == "home-credit-stopped")
    assert stopped["status"] == "Stopped" and stopped["elapsed_seconds"] == 3600
    assert client.describe_processing_job.call_count == 2


def test_cloud_access_failure_does_not_look_like_zero_active_jobs():
    client = Mock()
    client.get_paginator.return_value.paginate.side_effect = PermissionError("access denied")
    with pytest.raises(PermissionError):
        cloud_jobs(client, "us-west-2")


@pytest.mark.parametrize("fault", [None, "size", "truncated", "digest"])
def test_full_cloud_hash_verification_closes_streams_on_every_path(fault):
    payload = b"verified checkpoint"
    member = {
        "key": "project/checkpoint",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    body = io.BytesIO(payload[:-1] if fault == "truncated" else payload)
    if fault == "digest":
        member["sha256"] = "0" * 64
    client = Mock()
    client.get_object.return_value = {
        "Body": body,
        "ContentLength": len(payload) + (1 if fault == "size" else 0),
    }
    if fault:
        with pytest.raises(ValueError):
            verify_cloud_artifacts(client, "bucket", [member])
    else:
        receipt = verify_cloud_artifacts(client, "bucket", [member])
        assert receipt == {
            "verified_objects": 1,
            "verified_bytes": len(payload),
            "method": "full_sha256",
        }
    assert body.closed


def test_cloud_members_are_unique_and_belong_to_the_verified_project():
    members = cloud_artifacts(ROOT)
    assert len(members) == len({m["key"] for m in members})
    assert all(m["key"].startswith("home-credit-model-stability/") for m in members)
    assert any(m["key"].endswith("inference_bundle.zip") for m in members)
    assert any("feature-research/" in m["key"] for m in members)
    assert any("calibration-research/" in m["key"] for m in members)
