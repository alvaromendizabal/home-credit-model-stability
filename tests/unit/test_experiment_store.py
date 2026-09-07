"""Exercise durable state, CAS conflicts, leases, and interrupted transfers without AWS."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.observability.logging import RunLogger


class MissingKey(Exception):
    pass


class FakeS3:
    exceptions = SimpleNamespace(NoSuchKey=MissingKey)

    def __init__(self):
        self.objects = {}
        self.serial = 0
        self.fail_key = None
        self.downloads = 0
        self.corrupt_download = False
        self.bodies = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingKey(Key)
        obj = self.objects[Key]
        body = io.BytesIO(obj["Body"])
        self.bodies.append(body)
        return {**obj, "Body": body}

    def put_object(self, *, Bucket, Key, Body, Metadata, **kwargs):
        if Key == self.fail_key:
            raise OSError("injected upload interruption")
        current = self.objects.get(Key)
        if kwargs.get("IfNoneMatch") == "*" and current is not None:
            raise RuntimeError("conditional conflict")
        if "IfMatch" in kwargs and (current is None or current["ETag"] != kwargs["IfMatch"]):
            raise RuntimeError("conditional conflict")
        self.serial += 1
        self.objects[Key] = {"Body": Body, "Metadata": Metadata, "ETag": str(self.serial)}
        return {"ETag": str(self.serial)}

    def download_file(self, bucket, key, path):
        from pathlib import Path

        self.downloads += 1
        Path(path).write_bytes(b"corrupt" if self.corrupt_download else self.objects[key]["Body"])


def store(tmp_path, client=None):
    return ExperimentStore(
        client or FakeS3(), "bucket", "study", tmp_path, RunLogger("store-test", tmp_path / "logs")
    )


def state(revision=0):
    return {"schema_version": 1, "identity": {"commit": "abc"}, "revision": revision, "trials": []}


def test_failed_ledger_upload_does_not_advance_local_or_remote(tmp_path):
    s = store(tmp_path)
    s.commit(state())
    original = (tmp_path / "study.json").read_bytes()
    s.client.fail_key = "study/study.json"
    with pytest.raises(OSError):
        s.commit(state(1))
    assert (tmp_path / "study.json").read_bytes() == original
    assert s.client.objects["study/study.json"]["Body"] == original
    assert len([k for k in s.client.objects if k.startswith("study/history/")]) == 2


def test_new_machine_restores_committed_study_and_closes_body(tmp_path):
    first = store(tmp_path / "first")
    first.commit(state())
    second = store(tmp_path / "second", first.client)
    assert second.restore({"commit": "abc"}) == state()
    assert (tmp_path / "second/study.json").read_bytes() == canonical_json_bytes(state())
    assert all(body.closed for body in first.client.bodies)


def test_restore_rejects_wrong_identity_and_corruption(tmp_path):
    s = store(tmp_path)
    s.commit(state())
    with pytest.raises(ValueError, match="provenance"):
        s.restore({"commit": "other"})
    s.client.objects["study/study.json"]["Body"] += b" "
    with pytest.raises(ValueError, match="digest"):
        s.restore({"commit": "abc"})


def test_concurrent_ledger_write_cannot_overwrite_winner(tmp_path):
    first = store(tmp_path / "first")
    second = store(tmp_path / "second", first.client)
    first.commit(state())
    second.restore({"commit": "abc"})
    first.commit(state(1))
    with pytest.raises(RuntimeError, match="conditional conflict"):
        second.commit(state(2))
    assert json.loads(first.client.objects["study/study.json"]["Body"])["revision"] == 1


def test_partial_download_preserves_successful_file_and_resumes(tmp_path):
    s = store(tmp_path)
    payload = b"verified-model-predictions"
    s.put("object", payload)
    destination = tmp_path / "predictions.parquet"
    destination.write_bytes(b"previous")
    s.client.corrupt_download = True
    with pytest.raises(ValueError, match="digest"):
        s.download("object", destination, sha256_bytes(payload))
    assert destination.read_bytes() == b"previous"
    s.client.corrupt_download = False
    s.download("object", destination, sha256_bytes(payload))
    s.download("object", destination, sha256_bytes(payload))
    assert destination.read_bytes() == payload and s.client.downloads == 2


def test_lease_blocks_duplicate_launch_and_releases_cleanly(tmp_path):
    s = store(tmp_path)
    with WriterLease(s) as first:
        first.check()
        with pytest.raises(ValueError, match="another study writer"), WriterLease(s):
            pytest.fail("second writer acquired active lease")
    with WriterLease(s) as second:
        second.check()
    assert all(body.closed for body in s.client.bodies)


def test_expired_lease_is_reclaimed_and_stale_writer_is_fenced(tmp_path):
    s = store(tmp_path)
    s.put("study/writer.json", canonical_json_bytes({"owner": "old", "expires_epoch": 0}))
    with WriterLease(s) as current:
        current.check()
        old_etag = current.etag
        current._renew()
        with pytest.raises(RuntimeError, match="conditional conflict"):
            s.put("study/writer.json", b"old writer", IfMatch=old_etag)
        current.error = OSError("lost connection")
        with pytest.raises(RuntimeError, match="lease lost"):
            current.check()


def test_nonfinite_ledger_is_rejected_before_s3_commit(tmp_path):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.commit({**state(), "value": float("nan")})
    assert s.client.objects == {}
