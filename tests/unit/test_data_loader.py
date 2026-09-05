from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from home_credit.data.loader import RawManifestRecord, S3RawStore, S3Uri, parse_manifest_bytes


def _record(file_name: str, key: str) -> dict[str, object]:
    return {
        "file": file_name,
        "s3_key": key,
        "bytes": 10,
        "sha256": hashlib.sha256(file_name.encode()).hexdigest(),
    }


class _FakeS3Client:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def head_object(self, **_: str) -> dict[str, Any]:
        return self.response


def test_s3_uri_parser() -> None:
    parsed = S3Uri.parse("s3://bucket/path/to/manifest.jsonl")
    assert parsed.bucket == "bucket"
    assert parsed.key == "path/to/manifest.jsonl"
    assert parsed.arrow_path == "bucket/path/to/manifest.jsonl"


def test_manifest_parser_is_deterministic_and_validated() -> None:
    records = [
        _record("parquet_files/train/train_base.parquet", "raw/train_base.parquet"),
        _record("sample_submission.csv", "raw/sample_submission.csv"),
    ]
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    parsed = parse_manifest_bytes(payload)
    assert [record.file for record in parsed] == [record["file"] for record in records]


def test_duplicate_manifest_file_is_rejected() -> None:
    record = _record("parquet_files/train/train_base.parquet", "raw/train_base.parquet")
    payload = (json.dumps(record) + "\n" + json.dumps(record) + "\n").encode()
    with pytest.raises(ValueError, match="duplicate manifest file"):
        parse_manifest_bytes(payload)


def test_s3_record_verification_accepts_locked_metadata() -> None:
    record = RawManifestRecord(
        file="parquet_files/train/train_base.parquet",
        s3_key="raw/train_base.parquet",
        bytes=10,
        sha256="a" * 64,
    )
    store = S3RawStore(bucket="bucket", region="us-west-2")
    store._client = _FakeS3Client(
        {
            "ContentLength": 10,
            "ServerSideEncryption": "AES256",
            "Metadata": {"sha256": "a" * 64},
        }
    )
    assert store.verify_record(record) == 10


def test_s3_record_verification_rejects_sha256_mismatch() -> None:
    record = RawManifestRecord(
        file="parquet_files/train/train_base.parquet",
        s3_key="raw/train_base.parquet",
        bytes=10,
        sha256="a" * 64,
    )
    store = S3RawStore(bucket="bucket", region="us-west-2")
    store._client = _FakeS3Client(
        {
            "ContentLength": 10,
            "ServerSideEncryption": "AES256",
            "Metadata": {"sha256": "b" * 64},
        }
    )
    with pytest.raises(ValueError, match="SHA-256"):
        store.verify_record(record)
