"""Validated raw-manifest loading and bounded-memory S3 access."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import boto3  # type: ignore[import-untyped]
from pyarrow import fs as pafs

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SSE = "AES256"


@dataclass(frozen=True, slots=True)
class S3Uri:
    """Parsed S3 URI."""

    bucket: str
    key: str

    @classmethod
    def parse(cls, value: str) -> S3Uri:
        """Parse ``s3://bucket/key`` and reject incomplete locations."""
        if not value.startswith("s3://"):
            raise ValueError(f"expected s3:// URI, received: {value}")
        remainder = value.removeprefix("s3://")
        bucket, separator, key = remainder.partition("/")
        if not separator or not bucket or not key:
            raise ValueError(f"expected s3://bucket/key, received: {value}")
        return cls(bucket=bucket, key=key)

    @property
    def arrow_path(self) -> str:
        """Return the bucket/key form expected by PyArrow's S3FileSystem."""
        return f"{self.bucket}/{self.key}"


@dataclass(frozen=True, slots=True)
class RawManifestRecord:
    """One immutable object described by the successful Kaggle-to-S3 manifest."""

    file: str
    s3_key: str
    bytes: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> RawManifestRecord:
        """Validate and construct one raw manifest record."""
        file_name = value.get("file")
        s3_key = value.get("s3_key")
        size = value.get("bytes")
        digest = value.get("sha256")

        if not isinstance(file_name, str) or not file_name:
            raise ValueError("manifest record has invalid file")
        if not isinstance(s3_key, str) or not s3_key:
            raise ValueError(f"manifest record has invalid s3_key: {file_name}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"manifest record has invalid bytes: {file_name}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"manifest record has invalid sha256: {file_name}")
        return cls(file=file_name, s3_key=s3_key, bytes=size, sha256=digest)


@dataclass(slots=True)
class S3RawStore:
    """Read-only access to one S3 bucket using the ambient SageMaker role."""

    bucket: str
    region: str
    _filesystem: Any = None
    _client: Any = None

    @property
    def filesystem(self) -> Any:
        """Create the PyArrow S3 filesystem lazily."""
        if self._filesystem is None:
            self._filesystem = pafs.S3FileSystem(region=self.region)
        return self._filesystem

    @property
    def client(self) -> Any:
        """Create the boto3 S3 client lazily."""
        if self._client is None:
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def read_bytes(self, key: str) -> bytes:
        """Read one small S3 object fully into memory."""
        with self.filesystem.open_input_file(f"{self.bucket}/{key}") as stream:
            return cast(bytes, stream.read())

    def open_input_file(self, key: str) -> Any:
        """Open one S3 object for bounded-memory streaming reads."""
        return self.filesystem.open_input_file(f"{self.bucket}/{key}")

    def verify_record(self, record: RawManifestRecord) -> int:
        """Verify size, encryption, and SHA-256 metadata against the locked manifest."""
        response = self.client.head_object(Bucket=self.bucket, Key=record.s3_key)
        current_size = int(response["ContentLength"])
        encryption = response.get("ServerSideEncryption")
        metadata = response.get("Metadata") or {}
        remote_sha256 = metadata.get("sha256")

        if current_size != record.bytes:
            raise ValueError(
                f"S3 size does not match locked manifest for {record.file}: "
                f"manifest={record.bytes} current={current_size}"
            )
        if encryption != _EXPECTED_SSE:
            raise ValueError(
                f"S3 encryption does not match ingestion contract for {record.file}: "
                f"expected={_EXPECTED_SSE} current={encryption}"
            )
        if remote_sha256 != record.sha256:
            raise ValueError(
                f"S3 SHA-256 metadata does not match locked manifest for {record.file}"
            )
        return current_size


def parse_manifest_bytes(payload: bytes) -> list[RawManifestRecord]:
    """Parse, validate, and de-duplicate a JSONL raw-data manifest."""
    records: list[RawManifestRecord] = []
    seen_files: set[str] = set()
    seen_keys: set[str] = set()

    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"manifest line {line_number} is not a JSON object")
        record = RawManifestRecord.from_mapping(cast(dict[str, Any], raw))
        if record.file in seen_files:
            raise ValueError(f"duplicate manifest file: {record.file}")
        if record.s3_key in seen_keys:
            raise ValueError(f"duplicate manifest s3_key: {record.s3_key}")
        seen_files.add(record.file)
        seen_keys.add(record.s3_key)
        records.append(record)

    if not records:
        raise ValueError("raw-data manifest is empty")
    return records


def load_s3_manifest(
    uri: str,
    *,
    region: str,
) -> tuple[list[RawManifestRecord], str, S3RawStore]:
    """Load an S3 JSONL manifest and return records, its SHA-256, and its raw store."""
    location = S3Uri.parse(uri)
    store = S3RawStore(bucket=location.bucket, region=region)
    payload = store.read_bytes(location.key)
    return parse_manifest_bytes(payload), hashlib.sha256(payload).hexdigest(), store


def parquet_records(records: Sequence[RawManifestRecord]) -> Iterator[RawManifestRecord]:
    """Yield only Parquet records in deterministic path order."""
    yield from sorted(
        (record for record in records if record.file.endswith(".parquet")),
        key=lambda record: record.file,
    )
