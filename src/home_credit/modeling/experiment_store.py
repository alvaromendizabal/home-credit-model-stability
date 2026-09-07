"""Conditional S3 study commits, verified downloads, and a renewable writer lease."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

from home_credit.modeling.acceptance import require
from home_credit.modeling.checkpoints import (
    atomic_write,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from home_credit.observability.logging import RunLogger


class ExperimentStore:
    """Keep the study authoritative in S3 and mirror committed state locally.

    The caller holds a local process lock and this store's remote lease. Compare-and-
    swap writes reject concurrent changes. A failed upload never advances local state.
    """

    def __init__(
        self, client: Any, bucket: str, prefix: str, root: Path, logger: RunLogger
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.root = root
        self.logger = logger
        self.etag: str | None = None

    def read(self, key: str) -> tuple[bytes, str] | None:
        """Read an object with a closed body and validate its content digest."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        with response["Body"] as body:
            payload = cast(bytes, body.read())
        require(
            response.get("Metadata", {}).get("sha256") == sha256_bytes(payload),
            f"S3 digest mismatch: {key}",
        )
        return payload, str(response["ETag"])

    def put(self, key: str, payload: bytes, **conditions: str) -> str:
        """Commit bytes with transport checksums, encryption, and optional CAS."""
        response = self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ServerSideEncryption="AES256",
            Metadata={"sha256": sha256_bytes(payload)},
            ChecksumAlgorithm="SHA256",
            **conditions,
        )
        return str(response["ETag"])

    def restore(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        """Restore the authoritative ledger; never overwrite a different study."""
        remote = self.read(f"{self.prefix}/study.json")
        if remote is None:
            require(
                not (self.root / "study.json").exists(),
                "local study exists but its durable S3 ledger is missing",
            )
            return None
        payload, self.etag = remote
        state = cast(dict[str, Any], json.loads(payload))
        require(state["identity"] == identity, "study provenance changed; resume original commit")
        require(state["schema_version"] == 1, "unsupported study ledger")
        atomic_write(self.root / "study.json", payload)
        self.logger.event("study_restored", trials=len(state["trials"]), revision=state["revision"])
        return state

    def commit(self, state: dict[str, Any]) -> None:
        """Upload an immutable history snapshot, then conditionally advance the ledger."""
        payload = canonical_json_bytes(state)
        # JSON must be portable to reviewers and Optuna replay; reject NaN/Infinity.
        json.dumps(state, allow_nan=False)
        digest = sha256_bytes(payload)
        self.put(f"{self.prefix}/history/{digest}.json", payload)
        condition = {"IfNoneMatch": "*"} if self.etag is None else {"IfMatch": self.etag}
        self.etag = self.put(f"{self.prefix}/study.json", payload, **condition)
        atomic_write(self.root / "study.json", payload)
        self.logger.event("study_committed", revision=state["revision"], sha256=digest)

    def download(self, key: str, destination: Path, digest: str) -> None:
        """Reuse verified files and atomically replace incomplete or corrupt downloads."""
        if destination.is_file() and sha256_file(destination) == digest:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        self.client.download_file(self.bucket, key, str(temporary))
        require(sha256_file(temporary) == digest, f"download digest mismatch: {destination.name}")
        temporary.replace(destination)
        self.logger.event(
            "study_input_restored", file=destination.name, bytes=destination.stat().st_size
        )

    def publish(self, path: Path, category: str = "reports") -> str:
        """Publish a small report/log as a content-addressed object."""
        key = f"{self.prefix}/{category}/{sha256_file(path)}/{path.name}"
        self.client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {"sha256": sha256_file(path)},
                "ChecksumAlgorithm": "SHA256",
            },
        )
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        require(
            head["ContentLength"] == path.stat().st_size
            and head.get("Metadata", {}).get("sha256") == sha256_file(path),
            "published artifact verification failed",
        )
        self.logger.event("study_artifact_published", file=path.name, s3_key=key)
        return key


class WriterLease:
    """Prevent two machines from training or publishing the same study concurrently."""

    def __init__(self, store: ExperimentStore, *, ttl: float = 300, interval: float = 30) -> None:
        require(0 < interval < ttl / 2, "invalid lease renewal interval")
        self.store = store
        self.ttl = ttl
        self.interval = interval
        self.owner = uuid.uuid4().hex
        self.key = f"{store.prefix}/writer.json"
        self.etag: str | None = None
        self.error: Exception | None = None
        self.expires = 0.0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> WriterLease:
        previous = self.store.read(self.key)
        if previous is not None:
            payload, self.etag = previous
            remaining = float(json.loads(payload)["expires_epoch"]) - time.time()
            require(
                remaining <= 0,
                f"another study writer holds the lease for up to {math.ceil(remaining)}s; "
                "do not start a second worker",
            )
        self._renew()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        self.store.logger.event("study_writer_acquired", owner=self.owner, lease_seconds=self.ttl)
        return self

    def _renew(self, *, release: bool = False) -> None:
        expires = 0.0 if release else time.time() + self.ttl
        condition = {"IfNoneMatch": "*"} if self.etag is None else {"IfMatch": self.etag}
        self.etag = self.store.put(
            self.key,
            canonical_json_bytes(
                {
                    "owner": self.owner,
                    "expires_epoch": expires,
                }
            ),
            **condition,
        )
        self.expires = expires

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                self._renew()
            except Exception as exc:
                self.error = exc
                self.store.logger.event("study_writer_lease_lost", error_type=type(exc).__name__)
                return

    def check(self) -> None:
        """Fail before computation/publication if this writer lost its lease."""
        if self.error is not None:
            raise RuntimeError(
                "study writer lease lost; resume after connectivity returns"
            ) from self.error
        require(time.time() < self.expires, "study writer lease expired")

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
        if self.error is None and self.thread is not None and not self.thread.is_alive():
            self._renew(release=True)
