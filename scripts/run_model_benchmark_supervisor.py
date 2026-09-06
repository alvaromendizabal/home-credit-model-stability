#!/usr/bin/env python3
"""Supervise durable, process-isolated Home Credit benchmark execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import boto3
import psutil
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from home_credit.modeling.checkpoints import (
    CheckpointManifest,
    CheckpointPointer,
    benchmark_manifest_key,
    build_checkpoint_manifest,
    checkpoint_manifest_key,
    derive_run_key,
    latest_pointer_key,
    load_manifest_bytes,
    load_pointer_bytes,
    manifest_bytes,
    parse_s3_uri,
    pointer_bytes,
    sha256_bytes,
    sha256_file,
    validate_benchmark_state,
    verify_checkpoint_manifest,
)
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.runner import _load_protocol, _protocol_folds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Home Credit model-fold per fresh Python process while "
            "committing verified checkpoints to content-addressed S3 storage."
        )
    )
    parser.add_argument("--feature-s3-uri", required=True)
    parser.add_argument("--expected-feature-manifest-sha256", required=True)
    parser.add_argument(
        "--validation-protocol",
        type=Path,
        default=Path("configs/validation_protocol.json"),
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_benchmark.json"),
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--checkpoint-s3-uri", required=True)
    parser.add_argument("--final-s3-uri", required=True)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("/tmp/home-credit-features"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/home-credit-model-benchmark"),
    )
    parser.add_argument(
        "--status-log",
        type=Path,
        default=Path("logs/model-benchmark-supervisor.jsonl"),
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--child-heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-processes", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class SupervisorLog:
    """Tiny durable JSONL log written under persistent repository storage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "timestamp": _utc_now(),
            "event": name,
            **fields,
        }
        line = json.dumps(payload, sort_keys=True, default=str)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class DurableStore:
    """S3 content-addressed checkpoint store with manifest-last commits."""

    def __init__(
        self,
        *,
        checkpoint_s3_uri: str,
        final_s3_uri: str,
        region: str,
        log: SupervisorLog,
    ) -> None:
        self.bucket, self.prefix = parse_s3_uri(checkpoint_s3_uri)
        final_bucket, self.final_prefix = parse_s3_uri(final_s3_uri)
        if final_bucket != self.bucket:
            raise ValueError("checkpoint and final S3 URIs must use the same bucket")
        self.s3 = boto3.client("s3", region_name=region)
        self.log = log
        self.transfer = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )

    def restore_latest(
        self,
        output_root: Path,
        *,
        run_key: str,
        git_commit: str,
        feature_manifest_sha256: str,
        validation_protocol_sha256: str,
        benchmark_config_sha256: str,
        smoke: bool,
    ) -> int:
        pointer_key = latest_pointer_key(self.prefix)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=pointer_key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                local_count = validate_benchmark_state(
                    output_root,
                    git_commit=git_commit,
                    feature_manifest_sha256=feature_manifest_sha256,
                    validation_protocol_sha256=validation_protocol_sha256,
                    benchmark_config_sha256=benchmark_config_sha256,
                    smoke=smoke,
                )
                self.log.event(
                    "checkpoint_restore_empty", run_key=run_key, local_model_folds=local_count
                )
                return local_count
            raise

        with response["Body"] as body:
            raw_pointer = cast(bytes, body.read())
        pointer_sha = sha256_bytes(raw_pointer)
        if response.get("Metadata", {}).get("sha256") != pointer_sha:
            raise RuntimeError("S3 checkpoint pointer SHA-256 metadata mismatch")
        if response.get("ServerSideEncryption") != "AES256":
            raise RuntimeError("S3 checkpoint pointer encryption mismatch")
        pointer = load_pointer_bytes(raw_pointer)
        if pointer.run_key != run_key:
            raise RuntimeError("S3 checkpoint pointer run_key mismatch")

        response = self.s3.get_object(Bucket=self.bucket, Key=pointer.manifest_key)
        with response["Body"] as body:
            raw_manifest = cast(bytes, body.read())
        manifest_sha = sha256_bytes(raw_manifest)
        if manifest_sha != pointer.manifest_sha256:
            raise RuntimeError("S3 checkpoint manifest SHA-256 mismatch")
        if response.get("Metadata", {}).get("sha256") != manifest_sha:
            raise RuntimeError("S3 checkpoint manifest SHA-256 metadata mismatch")
        if response.get("ServerSideEncryption") != "AES256":
            raise RuntimeError("S3 checkpoint manifest encryption mismatch")

        manifest = load_manifest_bytes(raw_manifest)
        _validate_manifest_identity(
            manifest,
            run_key=run_key,
            git_commit=git_commit,
            feature_manifest_sha256=feature_manifest_sha256,
            validation_protocol_sha256=validation_protocol_sha256,
            benchmark_config_sha256=benchmark_config_sha256,
            smoke=smoke,
        )

        local_count = validate_benchmark_state(
            output_root,
            git_commit=git_commit,
            feature_manifest_sha256=feature_manifest_sha256,
            validation_protocol_sha256=validation_protocol_sha256,
            benchmark_config_sha256=benchmark_config_sha256,
            smoke=smoke,
        )
        if local_count > manifest.completed_model_folds:
            self.log.event("newer_local_checkpoints_preserved", completed_model_folds=local_count)
            return local_count
        output_root.mkdir(parents=True, exist_ok=True)

        total = len(manifest.files)
        # Publish state last, so an interrupted restore never advertises missing members.
        members = sorted(manifest.files, key=lambda member: member.path == "benchmark_state.json")
        for index, member in enumerate(members, start=1):
            destination = output_root / member.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and sha256_file(destination) == member.sha256:
                continue
            temporary = destination.with_suffix(destination.suffix + ".download")
            self.s3.download_file(
                self.bucket,
                member.object_key,
                str(temporary),
                Config=self.transfer,
            )
            if temporary.stat().st_size != member.bytes:
                raise RuntimeError(f"restored checkpoint size mismatch: {member.path}")
            if sha256_file(temporary) != member.sha256:
                raise RuntimeError(f"restored checkpoint SHA mismatch: {member.path}")
            os.replace(temporary, destination)
            if index == total or index % 10 == 0:
                self.log.event("checkpoint_restore_progress", restored=index, total=total)

        verify_checkpoint_manifest(output_root, manifest)
        self.log.event(
            "checkpoint_restore_completed",
            sequence=manifest.sequence,
            completed_model_folds=manifest.completed_model_folds,
            files=total,
        )
        return manifest.completed_model_folds

    def commit_checkpoint(
        self,
        output_root: Path,
        *,
        run_key: str,
        sequence: int,
        completed_model_folds: int,
        git_commit: str,
        feature_manifest_sha256: str,
        validation_protocol_sha256: str,
        benchmark_config_sha256: str,
        smoke: bool,
    ) -> str:
        manifest = build_checkpoint_manifest(
            output_root,
            run_key=run_key,
            prefix=self.prefix,
            sequence=sequence,
            completed_model_folds=completed_model_folds,
            git_commit=git_commit,
            feature_manifest_sha256=feature_manifest_sha256,
            validation_protocol_sha256=validation_protocol_sha256,
            benchmark_config_sha256=benchmark_config_sha256,
            smoke=smoke,
        )
        verify_checkpoint_manifest(output_root, manifest)

        total = len(manifest.files)
        for index, member in enumerate(manifest.files, start=1):
            self._put_immutable_file(
                output_root / member.path,
                key=member.object_key,
                sha256=member.sha256,
            )
            if index == total or index % 10 == 0:
                self.log.event("checkpoint_upload_progress", uploaded=index, total=total)

        raw_manifest = manifest_bytes(manifest)
        manifest_sha = sha256_bytes(raw_manifest)
        manifest_key = checkpoint_manifest_key(
            self.prefix,
            sequence=sequence,
            manifest_sha256=manifest_sha,
        )
        self._put_immutable_bytes(
            raw_manifest,
            key=manifest_key,
            sha256=manifest_sha,
        )

        pointer = CheckpointPointer(
            schema_version=1,
            run_key=run_key,
            sequence=sequence,
            manifest_key=manifest_key,
            manifest_sha256=manifest_sha,
        )
        raw_pointer = pointer_bytes(pointer)
        pointer_key = latest_pointer_key(self.prefix)
        pointer_sha = sha256_bytes(raw_pointer)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=pointer_key,
            Body=raw_pointer,
            ServerSideEncryption="AES256",
            Metadata={"sha256": pointer_sha},
            ContentType="application/json",
        )
        pointer_head = self.s3.head_object(Bucket=self.bucket, Key=pointer_key)
        if (
            int(pointer_head["ContentLength"]) != len(raw_pointer)
            or pointer_head.get("Metadata", {}).get("sha256") != pointer_sha
            or pointer_head.get("ServerSideEncryption") != "AES256"
        ):
            raise RuntimeError("S3 latest checkpoint pointer verification failed")
        self.log.event(
            "checkpoint_commit_completed",
            sequence=sequence,
            completed_model_folds=completed_model_folds,
            manifest_sha256=manifest_sha,
        )
        return manifest_sha

    def publish_final(
        self,
        output_root: Path,
        *,
        run_key: str,
        completed_model_folds: int,
        git_commit: str,
        feature_manifest_sha256: str,
        validation_protocol_sha256: str,
        benchmark_config_sha256: str,
        smoke: bool,
    ) -> str:
        summary = output_root / "benchmark_summary.json"
        if not summary.is_file():
            raise RuntimeError("cannot publish final benchmark without summary")
        summary_sha = sha256_file(summary)
        final_prefix = f"{self.final_prefix.rstrip('/')}/{summary_sha}"

        manifest = build_checkpoint_manifest(
            output_root,
            run_key=run_key,
            prefix=final_prefix,
            sequence=completed_model_folds,
            completed_model_folds=completed_model_folds,
            git_commit=git_commit,
            feature_manifest_sha256=feature_manifest_sha256,
            validation_protocol_sha256=validation_protocol_sha256,
            benchmark_config_sha256=benchmark_config_sha256,
            smoke=smoke,
        )
        verify_checkpoint_manifest(output_root, manifest)

        for member in manifest.files:
            self._put_immutable_file(
                output_root / member.path,
                key=member.object_key,
                sha256=member.sha256,
            )

        raw_manifest = manifest_bytes(manifest)
        manifest_sha = sha256_bytes(raw_manifest)
        key = benchmark_manifest_key(
            final_prefix,
            manifest_sha256=manifest_sha,
        )
        self._put_immutable_bytes(raw_manifest, key=key, sha256=manifest_sha)
        self.log.event(
            "final_benchmark_published",
            summary_sha256=summary_sha,
            manifest_sha256=manifest_sha,
            s3_prefix=f"s3://{self.bucket}/{final_prefix}/",
        )
        return f"s3://{self.bucket}/{final_prefix}/"

    def _put_immutable_file(self, path: Path, *, key: str, sha256: str) -> None:
        if self._object_matches(key=key, sha256=sha256, size=path.stat().st_size):
            return
        self.s3.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {"sha256": sha256},
            },
            Config=self.transfer,
        )
        if not self._object_matches(key=key, sha256=sha256, size=path.stat().st_size):
            raise RuntimeError(f"S3 upload verification failed: {key}")

    def _put_immutable_bytes(self, payload: bytes, *, key: str, sha256: str) -> None:
        if self._object_matches(key=key, sha256=sha256, size=len(payload)):
            return
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ServerSideEncryption="AES256",
            Metadata={"sha256": sha256},
            ContentType="application/json",
        )
        if not self._object_matches(key=key, sha256=sha256, size=len(payload)):
            raise RuntimeError(f"S3 upload verification failed: {key}")

    def _object_matches(self, *, key: str, sha256: str, size: int) -> bool:
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

        observed_size = int(response["ContentLength"])
        observed_sha = response.get("Metadata", {}).get("sha256")
        observed_sse = response.get("ServerSideEncryption")
        if observed_size != size or observed_sha != sha256 or observed_sse != "AES256":
            raise RuntimeError(
                f"immutable S3 object collision or metadata mismatch: s3://{self.bucket}/{key}"
            )
        return True


def _validate_manifest_identity(
    manifest: CheckpointManifest,
    *,
    run_key: str,
    git_commit: str,
    feature_manifest_sha256: str,
    validation_protocol_sha256: str,
    benchmark_config_sha256: str,
    smoke: bool,
) -> None:
    expected = {
        "run_key": run_key,
        "git_commit": git_commit,
        "feature_manifest_sha256": feature_manifest_sha256,
        "validation_protocol_sha256": validation_protocol_sha256,
        "benchmark_config_sha256": benchmark_config_sha256,
        "smoke": smoke,
    }
    observed: dict[str, object] = {
        "run_key": manifest.run_key,
        "git_commit": manifest.git_commit,
        "feature_manifest_sha256": manifest.feature_manifest_sha256,
        "validation_protocol_sha256": manifest.validation_protocol_sha256,
        "benchmark_config_sha256": manifest.benchmark_config_sha256,
        "smoke": manifest.smoke,
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"checkpoint manifest provenance mismatch: {key}")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sync_feature_snapshot(
    *,
    s3_uri: str,
    destination: Path,
    expected_manifest_sha256: str,
    region: str,
    log: SupervisorLog,
) -> None:
    manifest = destination / "feature_manifest.json"
    known_hashes: dict[str, str] = {"feature_manifest.json": expected_manifest_sha256}
    if manifest.is_file() and sha256_file(manifest) == expected_manifest_sha256:
        payload = json.loads(manifest.read_bytes())
        known_hashes.update(
            {
                f"blocks/{b['split']}/{b['family']}_depth{b['depth']}.parquet": b["output_sha256"]
                for b in payload["blocks"]
            }
        )
        if all(
            (destination / name).is_file() and sha256_file(destination / name) == digest
            for name, digest in known_hashes.items()
        ):
            log.event("feature_cache_reused", path=destination)
            return

    destination.mkdir(parents=True, exist_ok=True)

    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"):
        for item in page.get("Contents", []):
            key = cast(str, item["Key"])
            if not key.endswith("/"):
                objects.append(key)

    if not objects:
        raise RuntimeError(f"feature snapshot is empty: {s3_uri}")

    for index, key in enumerate(sorted(objects), start=1):
        relative = key[len(prefix.rstrip("/")) + 1 :]
        destination_path = (destination / relative).resolve()
        if not destination_path.is_relative_to(destination.resolve()):
            raise ValueError("feature snapshot path escaped cache")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            relative in known_hashes
            and destination_path.is_file()
            and sha256_file(destination_path) == known_hashes[relative]
        ):
            continue
        temporary = destination_path.with_suffix(destination_path.suffix + ".download")
        s3.download_file(bucket, key, str(temporary))
        os.replace(temporary, destination_path)
        if index == len(objects) or index % 10 == 0:
            log.event("feature_cache_restore_progress", restored=index, total=len(objects))

    if not manifest.is_file():
        raise RuntimeError("restored feature_manifest.json is missing")
    if sha256_file(manifest) != expected_manifest_sha256:
        raise RuntimeError("restored feature manifest SHA-256 mismatch")
    log.event("feature_cache_restore_completed", files=len(objects))


def _child_command(
    *,
    feature_dir: Path,
    feature_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
    config_path: Path,
    config_sha256: str,
    output_dir: Path,
    logs_dir: Path,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_model_benchmark.py",
        "--feature-dir",
        str(feature_dir),
        "--expected-feature-manifest-sha256",
        feature_sha256,
        "--validation-protocol",
        str(protocol_path),
        "--expected-protocol-sha256",
        protocol_sha256,
        "--config",
        str(config_path),
        "--expected-config-sha256",
        config_sha256,
        "--output-dir",
        str(output_dir),
        "--logs-dir",
        str(logs_dir),
        "--max-new-checkpoints",
        "1",
    ]
    if smoke:
        command.append("--smoke")
    return command


def _run_child(
    command: list[str],
    *,
    heartbeat_seconds: float,
    log: SupervisorLog,
) -> int:
    if heartbeat_seconds <= 0:
        raise ValueError("child heartbeat seconds must be positive")
    started = time.monotonic()
    process = subprocess.Popen(command)
    process_stats = psutil.Process(process.pid)
    process_stats.cpu_percent(interval=None)

    while True:
        try:
            return_code = process.wait(timeout=heartbeat_seconds)
            log.event(
                "child_process_completed",
                pid=process.pid,
                return_code=return_code,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            return return_code
        except subprocess.TimeoutExpired:
            try:
                rss_mb = round(process_stats.memory_info().rss / (1024 * 1024), 1)
                cpu_percent = round(process_stats.cpu_percent(interval=None), 1)
            except psutil.Error:
                rss_mb = -1.0
                cpu_percent = -1.0
            log.event(
                "child_process_heartbeat",
                pid=process.pid,
                elapsed_seconds=round(time.monotonic() - started, 1),
                rss_mb=rss_mb,
                cpu_percent=cpu_percent,
            )


def expected_model_folds(
    config_path: Path, config_sha: str, protocol_path: Path, protocol_sha: str, *, smoke: bool
) -> int:
    """Derive progress and completion from validated inputs, never a family constant."""
    config, actual = BenchmarkConfig.load(config_path)
    if actual != config_sha:
        raise ValueError("benchmark config hash mismatch")
    protocol = _load_protocol(protocol_path, expected_sha256=protocol_sha, config=config)
    return len(config.enabled_model_names) * (1 if smoke else len(_protocol_folds(protocol)))


def main() -> int:
    started = time.monotonic()
    args = _parser().parse_args()
    expected_total = expected_model_folds(
        args.config,
        args.expected_config_sha256,
        args.validation_protocol,
        args.expected_protocol_sha256,
        smoke=bool(args.smoke),
    )
    if args.max_processes < 1:
        raise ValueError("max-processes must be positive")
    if args.child_heartbeat_seconds <= 0:
        raise ValueError("child-heartbeat-seconds must be positive")

    git_commit = _git_commit()
    run_key = derive_run_key(
        git_commit=git_commit,
        feature_manifest_sha256=args.expected_feature_manifest_sha256,
        validation_protocol_sha256=args.expected_protocol_sha256,
        benchmark_config_sha256=args.expected_config_sha256,
        smoke=bool(args.smoke),
    )
    log = SupervisorLog(args.status_log.resolve())
    output_dir = args.work_dir.resolve()
    logs_dir = output_dir / "logs"

    log.event(
        "supervisor_started",
        run_key=run_key,
        git_commit=git_commit,
        output_dir=output_dir,
        checkpoint_s3_uri=args.checkpoint_s3_uri,
        final_s3_uri=args.final_s3_uri,
        smoke=bool(args.smoke),
    )

    store = DurableStore(
        checkpoint_s3_uri=args.checkpoint_s3_uri,
        final_s3_uri=args.final_s3_uri,
        region=args.region,
        log=log,
    )

    restored = store.restore_latest(
        output_dir,
        run_key=run_key,
        git_commit=git_commit,
        feature_manifest_sha256=args.expected_feature_manifest_sha256,
        validation_protocol_sha256=args.expected_protocol_sha256,
        benchmark_config_sha256=args.expected_config_sha256,
        smoke=bool(args.smoke),
    )
    log.event("supervisor_checkpoint_state", restored_model_folds=restored)

    _sync_feature_snapshot(
        s3_uri=args.feature_s3_uri,
        destination=args.feature_cache_dir.resolve(),
        expected_manifest_sha256=args.expected_feature_manifest_sha256,
        region=args.region,
        log=log,
    )

    previous_count = validate_benchmark_state(
        output_dir,
        git_commit=git_commit,
        feature_manifest_sha256=args.expected_feature_manifest_sha256,
        validation_protocol_sha256=args.expected_protocol_sha256,
        benchmark_config_sha256=args.expected_config_sha256,
        smoke=bool(args.smoke),
    )
    if previous_count > expected_total:
        raise RuntimeError("checkpoint count exceeds experiment plan")
    if previous_count != restored:
        raise RuntimeError(
            "restored checkpoint count disagrees with verified local state: "
            f"remote={restored} local={previous_count}"
        )

    if previous_count:
        store.commit_checkpoint(
            output_dir,
            run_key=run_key,
            sequence=previous_count,
            completed_model_folds=previous_count,
            git_commit=git_commit,
            feature_manifest_sha256=args.expected_feature_manifest_sha256,
            validation_protocol_sha256=args.expected_protocol_sha256,
            benchmark_config_sha256=args.expected_config_sha256,
            smoke=bool(args.smoke),
        )

    for process_index in range(1, args.max_processes + 1):
        summary_path = output_dir / "benchmark_summary.json"
        if summary_path.is_file() and previous_count == expected_total:
            break

        command = _child_command(
            feature_dir=args.feature_cache_dir.resolve(),
            feature_sha256=args.expected_feature_manifest_sha256,
            protocol_path=args.validation_protocol,
            protocol_sha256=args.expected_protocol_sha256,
            config_path=args.config,
            config_sha256=args.expected_config_sha256,
            output_dir=output_dir,
            logs_dir=logs_dir,
            smoke=bool(args.smoke),
        )
        log.event(
            "child_process_started",
            process_index=process_index,
            completed_model_folds=previous_count,
            command=command,
        )

        return_code = _run_child(
            command,
            heartbeat_seconds=args.child_heartbeat_seconds,
            log=log,
        )

        current_count = validate_benchmark_state(
            output_dir,
            git_commit=git_commit,
            feature_manifest_sha256=args.expected_feature_manifest_sha256,
            validation_protocol_sha256=args.expected_protocol_sha256,
            benchmark_config_sha256=args.expected_config_sha256,
            smoke=bool(args.smoke),
        )
        if current_count < previous_count:
            raise RuntimeError("benchmark checkpoint count regressed")
        if current_count > previous_count + 1:
            raise RuntimeError("child completed more than one model-fold checkpoint")

        if current_count > previous_count or summary_path.is_file():
            store.commit_checkpoint(
                output_dir,
                run_key=run_key,
                sequence=current_count,
                completed_model_folds=current_count,
                git_commit=git_commit,
                feature_manifest_sha256=args.expected_feature_manifest_sha256,
                validation_protocol_sha256=args.expected_protocol_sha256,
                benchmark_config_sha256=args.expected_config_sha256,
                smoke=bool(args.smoke),
            )

        if return_code != 0:
            raise RuntimeError(
                f"benchmark child failed with exit code {return_code}; "
                "all completed checkpoints were committed before supervisor exit"
            )
        if current_count == previous_count and not summary_path.is_file():
            raise RuntimeError("child exited successfully without a new checkpoint")

        previous_count = current_count
        log.event(
            "supervisor_progress",
            completed_model_folds=previous_count,
            total_model_folds=expected_total,
            percent=round(100.0 * previous_count / expected_total, 1),
        )

    summary_path = output_dir / "benchmark_summary.json"
    if previous_count != expected_total or not summary_path.is_file():
        raise RuntimeError(
            "supervisor exhausted process budget before benchmark completion: "
            f"completed={previous_count}/{expected_total}"
        )

    final_uri = store.publish_final(
        output_dir,
        run_key=run_key,
        completed_model_folds=previous_count,
        git_commit=git_commit,
        feature_manifest_sha256=args.expected_feature_manifest_sha256,
        validation_protocol_sha256=args.expected_protocol_sha256,
        benchmark_config_sha256=args.expected_config_sha256,
        smoke=bool(args.smoke),
    )
    log.event(
        "supervisor_completed",
        total_elapsed_seconds=round(time.monotonic() - started, 3),
        completed_model_folds=previous_count,
        total_model_folds=expected_total,
        benchmark_summary_sha256=sha256_file(summary_path),
        final_s3_uri=final_uri,
    )

    print(f"RUN_KEY={run_key}")
    print(f"COMPLETED_MODEL_FOLDS={previous_count}/{expected_total}")
    print(f"BENCHMARK_SUMMARY_SHA256={sha256_file(summary_path)}")
    print(f"FINAL_BENCHMARK_S3={final_uri}")
    print("DURABLE_MODEL_BENCHMARK_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
