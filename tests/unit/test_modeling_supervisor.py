from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

from botocore.exceptions import ClientError

from home_credit.modeling.checkpoints import derive_run_key


def _load_supervisor_module() -> ModuleType:
    path = Path("scripts/run_model_benchmark_supervisor.py").resolve()

    spec = importlib.util.spec_from_file_location(
        "home_credit_model_benchmark_supervisor_test_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load supervisor module: {path}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def test_supervisor_run_key_separates_smoke_and_full() -> None:
    common = {
        "git_commit": "git",
        "feature_manifest_sha256": "feature",
        "validation_protocol_sha256": "protocol",
        "benchmark_config_sha256": "config",
    }

    full = derive_run_key(
        **common,
        smoke=False,
    )

    smoke = derive_run_key(
        **common,
        smoke=True,
    )

    assert full != smoke


def test_supervisor_child_command_limits_each_process_to_one_checkpoint(
    tmp_path: Path,
) -> None:
    supervisor = _load_supervisor_module()

    command = supervisor._child_command(
        feature_dir=tmp_path / "features",
        feature_sha256="feature",
        protocol_path=Path("configs/validation_protocol.json"),
        protocol_sha256="protocol",
        config_path=Path("configs/model_benchmark.json"),
        config_sha256="config",
        output_dir=tmp_path / "output",
        logs_dir=tmp_path / "logs",
        smoke=False,
    )

    option_index = command.index("--max-new-checkpoints")

    assert command[option_index + 1] == "1"

    assert "--smoke" not in command


def test_supervisor_checkpoint_commit_is_manifest_last(
    tmp_path: Path,
    monkeypatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeS3:
        def __init__(
            self,
        ) -> None:
            self.objects: dict[
                str,
                dict[str, object],
            ] = {}

            self.calls: list[tuple[str, str]] = []

        def head_object(
            self,
            *,
            Bucket: str,
            Key: str,
        ):
            del Bucket

            if Key not in self.objects:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "404",
                            "Message": "missing",
                        }
                    },
                    "HeadObject",
                )

            item = self.objects[Key]

            return {
                "ContentLength": len(item["body"]),
                "Metadata": item["metadata"],
                "ServerSideEncryption": item["sse"],
            }

        def put_object(
            self,
            *,
            Bucket: str,
            Key: str,
            Body,
            ServerSideEncryption: str,
            Metadata: dict[
                str,
                str,
            ],
            ContentType: str | None = None,
        ):
            del Bucket, ContentType

            body = (
                Body
                if isinstance(
                    Body,
                    bytes,
                )
                else Body.read()
            )

            self.objects[Key] = {
                "body": body,
                "metadata": dict(Metadata),
                "sse": (ServerSideEncryption),
            }

            self.calls.append(
                (
                    "put_object",
                    Key,
                )
            )

            return {}

        def upload_file(
            self,
            Filename: str,
            Bucket: str,
            Key: str,
            ExtraArgs: dict[
                str,
                object,
            ],
            Config,
        ) -> None:
            del Bucket, Config

            self.objects[Key] = {
                "body": Path(Filename).read_bytes(),
                "metadata": dict(ExtraArgs["Metadata"]),
                "sse": ExtraArgs["ServerSideEncryption"],
            }

            self.calls.append(
                (
                    "upload_file",
                    Key,
                )
            )

        def get_object(
            self,
            *,
            Bucket: str,
            Key: str,
        ):
            del Bucket

            if Key not in self.objects:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "404",
                            "Message": "missing",
                        }
                    },
                    "GetObject",
                )

            item = self.objects[Key]

            return {
                "Body": BytesIO(item["body"]),
                "Metadata": item["metadata"],
                "ServerSideEncryption": item["sse"],
            }

        def download_file(
            self,
            Bucket: str,
            Key: str,
            Filename: str,
            Config,
        ) -> None:
            del Bucket, Config

            Path(Filename).write_bytes(self.objects[Key]["body"])

    fake = FakeS3()

    monkeypatch.setattr(
        supervisor.boto3,
        "client",
        lambda *args, **kwargs: fake,
    )

    output = tmp_path / "output"

    output.mkdir()

    (output / "artifact.txt").write_text(
        "payload\n",
        encoding="utf-8",
    )

    log = supervisor.SupervisorLog(tmp_path / "supervisor.jsonl")

    store = supervisor.DurableStore(
        checkpoint_s3_uri=("s3://bucket/checkpoints/run"),
        final_s3_uri=("s3://bucket/final"),
        region="us-west-2",
        log=log,
    )

    store.commit_checkpoint(
        output,
        run_key="run",
        sequence=1,
        completed_model_folds=1,
        git_commit="git",
        feature_manifest_sha256=("feature"),
        validation_protocol_sha256=("protocol"),
        benchmark_config_sha256=("config"),
        smoke=False,
    )

    assert fake.calls[-1] == (
        "put_object",
        "checkpoints/run/latest.json",
    )

    assert any(key.startswith("checkpoints/run/objects/") for key in fake.objects)

    assert any(key.startswith("checkpoints/run/manifests/checkpoint-001-") for key in fake.objects)

    (output / "artifact.txt").unlink()

    restored = store.restore_latest(
        output,
        run_key="run",
        git_commit="git",
        feature_manifest_sha256=("feature"),
        validation_protocol_sha256=("protocol"),
        benchmark_config_sha256=("config"),
        smoke=False,
    )

    assert restored == 1

    assert (output / "artifact.txt").read_text(encoding="utf-8") == "payload\n"

    # A completed local fold can precede its S3 upload after a network interruption.
    (output / "artifact.txt").write_text("newer local checkpoint")
    monkeypatch.setattr(supervisor, "validate_benchmark_state", lambda *args, **kwargs: 2)
    restored = store.restore_latest(
        output,
        run_key="run",
        git_commit="git",
        feature_manifest_sha256="feature",
        validation_protocol_sha256="protocol",
        benchmark_config_sha256="config",
        smoke=False,
    )
    assert restored == 2
    assert (output / "artifact.txt").read_text() == "newer local checkpoint"


def test_supervisor_counts_enabled_models_and_rejects_modified_config():
    import json

    import pytest

    from home_credit.modeling.checkpoints import sha256_file

    supervisor = _load_supervisor_module()
    protocol = Path("configs/validation_protocol.json")
    protocol_sha = json.loads(protocol.read_text())["protocol_sha256"]
    for path, expected in [
        (Path("configs/model_benchmark.json"), 20),
        (Path("configs/ablations/control.json"), 5),
    ]:
        assert (
            supervisor.expected_model_folds(
                path, sha256_file(path), protocol, protocol_sha, smoke=False
            )
            == expected
        )
        assert (
            supervisor.expected_model_folds(
                path, sha256_file(path), protocol, protocol_sha, smoke=True
            )
            == expected // 5
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        supervisor.expected_model_folds(path, "0" * 64, protocol, protocol_sha, smoke=False)


def test_remote_empty_preserves_verified_local_folds(tmp_path, monkeypatch):
    supervisor = _load_supervisor_module()

    class EmptyS3:
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    monkeypatch.setattr(supervisor.boto3, "client", lambda *args, **kwargs: EmptyS3())
    monkeypatch.setattr(supervisor, "validate_benchmark_state", lambda *args, **kwargs: 2)
    store = supervisor.DurableStore(
        checkpoint_s3_uri="s3://bucket/checkpoints",
        final_s3_uri="s3://bucket/final",
        region="us-west-2",
        log=supervisor.SupervisorLog(tmp_path / "log.jsonl"),
    )
    assert (
        store.restore_latest(
            tmp_path,
            run_key="run",
            git_commit="git",
            feature_manifest_sha256="feature",
            validation_protocol_sha256="protocol",
            benchmark_config_sha256="config",
            smoke=False,
        )
        == 2
    )


def test_partial_feature_cache_resumes_missing_files(tmp_path, monkeypatch):
    import hashlib
    import json

    supervisor = _load_supervisor_module()
    contents = b"validated feature block"
    digest = hashlib.sha256(contents).hexdigest()
    manifest = {
        "blocks": [{"split": "train", "family": "base", "depth": 0, "output_sha256": digest}]
    }
    raw = json.dumps(manifest).encode()
    (tmp_path / "feature_manifest.json").write_bytes(raw)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    downloads = []

    class Page:
        def paginate(self, **kwargs):
            return [
                {
                    "Contents": [
                        {"Key": "snapshot/feature_manifest.json"},
                        {"Key": "snapshot/blocks/train/base_depth0.parquet"},
                    ]
                }
            ]

    class S3:
        def get_paginator(self, name):
            return Page()

        def download_file(self, bucket, key, destination):
            downloads.append(key)
            Path(destination).write_bytes(contents)

    monkeypatch.setattr(supervisor.boto3, "client", lambda *args, **kwargs: S3())
    kwargs = dict(
        s3_uri="s3://bucket/snapshot",
        destination=tmp_path,
        expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        region="us-west-2",
        log=supervisor.SupervisorLog(tmp_path / "log.jsonl"),
    )
    supervisor._sync_feature_snapshot(**kwargs)
    assert downloads == ["snapshot/blocks/train/base_depth0.parquet"]
    supervisor._sync_feature_snapshot(**kwargs)
    assert len(downloads) == 1
    assert unrelated.read_text() == "keep"
