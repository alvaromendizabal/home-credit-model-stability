#!/usr/bin/env python3
"""Restore hash-pinned aggregate evidence with short-lived, read-only GitHub OIDC."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import requests

from home_credit.modeling.checkpoints import atomic_write, sha256_bytes
from home_credit.modeling.release import checked_member, read_object
from home_credit.modeling.selection import require
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def evidence_client(role_arn: str, region: str) -> Any:
    """Request credentials in memory; tokens are never printed, persisted or committed."""
    require(
        os.environ.get("GITHUB_REPOSITORY") == "alvaromendizabal/home-credit-model-stability",
        "unexpected OIDC repository",
    )
    require(
        os.environ.get("GITHUB_REF")
        in {"refs/heads/main", "refs/heads/feat/validated-model-release"},
        "untrusted publication branch",
    )
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    require(url.startswith("https://"), "OIDC requires HTTPS")
    response = requests.get(
        url,
        params={"audience": "sts.amazonaws.com"},
        headers={"Authorization": "Bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]},
        timeout=20,
    )
    response.raise_for_status()
    token = response.json()["value"]
    credentials = boto3.client("sts", region_name=region).assume_role_with_web_identity(
        RoleArn=role_arn,
        RoleSessionName="home-credit-aggregate-review",
        WebIdentityToken=token,
        DurationSeconds=900,
    )["Credentials"]
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def restore_evidence(root: Path, client: Any, bucket: str, logger: RunLogger) -> None:
    """Only explicitly pinned aggregate JSON may enter the public review directory."""
    release = read_object(root / "configs/model_release.json")
    entries = [{**release["selection"], "path": "reports/model_selection/selection.json"}]
    publication = root / "configs/portfolio_review.json"
    if publication.is_file():
        entries.extend(read_object(publication)["artifacts"])
    for entry in entries:
        path = checked_member(root, entry["path"])
        require(
            path.suffix == ".json" and path.is_relative_to(root / "reports"),
            "only aggregate JSON can be published",
        )
        with StageTimer(logger, "restore_" + path.stem, heartbeat_seconds=15):
            response = client.get_object(Bucket=bucket, Key=entry["key"])
            with response["Body"] as body:
                raw = body.read()
            require(sha256_bytes(raw) == entry["sha256"], "aggregate evidence digest mismatch")
            require(response.get("ServerSideEncryption") == "AES256", "evidence encryption changed")
            payload = json.loads(raw)
            require(
                isinstance(payload, dict) and payload.get("schema_version") == 1,
                "unsupported evidence schema",
            )
            json.dumps(payload, allow_nan=False)
            atomic_write(path, raw)
            logger.event(
                "aggregate_evidence_restored",
                file=path.name,
                sha256=entry["sha256"],
                bytes=len(raw),
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    role = os.environ["HOME_CREDIT_EVIDENCE_ROLE"]
    bucket = os.environ["HOME_CREDIT_ARTIFACT_BUCKET"]
    region = "us-west-2"
    restore_evidence(
        root, evidence_client(role, region), bucket, RunLogger("evidence-review", root / "logs")
    )


if __name__ == "__main__":
    main()
