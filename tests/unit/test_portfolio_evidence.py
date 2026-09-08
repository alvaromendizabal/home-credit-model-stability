"""Only verified aggregate evidence may replace a published review input."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from home_credit.modeling.checkpoints import sha256_bytes
from home_credit.observability.logging import RunLogger

SPEC = importlib.util.spec_from_file_location(
    "portfolio_evidence",
    Path(__file__).resolve().parents[2] / "scripts/restore_portfolio_evidence.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("failure", [None, "digest", "encryption", "schema", "nonfinite"])
def test_restore_verifies_before_replacing_existing_evidence(tmp_path, failure):
    raw = json.dumps({"schema_version": 1, "metric": 0.6}).encode()
    if failure == "schema":
        raw = b'{"schema_version": 2}'
    if failure == "nonfinite":
        raw = b'{"schema_version": 1, "metric": NaN}'
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/model_release.json").write_text(
        json.dumps(
            {
                "selection": {
                    "key": "aggregate/selection.json",
                    "sha256": sha256_bytes(raw) if failure != "digest" else "0" * 64,
                }
            }
        )
    )
    output = tmp_path / "reports/model_selection/selection.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"previous valid evidence")
    client = Mock()
    client.get_object.return_value = {
        "Body": io.BytesIO(raw),
        "ServerSideEncryption": "wrong" if failure == "encryption" else "AES256",
    }
    logger = RunLogger("evidence-test", tmp_path / "logs")
    if failure is None:
        MODULE.restore_evidence(tmp_path, client, "test-bucket", logger)
        assert output.read_bytes() == raw
    else:
        with pytest.raises(ValueError):
            MODULE.restore_evidence(tmp_path, client, "test-bucket", logger)
        assert output.read_bytes() == b"previous valid evidence"


def test_oidc_rejects_other_repositories_and_branches(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "another/repository")
    with pytest.raises(ValueError, match="repository"):
        MODULE.evidence_client("role", "us-west-2")
    monkeypatch.setenv("GITHUB_REPOSITORY", "alvaromendizabal/home-credit-model-stability")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/6/merge")
    with pytest.raises(ValueError, match="branch"):
        MODULE.evidence_client("role", "us-west-2")
