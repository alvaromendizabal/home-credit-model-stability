"""The compatibility comparison must reject unverified source and data before execution."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

verify_inputs = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/verify_date_parsing.py")
)["verify_inputs"]


@pytest.fixture
def comparison_inputs(tmp_path):
    root = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    raw = tmp_path / "raw"
    for path in (root / "reports/model_release", root / "configs", bundle, raw):
        path.mkdir(parents=True)
    source = bundle / "source.py"
    source.write_text("# Verified source fixture\n")
    data = raw / "test_base.parquet"
    data.write_bytes(b"raw fixture; never opened as parquet")
    members = [
        {
            "path": source.name,
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    bundle_manifest = bundle / "bundle.json"
    bundle_manifest.write_text(json.dumps({"files": members}))
    (root / "reports/model_release/state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "bundle": {
                        "manifest": {
                            "sha256": hashlib.sha256(bundle_manifest.read_bytes()).hexdigest()
                        }
                    }
                }
            }
        )
    )
    manifest = tmp_path / "raw_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "file": data.name,
                "bytes": data.stat().st_size,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        )
    )
    (root / "configs/validation_protocol.json").write_text(
        json.dumps(
            {"data_lock": {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
        )
    )
    return root, bundle, raw, manifest


def test_verifier_accepts_only_complete_matching_identities(comparison_inputs):
    assert len(verify_inputs(*comparison_inputs)) == 4


@pytest.mark.parametrize("changed", ["bundle_manifest", "raw_manifest", "source", "data"])
def test_verifier_rejects_changed_input_before_import(comparison_inputs, changed):
    _, bundle, raw, manifest = comparison_inputs
    path = {
        "bundle_manifest": bundle / "bundle.json",
        "raw_manifest": manifest,
        "source": bundle / "source.py",
        "data": raw / "test_base.parquet",
    }[changed]
    value = path.read_bytes()
    path.write_bytes(b"!" + value[1:])  # Same length: verify the hash, not just size.
    with pytest.raises(ValueError, match="changed"):
        verify_inputs(*comparison_inputs)
