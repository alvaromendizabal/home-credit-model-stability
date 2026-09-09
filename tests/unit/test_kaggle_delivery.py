"""The offline delivery rejects tampering before executing packaged source."""

from __future__ import annotations

import json
import runpy
import zipfile
from pathlib import Path

import pytest

_runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/kaggle_inference.py"))
BUNDLE_SHA256 = _runner["BUNDLE_SHA256"]
digest = _runner["digest"]
unpack_bundle = _runner["unpack_bundle"]
verify_assets = _runner["verify_assets"]


@pytest.fixture
def assets(tmp_path: Path) -> tuple[Path, str]:
    for name in ("inference_bundle.zip", "requirements.txt", "kaggle_inference.py"):
        (tmp_path / name).write_bytes(name.encode())
    manifest = {
        "schema_version": 1,
        "bundle_sha256": BUNDLE_SHA256,
        "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(tmp_path.iterdir())
        ],
    }
    (tmp_path / "runtime.json").write_text(json.dumps(manifest))
    return tmp_path, digest(tmp_path / "runtime.json")


def test_verify_assets_accepts_complete_manifest(assets):
    directory, expected = assets
    assert len(verify_assets(directory, expected)["files"]) == 3


@pytest.mark.parametrize("member", ["runtime.json", "requirements.txt", "kaggle_inference.py"])
def test_verify_assets_rejects_tampered_code_or_dependencies(assets, member):
    directory, expected = assets
    with (directory / member).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=r"digest mismatch|member changed"):
        verify_assets(directory, expected)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "wrong_bundle", "missing_script"])
def test_verify_assets_rejects_invalid_contract(assets, mutation):
    directory, _ = assets
    path = directory / "runtime.json"
    data = json.loads(path.read_text())
    if mutation == "duplicate":
        data["files"].append(data["files"][0])
    elif mutation == "traversal":
        data["files"][0]["path"] = "../outside"
    elif mutation == "wrong_bundle":
        data["bundle_sha256"] = "0" * 64
    else:
        data["files"] = [x for x in data["files"] if x["path"] != "kaggle_inference.py"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_assets(directory, digest(path))


def test_bundle_archive_rejects_traversal_before_writing(tmp_path):
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("safe.txt", "valid")
        stream.writestr("../outside.txt", "invalid")
    with pytest.raises(ValueError, match="unsafe"):
        unpack_bundle(archive, tmp_path / "output")
    assert not (tmp_path / "output/safe.txt").exists()


def test_bundle_archive_preserves_native_bytes(tmp_path):
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("models/native.txt", b"native model\n")
    unpack_bundle(archive, tmp_path / "output")
    assert (tmp_path / "output/models/native.txt").read_bytes() == b"native model\n"
