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
_builder = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/build_kaggle_assets.py")
)
unpack_bundle = _builder["unpack_bundle"]
verify_assets = _runner["verify_assets"]


@pytest.fixture
def assets(tmp_path: Path) -> tuple[Path, str]:
    (tmp_path / "bundle").mkdir()
    for name in ("bundle/bundle.json", "requirements.txt", "kaggle_inference.py"):
        (tmp_path / name).write_bytes(name.encode())
    manifest = {
        "schema_version": 1,
        "bundle_sha256": BUNDLE_SHA256,
        "files": [
            {
                "path": p.relative_to(tmp_path).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
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


def test_package_has_explicit_native_members_and_no_nested_zip(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "reports/model_release").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/kaggle_inference.py").write_text("# synthetic packaging fixture\n")
    (root / "uv.lock").write_text("# synthetic dependency fixture\n")
    bundle = tmp_path / "native.zip"
    with zipfile.ZipFile(bundle, "w") as stream:
        stream.writestr("bundle.json", '{"phase":"all_labels"}')
        stream.writestr("model.txt", b"synthetic native bytes\n")
    state = {
        "stages": {
            "bundle": {
                "archive": {"sha256": digest(bundle)},
                "manifest": {"sha256": BUNDLE_SHA256},
            },
            "all_labels/lightgbm": {"fit_git_commit": "a" * 40},
        }
    }
    (root / "reports/model_release/state.json").write_text(json.dumps(state))
    build = _builder["build"]
    monkeypatch.setitem(build.__globals__, "locked_wheels", lambda _: [])
    destination = tmp_path / "delivery"
    receipt = build(root, bundle, destination)
    verify_assets(destination, receipt["runtime_sha256"])
    with zipfile.ZipFile(destination.with_suffix(".zip")) as stream:
        assert "bundle/bundle.json" in stream.namelist()
        assert stream.read("bundle/model.txt") == b"synthetic native bytes\n"
        assert not any(name.endswith(".zip") for name in stream.namelist())
