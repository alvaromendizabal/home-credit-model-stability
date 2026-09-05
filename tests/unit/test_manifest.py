from __future__ import annotations

from pathlib import Path

from home_credit.data.manifest import build_manifest, sha256_file, write_manifest


def test_manifest_is_sorted_and_counts_bytes(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert [item["path"] for item in manifest["files"]] == ["a.txt", "b.txt"]
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == 3


def test_manifest_hashes_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"home-credit")
    first = sha256_file(path)
    second = sha256_file(path)
    assert first == second
    assert len(first) == 64


def test_manifest_write_is_atomic_and_valid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.txt").write_text("x", encoding="utf-8")
    output = tmp_path / "manifest.json"
    write_manifest(build_manifest(source), output)
    assert output.is_file()
    assert not output.with_suffix(".json.tmp").exists()
