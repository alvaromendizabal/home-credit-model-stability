from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = re.compile(r"(^|[_-])(fix|fixed|repair|repaired|final[0-9]*|v[0-9]+)([_\.-]|$)", re.I)
IGNORED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "artifacts",
    "logs",
}


def project_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_no_forbidden_source_suffixes() -> None:
    offenders = [path.as_posix() for path in project_files() if FORBIDDEN.search(path.name)]
    assert offenders == []


def test_runtime_caches_are_ignored_by_git() -> None:
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/"):
        assert pattern in gitignore
