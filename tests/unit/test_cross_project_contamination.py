from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    "src",
    "scripts",
    "tests",
    "configs",
    ".github",
)

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".json",
}

FOREIGN_TOKENS = (
    "otto" + "_" + "recsys",
    "otto" + "-" + "recommender-system",
    "OTTO" + "_" + "RETRIEVAL",
)


def test_no_cross_project_source_contamination() -> None:
    offenders: list[str] = []

    for root_name in SCAN_ROOTS:
        scan_root = ROOT / root_name
        if not scan_root.exists():
            continue

        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix not in TEXT_SUFFIXES and path.name != "Makefile":
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            if any(token in content for token in FOREIGN_TOKENS):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == [], (
        "Foreign-project source detected in Home Credit repository: " + ", ".join(sorted(offenders))
    )
