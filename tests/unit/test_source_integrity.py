from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_dependency_free_source_integrity_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/source_check.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
