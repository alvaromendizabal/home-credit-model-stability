from __future__ import annotations

from pathlib import Path


def test_git_setup_is_child_script_with_quality_and_runtime_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "git_setup.sh").read_text(encoding="utf-8")

    assert "bash scripts/check.sh" in script
    assert "pre-commit run --all-files" in script
    assert "runtime_content_staged" in script
    assert "git ls-remote --heads origin main" in script
    assert "git push -u origin main" in script
    assert "PUSH_MODE" in script
    assert "git_push_skipped reason=no_push_mode" in script
    assert "terminal_will_remain_open" in script
