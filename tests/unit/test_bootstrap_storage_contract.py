from __future__ import annotations

from pathlib import Path


def test_bootstrap_keeps_large_environment_off_small_home_volume() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "start_here.sh").read_text(encoding="utf-8")

    assert "/tmp/home-credit-model-stability-venv" in script
    assert 'ln -s "$VENV_TARGET" .venv' in script
    assert 'UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/home-credit-uv-cache}"' in script
    assert 'UV_LINK_MODE="${UV_LINK_MODE:-copy}"' in script
