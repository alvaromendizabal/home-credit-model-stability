from __future__ import annotations

from pathlib import Path

import pytest

from home_credit.runtime.environment import free_disk_gib, require_free_disk


def test_free_disk_gib_is_positive(tmp_path: Path) -> None:
    assert free_disk_gib(tmp_path) > 0


def test_require_free_disk_accepts_reachable_floor(tmp_path: Path) -> None:
    available = free_disk_gib(tmp_path)
    assert require_free_disk(max(available / 2, 0.01), tmp_path) == available


def test_require_free_disk_rejects_impossible_floor(tmp_path: Path) -> None:
    available = free_disk_gib(tmp_path)
    with pytest.raises(RuntimeError, match="insufficient free disk space"):
        require_free_disk(available + 1.0, tmp_path)


def test_require_free_disk_rejects_nonpositive_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        require_free_disk(0.0, tmp_path)
