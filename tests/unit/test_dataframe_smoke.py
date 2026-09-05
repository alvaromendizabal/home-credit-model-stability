from __future__ import annotations

from home_credit.runtime.smoke import dataframe_roundtrip


def test_dataframe_roundtrip() -> None:
    assert dataframe_roundtrip() == (10, 3)
