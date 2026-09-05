from __future__ import annotations

import pytest

from home_credit.runtime.environment import CORE_PACKAGES, collect_environment_report


@pytest.mark.integration
def test_all_core_packages_import_together() -> None:
    report = collect_environment_report()
    assert set(report.packages) == set(CORE_PACKAGES)
    assert report.python.startswith("3.12.")
