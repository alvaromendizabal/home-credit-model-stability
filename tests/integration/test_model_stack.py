from __future__ import annotations

import pytest

from home_credit.runtime.smoke import model_smoke


@pytest.mark.integration
def test_primary_gbdt_stack_can_fit_and_predict() -> None:
    results = model_smoke()
    assert {result.model for result in results} == {"lightgbm", "catboost", "xgboost"}
    assert all(result.auc >= 0.5 for result in results)
