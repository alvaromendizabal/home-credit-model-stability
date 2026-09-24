from __future__ import annotations

import json
from pathlib import Path

from scripts.review_post_release_frontier import build_summary

ROOT = Path(__file__).resolve().parents[2]


def test_post_release_frontier_reproduces_published_summary() -> None:
    expected = json.loads((ROOT / "reports/post_release_frontier/summary.json").read_text())
    observed = build_summary()
    assert observed == expected
    assert observed["applprev_histogram"]["five_fold_stability_wins"] == 5
    assert observed["applprev_histogram"]["external_probe_internal_promotion_passed"] is False
    assert observed["zero_fit_ensemble_audit"]["top_descriptive_weights"]["xgboost"] == 0.0
    assert observed["zero_fit_ensemble_audit"]["top_descriptive_weights"]["catboost"] == 0.0
    assert observed["denselight"]["status"] == "registered_not_executed"
