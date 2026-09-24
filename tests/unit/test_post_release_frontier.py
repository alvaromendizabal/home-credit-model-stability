from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "post_release_frontier", ROOT / "scripts/review_post_release_frontier.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_post_release_frontier_reproduces_published_summary() -> None:
    expected = json.loads((ROOT / "reports/post_release_frontier/summary.json").read_text())
    observed = MODULE.build_summary()
    assert observed == expected
    assert observed["applprev_histogram"]["five_fold_stability_wins"] == 5
    assert observed["applprev_histogram"]["external_probe_internal_promotion_passed"] is False
    assert observed["zero_fit_ensemble_audit"]["top_descriptive_weights"]["xgboost"] == 0.0
    assert observed["zero_fit_ensemble_audit"]["top_descriptive_weights"]["catboost"] == 0.0
    assert observed["denselight"]["status"] == "registered_not_executed"
