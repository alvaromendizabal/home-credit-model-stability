from __future__ import annotations

import pytest

from home_credit.modeling.data import FeatureRef
from home_credit.modeling.screening import _select_with_block_floor, feature_refs_from_payload


def _feature(name: str, block: str) -> FeatureRef:
    return FeatureRef(
        name=name,
        block=block,
        family=block.split("_depth", maxsplit=1)[0],
        depth=0,
        dtype="double",
        categorical=False,
    )


def test_block_floor_preserves_source_diversity() -> None:
    ranking = [
        (_feature("a1", "a_depth0"), 1.0, 10.0, 0.0),
        (_feature("a2", "a_depth0"), 0.9, 9.0, 0.0),
        (_feature("a3", "a_depth0"), 0.8, 8.0, 0.0),
        (_feature("b1", "b_depth0"), 0.1, 1.0, 0.0),
        (_feature("b2", "b_depth0"), 0.05, 0.5, 0.0),
    ]

    selected = _select_with_block_floor(
        ranking,
        max_features=3,
        min_features_per_block=1,
    )

    assert selected == {"a1", "a2", "b1"}


def test_feature_screen_payload_rejects_duplicate_names() -> None:
    payload = {
        "selected_features": [
            {
                "name": "duplicate",
                "block": "a_depth0",
                "family": "a",
                "depth": 0,
                "dtype": "double",
                "categorical": False,
            },
            {
                "name": "duplicate",
                "block": "b_depth0",
                "family": "b",
                "depth": 0,
                "dtype": "double",
                "categorical": False,
            },
        ]
    }

    with pytest.raises(ValueError, match="duplicate"):
        feature_refs_from_payload(payload)
