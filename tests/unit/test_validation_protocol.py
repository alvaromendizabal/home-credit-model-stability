from __future__ import annotations

import pytest

from home_credit.validation.protocol import (
    TemporalFold,
    attach_protocol_sha256,
    build_expanding_folds,
    protocol_sha256,
    validate_expanding_folds,
    verify_protocol_sha256,
)


def test_expanding_folds_match_locked_range() -> None:
    folds = build_expanding_folds(
        development_week_min=0,
        development_week_max=72,
        n_splits=5,
        validation_window_weeks=8,
    )

    actual = [
        (
            fold.train_week_max,
            fold.validation_week_min,
            fold.validation_week_max,
        )
        for fold in folds
    ]

    assert actual == [
        (32, 33, 40),
        (40, 41, 48),
        (48, 49, 56),
        (56, 57, 64),
        (64, 65, 72),
    ]


def test_expanding_folds_have_no_leakage() -> None:
    folds = build_expanding_folds(
        development_week_min=0,
        development_week_max=72,
    )

    for fold in folds:
        assert fold.train_week_max < fold.validation_week_min

        assert fold.train_week_min == 0
        assert fold.validation_weeks == 8


def test_validation_tail_is_used_once() -> None:
    folds = build_expanding_folds(
        development_week_min=0,
        development_week_max=72,
    )

    validation_weeks = [
        week
        for fold in folds
        for week in range(
            fold.validation_week_min,
            fold.validation_week_max + 1,
        )
    ]

    assert validation_weeks == list(range(33, 73))

    assert len(validation_weeks) == len(set(validation_weeks))


def test_insufficient_history_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="too short",
    ):
        build_expanding_folds(
            development_week_min=0,
            development_week_max=30,
            n_splits=5,
            validation_window_weeks=8,
        )


def test_overlap_is_rejected() -> None:
    folds = list(
        build_expanding_folds(
            development_week_min=0,
            development_week_max=72,
        )
    )

    folds[0] = TemporalFold(
        fold=1,
        train_week_min=0,
        train_week_max=33,
        validation_week_min=33,
        validation_week_max=40,
    )

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        validate_expanding_folds(
            tuple(folds),
            development_week_min=0,
            development_week_max=72,
        )


def test_protocol_hash_is_deterministic() -> None:
    left = {
        "name": "temporal",
        "outer": {
            "validation_week_min": 73,
            "validation_week_max": 91,
        },
    }

    right = {
        "outer": {
            "validation_week_max": 91,
            "validation_week_min": 73,
        },
        "name": "temporal",
    }

    changed = {
        "name": "temporal",
        "outer": {
            "validation_week_min": 74,
            "validation_week_max": 91,
        },
    }

    assert protocol_sha256(left) == protocol_sha256(right)

    assert protocol_sha256(left) != protocol_sha256(changed)


def test_embedded_hash_detects_mutation() -> None:
    frozen = attach_protocol_sha256(
        {
            "name": "temporal",
            "outer": {
                "validation_week_min": 73,
                "validation_week_max": 91,
            },
        }
    )

    assert verify_protocol_sha256(frozen)

    mutated = dict(frozen)
    mutated["name"] = "other"

    assert not verify_protocol_sha256(mutated)
