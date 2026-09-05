from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from home_credit.modeling.data import (
    FeatureBlockRef,
    FeatureRef,
    FeatureSnapshot,
    load_feature_frame,
    load_fold_frames,
)


def _snapshot(tmp_path: Path) -> tuple[FeatureSnapshot, tuple[FeatureRef, ...]]:
    blocks = tmp_path / "blocks" / "train"
    blocks.mkdir(parents=True)

    base_path = blocks / "base_depth0.parquet"
    pl.DataFrame(
        {
            "case_id": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "WEEK_NUM": [0, 0, 1, 1, 2, 2],
            "base_numeric": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    ).write_parquet(base_path)

    static_path = blocks / "static_depth0.parquet"
    pl.DataFrame(
        {
            "case_id": [1, 3, 4, 6],
            "static_numeric": [100.0, 300.0, 400.0, 600.0],
        }
    ).write_parquet(static_path)

    base_block = FeatureBlockRef(
        split="train",
        family="base",
        depth=0,
        path=base_path,
        output_sha256="base",
        rows=6,
        feature_columns=2,
    )
    static_block = FeatureBlockRef(
        split="train",
        family="static",
        depth=0,
        path=static_path,
        output_sha256="static",
        rows=4,
        feature_columns=1,
    )
    features = (
        FeatureRef(
            name="base_numeric",
            block="base_depth0",
            family="base",
            depth=0,
            dtype="double",
            categorical=False,
        ),
        FeatureRef(
            name="static_numeric",
            block="static_depth0",
            family="static",
            depth=0,
            dtype="double",
            categorical=False,
        ),
    )
    snapshot = FeatureSnapshot(
        root=tmp_path,
        manifest_sha256="manifest",
        protocol_sha256="protocol",
        recipe_sha256="recipe",
        execution_sha256="execution",
        feature_git_commit="commit",
        train_blocks=(base_block, static_block),
        test_blocks=(),
        features=features,
    )
    return snapshot, features


def test_load_feature_frame_filters_weeks_and_left_joins(tmp_path: Path) -> None:
    snapshot, features = _snapshot(tmp_path)

    frame = load_feature_frame(
        snapshot,
        features,
        week_min=1,
        week_max=2,
        max_rows=None,
        seed=7,
    )

    assert frame.get_column("case_id").to_list() == [3, 4, 5, 6]
    assert frame.get_column("WEEK_NUM").to_list() == [1, 1, 2, 2]
    assert frame.get_column("static_numeric").null_count() == 1


def test_deterministic_row_cap_is_repeatable(tmp_path: Path) -> None:
    snapshot, features = _snapshot(tmp_path)

    first = load_feature_frame(
        snapshot,
        features,
        week_min=0,
        week_max=2,
        max_rows=3,
        seed=20260905,
    )
    second = load_feature_frame(
        snapshot,
        features,
        week_min=0,
        week_max=2,
        max_rows=3,
        seed=20260905,
    )

    assert first.get_column("case_id").to_list() == second.get_column("case_id").to_list()
    assert first.height == 3


def test_fold_loader_rejects_temporal_overlap(tmp_path: Path) -> None:
    snapshot, features = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        load_fold_frames(
            snapshot,
            features,
            train_week_min=0,
            train_week_max=1,
            validation_week_min=1,
            validation_week_max=2,
            seed=1,
        )


def test_load_feature_frame_rejects_non_integer_week_num(
    tmp_path: Path,
) -> None:
    snapshot, features = _snapshot(tmp_path)

    base = snapshot.train_block(
        "base",
        0,
    )

    frame = pl.read_parquet(base.path).with_columns(pl.col("WEEK_NUM").cast(pl.Float64))

    frame.write_parquet(base.path)

    with pytest.raises(
        ValueError,
        match=("WEEK_NUM extrema must be integers"),
    ):
        load_feature_frame(
            snapshot,
            features,
            week_min=0,
            week_max=2,
            max_rows=None,
            seed=20260905,
        )
