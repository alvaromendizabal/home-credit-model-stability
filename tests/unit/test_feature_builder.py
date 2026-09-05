from __future__ import annotations

import json
from pathlib import Path

import pytest

from home_credit.data.loader import RawManifestRecord
from home_credit.features.builder import (
    FeatureRecipe,
    group_logical_sources,
    load_validation_protocol,
    select_sources,
)
from home_credit.validation.protocol import attach_protocol_sha256


def _record(file: str, key: str) -> RawManifestRecord:
    return RawManifestRecord(
        file=file,
        s3_key=key,
        bytes=100,
        sha256="a" * 64,
    )


def test_group_logical_sources_combines_shards() -> None:
    records = [
        _record("parquet_files/train/train_base.parquet", "raw/train_base.parquet"),
        _record(
            "parquet_files/train/train_credit_bureau_a_1_1.parquet",
            "raw/train_credit_bureau_a_1_1.parquet",
        ),
        _record(
            "parquet_files/train/train_credit_bureau_a_1_0.parquet",
            "raw/train_credit_bureau_a_1_0.parquet",
        ),
        _record("parquet_files/test/test_base.parquet", "raw/test_base.parquet"),
    ]

    sources = group_logical_sources(records)
    credit = next(source for source in sources if source.family == "credit_bureau_a")

    assert credit.split == "train"
    assert credit.depth == 1
    assert [record.file for record in credit.records] == [
        "parquet_files/train/train_credit_bureau_a_1_0.parquet",
        "parquet_files/train/train_credit_bureau_a_1_1.parquet",
    ]


def test_source_selection_always_retains_base() -> None:
    sources = group_logical_sources(
        [
            _record("parquet_files/train/train_base.parquet", "raw/train_base.parquet"),
            _record("parquet_files/train/train_other_1.parquet", "raw/train_other_1.parquet"),
            _record("parquet_files/train/train_person_1.parquet", "raw/train_person_1.parquet"),
        ]
    )

    selected = select_sources(
        sources,
        splits=frozenset({"train"}),
        families=frozenset({"other"}),
    )

    assert {(source.family, source.depth) for source in selected} == {
        ("base", 0),
        ("other", 1),
    }


def test_recipe_load_is_fingerprinted_and_rejects_global_encoding(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "name": "competition_core",
        "engine": "polars_streaming",
        "compression": "zstd",
        "scan_batch_rows": 131072,
        "time_windows_days": [30, 180],
        "person_subgroups": ["all", "applicant", "related"],
        "numeric_aggregations": ["min"],
        "date_aggregations": ["min"],
        "categorical_aggregations": ["n_unique"],
        "global_frequency_encoding": "deferred_to_fold_fit",
        "quantiles": "deferred_to_extended_ablation",
        "notes": "test",
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    recipe, digest = FeatureRecipe.load(path)
    assert recipe.name == "competition_core"
    assert len(digest) == 64

    payload["global_frequency_encoding"] = "fit_on_all_rows"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fold-fitted"):
        FeatureRecipe.load(path)


def test_validation_protocol_requires_expected_fingerprint(tmp_path: Path) -> None:
    payload = attach_protocol_sha256(
        {
            "schema_version": 1,
            "name": "validation",
        }
    )
    path = tmp_path / "validation_protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    expected = str(payload["protocol_sha256"])
    loaded = load_validation_protocol(path, expected_sha256=expected)
    assert loaded["protocol_sha256"] == expected

    with pytest.raises(ValueError, match="mismatch"):
        load_validation_protocol(path, expected_sha256="0" * 64)
