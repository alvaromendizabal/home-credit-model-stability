from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from home_credit.data.loader import RawManifestRecord
from home_credit.features.builder import (
    FeatureRecipe,
    _base_block,
    _decision_frame,
    group_logical_sources,
    load_validation_protocol,
    select_sources,
)
from home_credit.validation.protocol import attach_protocol_sha256


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("representation", ["string", "date", "datetime", "zoned", "days", "null"])
def test_decision_dates_preserve_calendar_features(split, representation, capfd) -> None:
    dates = pl.Series("date_decision", [date(2020, 2, 29), date(2020, 12, 31), None])
    expected = dates
    if representation == "string":
        dates = dates.cast(pl.String)
    elif representation == "datetime":
        dates = dates.cast(pl.Datetime("ms"))
    elif representation == "zoned":
        dates = dates.cast(pl.Datetime("us", "America/Los_Angeles"))
        expected = dates.cast(pl.Date)
    elif representation == "days":
        dates = dates.cast(pl.Int32)
    elif representation == "null":
        dates = pl.Series("date_decision", [None] * 3)
        expected = dates.cast(pl.Date)
    columns = {"case_id": [1, 2, 3], "WEEK_NUM": [1] * 3, "MONTH": [1] * 3}
    if split == "train":
        columns["target"] = [0, 1, 0]
    base = pl.DataFrame(columns).with_columns(dates).lazy()
    decision = _decision_frame(base).collect(engine="streaming")
    assert decision["_decision_date"].equals(expected.alias("_decision_date"))
    assert_frame_equal(
        _base_block(base, split=split).collect(engine="streaming"),
        _base_block(base.with_columns(expected), split=split).collect(engine="streaming"),
    )
    assert "DeprecationWarning" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            ["2020-02-29", "2019-02-29", "2020/01/31", "2020-01-31 00:00:00", "", None],
            [date(2020, 2, 29), None, None, None, None, None],
        ),
        (["invalid", None], [None, None]),
        ([], []),
    ],
)
def test_text_decision_dates_use_iso_format_without_inference(values, expected, capfd) -> None:
    base = pl.DataFrame(
        {"case_id": range(len(values)), "date_decision": pl.Series(values, dtype=pl.String)}
    ).lazy()
    result = _decision_frame(base).collect(engine="streaming")
    assert result["_decision_date"].to_list() == expected
    assert result.schema["_decision_date"] == pl.Date
    assert "DeprecationWarning" not in capfd.readouterr().err


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


def test_normalized_scan_harmonizes_cross_shard_dtype_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds

    import home_credit.features.builder as builder
    from home_credit.features.aggregation import (
        aggregate_case_history,
    )

    datasets = {
        "raw/shard_0.parquet": (
            ds.dataset(
                pa.table(
                    {
                        "case_id": [1],
                        "num_group1": [0],
                        "amount_1A": [10.0],
                        "status_1L": [1.0],
                    }
                )
            )
        ),
        "raw/shard_1.parquet": (
            ds.dataset(
                pa.table(
                    {
                        "case_id": [2],
                        "num_group1": [0],
                        "amount_1A": [20.0],
                        "status_1L": ["active"],
                    }
                )
            )
        ),
    }

    monkeypatch.setattr(
        builder,
        "_dataset_for_record",
        lambda record, store: datasets[record.s3_key],
    )

    source = builder.LogicalSource(
        split="test",
        family="demo",
        depth=1,
        records=(
            _record(
                ("parquet_files/test/test_demo_1_0.parquet"),
                "raw/shard_0.parquet",
            ),
            _record(
                ("parquet_files/test/test_demo_1_1.parquet"),
                "raw/shard_1.parquet",
            ),
        ),
    )

    recipe = FeatureRecipe(
        schema_version=1,
        name="competition_core",
        engine="polars_streaming",
        compression="zstd",
        scan_batch_rows=1024,
        time_windows_days=(30,),
        person_subgroups=("all",),
        numeric_aggregations=("min",),
        date_aggregations=("min",),
        categorical_aggregations=("n_unique",),
        global_frequency_encoding=("deferred_to_fold_fit"),
        quantiles=("deferred_to_extended_ablation"),
        notes="test",
    )

    frame, semantic = builder._normalized_scan(
        source,
        store=object(),
        decision_frame=None,
        recipe=recipe,
    )

    schema = frame.collect_schema()

    assert semantic.numeric == ("amount_1A",)

    assert semantic.categorical == ("status_1L",)

    assert schema["amount_1A"] == pl.Float64

    assert schema["status_1L"] == pl.String

    result = aggregate_case_history(
        frame,
        family="demo",
        depth=1,
        numeric_columns=(semantic.numeric),
        categorical_columns=(semantic.categorical),
        date_columns=(semantic.date),
        time_windows_days=(30,),
    ).collect()

    assert result.height == 2

    assert "demo__d1__amount_1A__sum" in result.columns

    assert "demo__d1__status_1L__n_unique" in result.columns


def _runtime_feature_recipe() -> FeatureRecipe:
    return FeatureRecipe(
        schema_version=1,
        name="competition_core",
        engine="polars_streaming",
        compression="zstd",
        scan_batch_rows=1024,
        time_windows_days=(30,),
        person_subgroups=("all",),
        numeric_aggregations=("min",),
        date_aggregations=("min",),
        categorical_aggregations=("n_unique",),
        global_frequency_encoding=("deferred_to_fold_fit"),
        quantiles=("deferred_to_extended_ablation"),
        notes="test",
    )


def test_normalized_scan_handles_empty_null_case_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds

    import home_credit.features.builder as builder
    from home_credit.features.aggregation import (
        aggregate_case_history,
    )

    empty_dataset = ds.dataset(
        pa.table(
            {
                "case_id": pa.array(
                    [],
                    type=pa.null(),
                ),
                "num_group1": pa.array(
                    [],
                    type=pa.null(),
                ),
                "amount_1A": pa.array(
                    [],
                    type=pa.null(),
                ),
                "event_1D": pa.array(
                    [],
                    type=pa.null(),
                ),
            }
        )
    )

    monkeypatch.setattr(
        builder,
        "_dataset_for_record",
        lambda record, store: empty_dataset,
    )

    source = builder.LogicalSource(
        split="test",
        family="tax_registry_c",
        depth=1,
        records=(
            _record(
                ("parquet_files/test/test_tax_registry_c_1.parquet"),
                "raw/test_tax_registry_c_1.parquet",
            ),
        ),
    )

    decision_frame = pl.DataFrame(
        {
            "case_id": [1],
            "_decision_date": [date(2020, 1, 1)],
        }
    ).lazy()

    frame, semantic = builder._normalized_scan(
        source,
        store=object(),
        decision_frame=(decision_frame),
        recipe=(_runtime_feature_recipe()),
    )

    schema = frame.collect_schema()

    assert schema["case_id"] == pl.Int64

    assert frame.collect().height == 0

    aggregated = aggregate_case_history(
        frame,
        family=source.family,
        depth=source.depth,
        numeric_columns=(semantic.numeric),
        categorical_columns=(semantic.categorical),
        date_columns=(semantic.date),
        time_windows_days=(30,),
    ).collect()

    assert aggregated.height == 0
    assert aggregated.schema["case_id"] == pl.Int64


def test_normalized_scan_restricts_rows_to_base_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds

    import home_credit.features.builder as builder

    dataset = ds.dataset(
        pa.table(
            {
                "case_id": [
                    1,
                    999,
                ],
                "num_group1": [
                    0,
                    0,
                ],
                "amount_1A": [
                    10.0,
                    20.0,
                ],
            }
        )
    )

    monkeypatch.setattr(
        builder,
        "_dataset_for_record",
        lambda record, store: dataset,
    )

    source = builder.LogicalSource(
        split="test",
        family="demo",
        depth=1,
        records=(
            _record(
                ("parquet_files/test/test_demo_1.parquet"),
                "raw/test_demo_1.parquet",
            ),
        ),
    )

    decision_frame = pl.DataFrame(
        {
            "case_id": [1],
            "_decision_date": [date(2020, 1, 1)],
        }
    ).lazy()

    frame, _ = builder._normalized_scan(
        source,
        store=object(),
        decision_frame=(decision_frame),
        recipe=(_runtime_feature_recipe()),
    )

    result = frame.select("case_id").collect()

    assert result["case_id"].to_list() == [1]


def test_normalized_scan_allows_zero_overlap_with_base_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds

    import home_credit.features.builder as builder

    dataset = ds.dataset(
        pa.table(
            {
                "case_id": [
                    999,
                    1000,
                ],
                "num_group1": [
                    0,
                    1,
                ],
                "amount_1A": [
                    10.0,
                    20.0,
                ],
            }
        )
    )

    monkeypatch.setattr(
        builder,
        "_dataset_for_record",
        lambda record, store: dataset,
    )

    source = builder.LogicalSource(
        split="test",
        family="demo",
        depth=1,
        records=(
            _record(
                ("parquet_files/test/test_demo_1.parquet"),
                "raw/test_demo_1.parquet",
            ),
        ),
    )

    decision_frame = pl.DataFrame(
        {
            "case_id": [
                1,
                2,
            ],
            "_decision_date": [
                date(2020, 1, 1),
                date(2020, 1, 2),
            ],
        }
    ).lazy()

    frame, semantic = builder._normalized_scan(
        source,
        store=object(),
        decision_frame=(decision_frame),
        recipe=(_runtime_feature_recipe()),
    )

    result = frame.collect()

    assert result.height == 0
    assert result.schema["case_id"] == pl.Int64

    assert semantic.numeric == ("amount_1A",)
