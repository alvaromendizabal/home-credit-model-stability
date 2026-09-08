"""Order-invariant, per-applicant summaries of verified raw numeric histories.

Quantiles use linear interpolation. IQR and p90 need three finite observations;
adjusted Fisher-Pearson skew needs five and positive variance. Undefined values
remain missing. Dates and source group indices never become numeric predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from home_credit.features.semantics import (
    STRUCTURAL_COLUMNS,
    assert_no_target_leakage,
    classify_columns,
    resolve_semantic_columns,
)
from home_credit.modeling.acceptance import require
from home_credit.modeling.data import FeatureRef

STATISTICS = ("median", "iqr", "p90", "skew")


@dataclass(frozen=True, slots=True)
class HistoryFeature:
    """A raw column, aggregation and source identity; no learned population state."""

    source: str
    depth: int
    column: str
    statistic: str

    @property
    def name(self) -> str:
        return f"history__{self.source}_depth{self.depth}__{self.column}__{self.statistic}"

    @property
    def ref(self) -> FeatureRef:
        family = f"history_{self.source}"
        return FeatureRef(
            self.name, f"{family}_depth{self.depth}", family, self.depth, "float", False
        )


def inspect_source(
    paths: tuple[Path, ...], source: str, depth: int
) -> tuple[tuple[HistoryFeature, ...], dict[str, Any]]:
    """Audit actual shard schemas, recording why chronology is not established."""
    require(depth in {1, 2} and bool(paths), "raw histories require depth one or two")
    observations = []
    types: dict[str, set[str]] = {}
    for path in paths:
        fields = tuple((f.name, str(f.type)) for f in pq.ParquetFile(path).schema_arrow)
        assert_no_target_leakage(tuple(n for n, _ in fields), context=source)
        require("case_id" in dict(fields), "history has no case identifier")
        observations.append(classify_columns(fields))
        for name, dtype in fields:
            types.setdefault(name, set()).add(dtype)
    semantic = resolve_semantic_columns(observations)
    specs = tuple(
        HistoryFeature(source, depth, column, statistic)
        for column in sorted(semantic.numeric)
        for statistic in STATISTICS
    )
    audit = {
        "source": source,
        "depth": depth,
        "numeric_columns": list(semantic.numeric),
        "categorical_columns_excluded": list(semantic.categorical),
        "unsupported_columns_excluded": list(semantic.unsupported),
        "structural_columns_excluded": sorted(set(types) & STRUCTURAL_COLUMNS),
        "date_fields": [
            {
                "column": name,
                "physical_types": sorted(types[name]),
                "event_order_verified": False,
                "availability_time_verified": False,
                "decision": "exclude from lag, trend and acceleration features",
                "reason": "A date suffix or storage type does not establish event meaning "
                "or when the value became available. The locked snapshot has no "
                "field-level availability contract.",
            }
            for name in semantic.date
        ],
        "chronology_admitted": False,
        "source_order_is_time": False,
        "scope": "Competition-provided historical records; order-invariant summaries. "
        "Production point-in-time availability is not certified by this dataset.",
    }
    return specs, audit


def scan_numeric_history(paths: tuple[Path, ...], columns: tuple[str, ...]) -> pl.LazyFrame:
    """Normalize numeric shards while retaining missing values and all history rows."""
    require(bool(paths) and bool(columns), "empty numeric history")
    frames = []
    for path in paths:
        frame = pl.scan_parquet(path)
        names = set(frame.collect_schema().names())
        expressions = [pl.col("case_id").cast(pl.Int64, strict=True)]
        for name in columns:
            value = (
                pl.col(name).cast(pl.Float64, strict=True)
                if name in names
                else pl.lit(None, dtype=pl.Float64)
            )
            expressions.append(pl.when(value.is_finite()).then(value).otherwise(None).alias(name))
        frames.append(frame.select(expressions))
    return pl.concat(frames, how="vertical")


def aggregate_history(
    history: pl.LazyFrame, cases: pl.DataFrame, specs: tuple[HistoryFeature, ...]
) -> pl.DataFrame:
    """Pool a case's rows across shards; never average shard-level quantiles.

    The explicit case population must contain development weeks only. Left joining
    it preserves applicants with no source history and their missing predictors.
    """
    require(bool(specs), "no history features")
    require(
        {"case_id", "WEEK_NUM"} <= set(cases.columns)
        and cases.height > 0
        and cases["case_id"].null_count() == 0
        and cases["case_id"].n_unique() == len(cases)
        and cases["WEEK_NUM"].null_count() == 0
        and bool(cases["WEEK_NUM"].is_between(0, 72).all()),
        "history population must be unique development cases in weeks 0-72",
    )
    require(len({s.name for s in specs}) == len(specs), "duplicate history specifications")
    require(
        all(
            s.statistic in STATISTICS
            and s.column not in STRUCTURAL_COLUMNS
            and not s.column.endswith("D")
            for s in specs
        ),
        "unsupported history feature or date/index leakage",
    )
    expressions = []
    for spec in specs:
        value = pl.col(spec.column)
        if spec.statistic == "median":
            summary = value.quantile(0.5, interpolation="linear")
        elif spec.statistic == "iqr":
            summary = pl.when(value.count() >= 3).then(
                value.quantile(0.75, interpolation="linear")
                - value.quantile(0.25, interpolation="linear")
            )
        elif spec.statistic == "p90":
            summary = pl.when(value.count() >= 3).then(value.quantile(0.9, interpolation="linear"))
        else:
            summary = pl.when((value.count() >= 5) & (value.var() > 0)).then(value.skew(bias=False))
        expressions.append(summary.alias(spec.name))
    population = cases.select(pl.col("case_id").cast(pl.Int64))
    ids = population["case_id"].to_numpy()
    lower, upper = int(ids.min()), int(ids.max())
    aggregated = (
        history.filter(pl.col("case_id").is_between(lower, upper))
        .join(population.lazy(), on="case_id", how="semi")
        .group_by("case_id")
        .agg(expressions)
    )
    finite = []
    for spec in specs:
        value = pl.col(spec.name).cast(pl.Float32)
        finite.append(pl.when(value.is_finite()).then(value).otherwise(None).alias(spec.name))
    return (
        population.lazy()
        .join(aggregated, on="case_id", how="left", validate="1:1")
        .select("case_id", *finite)
        .sort("case_id")
        .collect(engine="streaming")
    )
