"""Streaming-oriented case-level aggregations for relational Home Credit tables."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from home_credit.features.semantics import (
    CASE_ID,
    NUM_GROUP1,
    NUM_GROUP2,
    feature_name,
)


def history_order_columns(depth: int, columns: Sequence[str]) -> tuple[str, ...]:
    """Return the deterministic historical ordering columns available in a table."""
    available = set(columns)
    if depth == 2 and NUM_GROUP1 in available and NUM_GROUP2 in available:
        return (NUM_GROUP1, NUM_GROUP2)
    if depth >= 1 and NUM_GROUP1 in available:
        return (NUM_GROUP1,)
    return ()


def _ordered_first(column: str, order_by: tuple[str, ...]) -> pl.Expr:
    expr = pl.col(column)
    if order_by:
        expr = expr.sort_by(list(order_by), maintain_order=True)
    return expr.first()


def _ordered_last(column: str, order_by: tuple[str, ...]) -> pl.Expr:
    expr = pl.col(column)
    if order_by:
        expr = expr.sort_by(list(order_by), maintain_order=True)
    return expr.last()


def _numeric_expressions(
    columns: Sequence[str],
    *,
    family: str,
    depth: int,
    subgroup: str | None,
    order_by: tuple[str, ...],
) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []

    for column in columns:
        expressions.extend(
            [
                pl.col(column)
                .min()
                .alias(feature_name(family, depth, column, "min", subgroup=subgroup)),
                pl.col(column)
                .max()
                .alias(feature_name(family, depth, column, "max", subgroup=subgroup)),
                pl.col(column)
                .mean()
                .alias(feature_name(family, depth, column, "mean", subgroup=subgroup)),
                pl.col(column)
                .std(ddof=1)
                .alias(feature_name(family, depth, column, "std", subgroup=subgroup)),
                pl.col(column)
                .sum()
                .alias(feature_name(family, depth, column, "sum", subgroup=subgroup)),
                _ordered_first(column, order_by).alias(
                    feature_name(family, depth, column, "first", subgroup=subgroup)
                ),
                _ordered_last(column, order_by).alias(
                    feature_name(family, depth, column, "last", subgroup=subgroup)
                ),
            ]
        )

    return expressions


def _date_expressions(
    columns: Sequence[str],
    *,
    family: str,
    depth: int,
    subgroup: str | None,
    order_by: tuple[str, ...],
    time_windows_days: Sequence[int],
) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []

    for column in columns:
        expressions.extend(
            [
                pl.col(column)
                .min()
                .alias(feature_name(family, depth, column, "min", subgroup=subgroup)),
                pl.col(column)
                .max()
                .alias(feature_name(family, depth, column, "max", subgroup=subgroup)),
                pl.col(column)
                .mean()
                .alias(feature_name(family, depth, column, "mean", subgroup=subgroup)),
                pl.col(column)
                .std(ddof=1)
                .alias(feature_name(family, depth, column, "std", subgroup=subgroup)),
                _ordered_first(column, order_by).alias(
                    feature_name(family, depth, column, "first", subgroup=subgroup)
                ),
                _ordered_last(column, order_by).alias(
                    feature_name(family, depth, column, "last", subgroup=subgroup)
                ),
            ]
        )

        for window_days in time_windows_days:
            expressions.append(
                pl.col(column)
                .is_between(-float(window_days), 0.0, closed="both")
                .sum()
                .cast(pl.UInt32)
                .alias(
                    feature_name(
                        family, depth, column, f"count_last_{window_days}d", subgroup=subgroup
                    )
                )
            )

    return expressions


def _categorical_expressions(
    columns: Sequence[str],
    *,
    family: str,
    depth: int,
    subgroup: str | None,
    order_by: tuple[str, ...],
) -> tuple[list[pl.Expr], list[tuple[str, str, str]]]:
    expressions: list[pl.Expr] = []
    ratios: list[tuple[str, str, str]] = []

    for column in columns:
        non_null_name = feature_name(family, depth, column, "non_null_count", subgroup=subgroup)
        unique_name = feature_name(family, depth, column, "n_unique", subgroup=subgroup)
        diversity_name = feature_name(family, depth, column, "diversity_ratio", subgroup=subgroup)

        expressions.extend(
            [
                pl.col(column).count().cast(pl.UInt32).alias(non_null_name),
                pl.col(column).n_unique().cast(pl.UInt32).alias(unique_name),
                _ordered_first(column, order_by).alias(
                    feature_name(family, depth, column, "first", subgroup=subgroup)
                ),
                _ordered_last(column, order_by).alias(
                    feature_name(family, depth, column, "last", subgroup=subgroup)
                ),
            ]
        )
        ratios.append((unique_name, non_null_name, diversity_name))

    return expressions, ratios


def aggregate_case_history(
    frame: pl.LazyFrame,
    *,
    family: str,
    depth: int,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    date_columns: Sequence[str],
    time_windows_days: Sequence[int],
    subgroup: str | None = None,
) -> pl.LazyFrame:
    """Condense a depth>0 history table to exactly one row per observed ``case_id``."""
    if depth <= 0:
        raise ValueError("history aggregation requires depth > 0")

    schema_names = frame.collect_schema().names()
    if CASE_ID not in schema_names:
        raise ValueError("history table is missing case_id")

    order_by = history_order_columns(depth, schema_names)
    predictors = tuple(sorted({*numeric_columns, *categorical_columns, *date_columns}))

    expressions: list[pl.Expr] = [
        pl.len()
        .cast(pl.UInt32)
        .alias(feature_name(family, depth, "rows", "count", subgroup=subgroup))
    ]

    if NUM_GROUP1 in schema_names:
        expressions.extend(
            [
                pl.col(NUM_GROUP1)
                .max()
                .alias(
                    feature_name(
                        family,
                        depth,
                        NUM_GROUP1,
                        "max",
                        subgroup=subgroup,
                    )
                ),
                pl.col(NUM_GROUP1)
                .n_unique()
                .cast(pl.UInt32)
                .alias(
                    feature_name(
                        family,
                        depth,
                        NUM_GROUP1,
                        "n_unique",
                        subgroup=subgroup,
                    )
                ),
            ]
        )

    if NUM_GROUP2 in schema_names:
        expressions.extend(
            [
                pl.col(NUM_GROUP2)
                .max()
                .alias(
                    feature_name(
                        family,
                        depth,
                        NUM_GROUP2,
                        "max",
                        subgroup=subgroup,
                    )
                ),
                pl.col(NUM_GROUP2)
                .n_unique()
                .cast(pl.UInt32)
                .alias(
                    feature_name(
                        family,
                        depth,
                        NUM_GROUP2,
                        "n_unique",
                        subgroup=subgroup,
                    )
                ),
            ]
        )

    if predictors:
        missing_fraction = (
            pl.sum_horizontal(
                [pl.col(column).is_null().cast(pl.UInt16) for column in predictors]
            ).cast(pl.Float32)
            / float(len(predictors))
        ).alias("_row_missing_fraction")
        frame = frame.with_columns(missing_fraction)
        expressions.extend(
            [
                pl.col("_row_missing_fraction")
                .mean()
                .alias(
                    feature_name(
                        family,
                        depth,
                        "row_missing_fraction",
                        "mean",
                        subgroup=subgroup,
                    )
                ),
                pl.col("_row_missing_fraction")
                .max()
                .alias(
                    feature_name(
                        family,
                        depth,
                        "row_missing_fraction",
                        "max",
                        subgroup=subgroup,
                    )
                ),
            ]
        )

    expressions.extend(
        _numeric_expressions(
            numeric_columns,
            family=family,
            depth=depth,
            subgroup=subgroup,
            order_by=order_by,
        )
    )
    expressions.extend(
        _date_expressions(
            date_columns,
            family=family,
            depth=depth,
            subgroup=subgroup,
            order_by=order_by,
            time_windows_days=time_windows_days,
        )
    )
    categorical_expressions, ratios = _categorical_expressions(
        categorical_columns,
        family=family,
        depth=depth,
        subgroup=subgroup,
        order_by=order_by,
    )
    expressions.extend(categorical_expressions)

    result = frame.group_by(CASE_ID, maintain_order=True).agg(expressions)

    if ratios:
        result = result.with_columns(
            [
                pl.when(pl.col(non_null_name) > 0)
                .then(pl.col(unique_name).cast(pl.Float32) / pl.col(non_null_name).cast(pl.Float32))
                .otherwise(None)
                .alias(diversity_name)
                for unique_name, non_null_name, diversity_name in ratios
            ]
        )

    return result


def build_person_subgroups(frame: pl.LazyFrame) -> tuple[tuple[str, pl.LazyFrame], ...]:
    """Return all/applicant/related person histories without learned transformations."""
    schema_names = frame.collect_schema().names()
    if NUM_GROUP1 not in schema_names:
        return (("all", frame),)

    return (
        ("all", frame),
        ("applicant", frame.filter(pl.col(NUM_GROUP1) == 0)),
        ("related", frame.filter(pl.col(NUM_GROUP1) > 0)),
    )
