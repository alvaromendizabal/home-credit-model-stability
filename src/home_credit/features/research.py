"""Case-local feature hypotheses and fold-fitted peer statistics.

This extension is exploratory development research after the frozen release.
It does not change the released feature recipe, models, or holdout result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from home_credit.modeling.acceptance import require
from home_credit.modeling.data import FeatureRef


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One transparent formula with explicit source columns and availability."""

    name: str
    family: str
    operation: str
    sources: tuple[str, ...]
    rationale: str

    @property
    def ref(self) -> FeatureRef:
        return FeatureRef(
            self.name,
            f"research_{self.family}",
            self.family,
            0,
            "String" if self.operation == "category_pair" else "Float32",
            self.operation == "category_pair",
        )


def catalog(scores: list[dict[str, Any]]) -> tuple[Hypothesis, ...]:
    """Enumerate justified combinations; prioritize only with the early-window screen."""
    names = {s["name"] for s in scores}
    require(not {"target", "case_id", "WEEK_NUM", "MONTH"} & names, "forbidden source")
    ranked = sorted(scores, key=lambda x: (-x["selection_score"], x["name"]))
    numeric = [s["name"] for s in ranked if not s["categorical"]]
    categories = [s["name"] for s in ranked if s["categorical"] and s["target_gain"] > 0][:16]
    result: list[Hypothesis] = []

    def add(family: str, op: str, sources: tuple[str, ...], rationale: str) -> None:
        if set(sources) <= names:
            identity = json.dumps([family, op, sources], separators=(",", ":"))
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
            result.append(
                Hypothesis(f"research__{family}__{op}__{suffix}", family, op, sources, rationale)
            )

    groups = sorted({n.rsplit("__", 1)[0] for n in numeric if n.endswith("__mean")})
    for stem in groups:
        add(
            "dispersion",
            "difference",
            (stem + "__max", stem + "__min"),
            "Historical range in the original variable's units.",
        )
        add(
            "dispersion",
            "relative_std",
            (stem + "__std", stem + "__mean"),
            "Variability relative to the magnitude of historical activity.",
        )
        add(
            "dispersion",
            "relative_range",
            (stem + "__max", stem + "__min", stem + "__mean"),
            "Range normalized by historical magnitude.",
        )
        add(
            "source_order",
            "difference",
            (stem + "__last", stem + "__first"),
            "Change by num_group source order; not assumed to be chronological.",
        )
    for name in sorted(names):
        if name.endswith("__count_last_30d"):
            stem = name.rsplit("__", 1)[0]
            for small, large in zip((30, 180, 365, 730), (180, 365, 730, 1825), strict=True):
                add(
                    "recency",
                    "positive_ratio",
                    (f"{stem}__count_last_{small}d", f"{stem}__count_last_{large}d"),
                    "Recent events as a share of a longer window.",
                )
    anchors = [
        n
        for n in (
            "static__d0__annuity_780A",
            "static__d0__credamount_770A",
            "static__d0__price_1097A",
        )
        if n in names
    ]
    money = [n for n in numeric if re.search(r"\dA(?:__(?:mean|max|sum|first|last))?$", n)]
    for anchor in anchors:
        for source in money:
            if source != anchor:
                add(
                    "amount_ratios",
                    "positive_ratio",
                    (source, anchor),
                    "Recorded monetary activity relative to current credit, price or annuity; "
                    "a relative exposure proxy, not a verified debt-to-income measure.",
                )
    for name in numeric:
        add(
            "missingness",
            "missing",
            (name,),
            "Absence of a source measurement at application time.",
        )
        if "__applicant__" in name:
            related = name.replace("__applicant__", "__related__")
            add(
                "household",
                "difference",
                (name, related),
                "Applicant versus related-person context; no cross-customer outcome aggregation.",
            )
    for a, b in itertools.combinations(categories, 2):
        add(
            "category_interactions",
            "category_pair",
            (a, b),
            "Joint category frequency is fitted only on the current training fold.",
        )
    for name in numeric[:48]:
        add("peer_statistics", "rank", (name,), "Empirical percentile against training cases only.")
    for category in categories[:8]:
        for amount in anchors:
            for op in ("peer_median_ratio", "peer_iqr_position"):
                add(
                    "peer_statistics",
                    op,
                    (amount, category),
                    "Exposure relative to training-only category median or interquartile range.",
                )
    require(len({h.name for h in result}) == len(result), "duplicate feature hypotheses")
    return tuple(sorted(result, key=lambda h: h.name))


def required_sources(specs: tuple[Hypothesis, ...]) -> set[str]:
    return {source for spec in specs for source in spec.sources}


def case_features(frame: pl.DataFrame, specs: tuple[Hypothesis, ...]) -> pl.DataFrame:
    """Evaluate case-local formulas; undefined ratios are null, never infinity."""
    expressions = []
    for spec in specs:
        if spec.family == "peer_statistics":
            continue
        raw = [pl.col(n) for n in spec.sources]
        values = [v.cast(pl.Float64, strict=False) for v in raw]
        a = values[0]
        if spec.operation == "category_pair":
            # Length-safe JSON struct encoding prevents separator and null-token collisions.
            expression = pl.struct([v.cast(pl.String).alias(str(i)) for i, v in enumerate(raw)])
            expressions.append(expression.struct.json_encode().alias(spec.name))
            continue
        if spec.operation == "missing":
            value = (a.is_null() | ~a.is_finite()).cast(pl.Float64)
        elif spec.operation == "difference":
            value = a - values[1]
        elif spec.operation == "positive_ratio":
            value = pl.when(values[1] > 0).then(a / values[1])
        elif spec.operation == "relative_std":
            value = pl.when(values[1].abs() > 1e-8).then(a / values[1].abs())
        elif spec.operation == "relative_range":
            value = pl.when(values[2].abs() > 1e-8).then((a - values[1]) / values[2].abs())
        else:
            raise ValueError(f"unknown case operation: {spec.operation}")
        value = value.cast(pl.Float32)
        expressions.append(pl.when(value.is_finite()).then(value).otherwise(None).alias(spec.name))
    return frame.select(expressions)


def peer_features(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    specs: tuple[Hypothesis, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Fit label-free rank/quantile references on train, and preserve them for replay."""
    train_columns: list[pl.Series] = []
    validation_columns: list[pl.Series] = []
    state: dict[str, Any] = {}
    for spec in specs:
        if spec.family != "peer_statistics":
            continue
        name = spec.sources[0]
        x = train[name].cast(pl.Float64).to_numpy()
        if spec.operation == "rank":
            values, counts = np.unique(x[np.isfinite(x)], return_counts=True)
            cumulative = np.cumsum(counts)
            state[spec.name] = {
                "operation": "rank",
                "source": name,
                "values": values.tolist(),
                "counts": cumulative.tolist(),
            }
            for frame, destination in ((train, train_columns), (validation, validation_columns)):
                current = frame[name].cast(pl.Float64).to_numpy()
                indices = np.searchsorted(values, current, side="right")
                distribution = np.r_[0, cumulative] / max(1, len(x[np.isfinite(x)]))
                out = distribution[indices]
                out[~np.isfinite(current)] = np.nan
                destination.append(pl.Series(spec.name, out, dtype=pl.Float32).fill_nan(None))
        else:
            category = spec.sources[1]
            key = pl.struct(pl.col(category).cast(pl.String).alias("category")).struct.json_encode()
            reference = train.select(key.alias("key"), pl.col(name).cast(pl.Float64).alias("value"))
            reference = (
                reference.filter(pl.col("value").is_finite())
                .group_by("key")
                .agg(
                    pl.col("value").quantile(0.25).alias("q25"),
                    pl.col("value").median().alias("median"),
                    pl.col("value").quantile(0.75).alias("q75"),
                    pl.len().alias("count"),
                )
                .filter(pl.col("count") >= 50)
                .sort("key")
            )
            state[spec.name] = {
                "operation": spec.operation,
                "sources": list(spec.sources),
                "minimum_peer_count": 50,
                "reference": reference.to_dicts(),
            }
            for frame, destination in ((train, train_columns), (validation, validation_columns)):
                query = frame.select(key.alias("key"), pl.col(name).cast(pl.Float64).alias("value"))
                query = query.join(reference, on="key", how="left", maintain_order="left")
                value: pl.Expr
                if spec.operation == "peer_median_ratio":
                    value = pl.when(pl.col("median") > 0).then(pl.col("value") / pl.col("median"))
                elif spec.operation == "peer_iqr_position":
                    spread = pl.col("q75") - pl.col("q25")
                    value = pl.when(spread > 0).then((pl.col("value") - pl.col("median")) / spread)
                else:
                    raise ValueError("unknown peer operation")
                value = value.cast(pl.Float32)
                destination.append(
                    query.select(
                        pl.when(value.is_finite()).then(value).otherwise(None).alias(spec.name)
                    )[spec.name]
                )
    return pl.DataFrame(train_columns), pl.DataFrame(validation_columns), state


def prune_candidates(
    train: pl.DataFrame,
    specs: tuple[Hypothesis, ...],
) -> tuple[tuple[Hypothesis, ...], list[dict[str, Any]]]:
    """Training-only constant, near-constant and exact-duplicate screening.

    Fingerprints cover every training row. Approximate correlations are reported
    later on retained features; a high correlation alone does not prove redundancy.
    """
    seen: dict[str, str] = {}
    retained = []
    records = []
    for spec in specs:
        column = train[spec.name]
        counts = column.value_counts()["count"]
        dominant_fraction = float(counts.to_numpy().max(initial=0)) / max(1, len(column))
        # Hash the canonical complete column, including the null mask and its dtype.
        fingerprint = hashlib.sha256(column.hash(seed=20260905).to_numpy().tobytes()).hexdigest()
        reason = None
        duplicate_of = None
        if column.n_unique() <= 1:
            reason = "constant"
        elif dominant_fraction >= 0.9995:
            reason = "near_constant"
        elif fingerprint in seen and column.equals(train[seen[fingerprint]], check_dtypes=True):
            reason, duplicate_of = "exact_duplicate", seen[fingerprint]
        else:
            seen[fingerprint] = spec.name
            retained.append(spec)
        records.append(
            {
                **asdict(spec),
                "dominant_fraction": dominant_fraction,
                "structural_rejection": reason,
                "duplicate_of": duplicate_of,
            }
        )
    return tuple(retained), records


def apply_peer_references(
    frame: pl.DataFrame, specs: tuple[Hypothesis, ...], state: dict[str, Any]
) -> pl.DataFrame:
    """Replay persisted rank and peer maps without accessing any training rows."""
    peers = tuple(s for s in specs if s.family == "peer_statistics")
    require(set(state) == {s.name for s in peers}, "saved peer feature coverage changed")
    columns = []
    for spec in peers:
        saved = state[spec.name]
        require(saved["operation"] == spec.operation, "saved peer operation changed")
        name = spec.sources[0]
        if spec.operation == "rank":
            require(saved["source"] == name, "saved rank source changed")
            values = np.asarray(saved["values"], dtype=np.float64)
            counts = np.asarray(saved["counts"], dtype=np.int64)
            require(
                values.ndim == counts.ndim == 1
                and len(values) == len(counts)
                and bool(np.isfinite(values).all() and (np.diff(values) > 0).all())
                and bool((counts > 0).all() and (np.diff(counts) > 0).all()),
                "invalid saved empirical distribution",
            )
            current = frame[name].cast(pl.Float64).to_numpy()
            distribution = np.r_[0, counts] / max(1, int(counts[-1]) if len(counts) else 0)
            out = distribution[np.searchsorted(values, current, side="right")]
            out[~np.isfinite(current)] = np.nan
            columns.append(pl.Series(spec.name, out, dtype=pl.Float32).fill_nan(None))
            continue
        require(saved["sources"] == list(spec.sources), "saved peer sources changed")
        require(saved["minimum_peer_count"] == 50, "saved peer support rule changed")
        references = saved["reference"]
        if not references:
            columns.append(pl.Series(spec.name, [None] * len(frame), dtype=pl.Float32))
            continue
        reference = pl.DataFrame(references)
        require(reference["key"].n_unique() == len(reference), "duplicate saved peer group")
        require(
            bool(
                reference.select(
                    (
                        (pl.col("count") >= 50)
                        & (pl.col("q25") <= pl.col("median"))
                        & (pl.col("median") <= pl.col("q75"))
                        & pl.all_horizontal(pl.col("q25", "median", "q75").is_finite())
                    ).all()
                ).item()
            ),
            "invalid saved peer summary",
        )
        key = pl.struct(
            pl.col(spec.sources[1]).cast(pl.String).alias("category")
        ).struct.json_encode()
        query = frame.select(key.alias("key"), pl.col(name).cast(pl.Float64).alias("value"))
        query = query.join(reference, on="key", how="left", validate="m:1", maintain_order="left")
        value: pl.Expr
        if spec.operation == "peer_median_ratio":
            value = pl.when(pl.col("median") > 0).then(pl.col("value") / pl.col("median"))
        else:
            require(spec.operation == "peer_iqr_position", "unknown saved peer operation")
            spread = pl.col("q75") - pl.col("q25")
            value = pl.when(spread > 0).then((pl.col("value") - pl.col("median")) / spread)
        value = value.cast(pl.Float32)
        columns.append(
            query.select(pl.when(value.is_finite()).then(value).otherwise(None).alias(spec.name))[
                spec.name
            ]
        )
    return pl.DataFrame(columns)
