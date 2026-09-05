"""Home Credit feature semantics and leakage-safe column classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureKind = Literal["numeric", "categorical", "date", "unsupported"]

CASE_ID = "case_id"
TARGET = "target"
WEEK_NUM = "WEEK_NUM"
MONTH = "MONTH"
DATE_DECISION = "date_decision"
NUM_GROUP1 = "num_group1"
NUM_GROUP2 = "num_group2"

STRUCTURAL_COLUMNS = frozenset(
    {
        CASE_ID,
        TARGET,
        WEEK_NUM,
        MONTH,
        DATE_DECISION,
        NUM_GROUP1,
        NUM_GROUP2,
    }
)

_FORBIDDEN_MODEL_COLUMNS = frozenset(
    {
        TARGET,
        "score",
        "prediction",
        "predicted_target",
    }
)

_NUMERIC_TYPE_PREFIXES = (
    "int",
    "uint",
    "float",
    "double",
    "halffloat",
    "decimal",
)

_CATEGORICAL_TYPE_TOKENS = (
    "string",
    "dictionary<",
    "bool",
)

_TEMPORAL_TYPE_PREFIXES = (
    "date",
    "timestamp",
    "time",
    "duration",
)


@dataclass(frozen=True, slots=True)
class SemanticColumns:
    """Predictor columns grouped by modeling semantics."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    date: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def predictors(self) -> tuple[str, ...]:
        """Return all supported predictors in deterministic order."""
        return tuple(sorted((*self.numeric, *self.categorical, *self.date)))


def is_date_feature(name: str) -> bool:
    """Return whether a Home Credit predictor uses the ``D`` date suffix."""
    return name.endswith("D") and name not in STRUCTURAL_COLUMNS


def normalize_type_name(type_name: str) -> str:
    """Normalize a physical Arrow type string for deterministic classification."""
    return type_name.lower().replace(" ", "")


def classify_feature(name: str, type_name: str) -> FeatureKind:
    """Classify one raw predictor using Home Credit semantics plus physical dtype."""
    if name in STRUCTURAL_COLUMNS:
        return "unsupported"

    if is_date_feature(name):
        return "date"

    normalized = normalize_type_name(type_name)

    if normalized == "null":
        # The tiny official test sample stores some entirely-null predictors with
        # Arrow null dtype. Recover only semantics that are encoded unambiguously in
        # the competition feature suffixes rather than inventing a generic dtype.
        if name.endswith("M"):
            return "categorical"
        if name.endswith(("A", "P", "L")):
            return "numeric"
        return "unsupported"

    if normalized.startswith(_NUMERIC_TYPE_PREFIXES):
        return "numeric"

    if any(token in normalized for token in _CATEGORICAL_TYPE_TOKENS):
        return "categorical"

    if normalized.startswith(_TEMPORAL_TYPE_PREFIXES):
        return "date"

    return "unsupported"


def classify_columns(fields: tuple[tuple[str, str], ...]) -> SemanticColumns:
    """Classify an ordered schema into deterministic semantic predictor groups."""
    groups: dict[FeatureKind, list[str]] = {
        "numeric": [],
        "categorical": [],
        "date": [],
        "unsupported": [],
    }

    for name, type_name in fields:
        if name in STRUCTURAL_COLUMNS:
            continue
        kind = classify_feature(name, type_name)
        groups[kind].append(name)

    return SemanticColumns(
        numeric=tuple(sorted(groups["numeric"])),
        categorical=tuple(sorted(groups["categorical"])),
        date=tuple(sorted(groups["date"])),
        unsupported=tuple(sorted(groups["unsupported"])),
    )


def assert_no_target_leakage(columns: tuple[str, ...], *, context: str) -> None:
    """Reject target/prediction columns from non-base model feature blocks."""
    forbidden = sorted(set(columns).intersection(_FORBIDDEN_MODEL_COLUMNS))
    if forbidden:
        raise ValueError(f"forbidden target/prediction columns in {context}: {forbidden}")


def feature_prefix(family: str, depth: int, *, subgroup: str | None = None) -> str:
    """Return a deterministic collision-resistant prefix for one feature block."""
    base = f"{family}__d{depth}"
    if subgroup is None:
        return base
    return f"{base}__{subgroup}"


def feature_name(
    family: str,
    depth: int,
    column: str,
    operation: str,
    *,
    subgroup: str | None = None,
) -> str:
    """Build one deterministic feature name."""
    return f"{feature_prefix(family, depth, subgroup=subgroup)}__{column}__{operation}"
