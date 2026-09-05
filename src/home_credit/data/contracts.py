"""Structural contracts for the immutable Home Credit raw-data catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from home_credit.data.catalog import CatalogEntry, summarize_tables

_ALLOWED_EMPTY_TEST_FILES = frozenset(
    {
        "parquet_files/test/test_tax_registry_c_1.parquet",
    }
)


@dataclass(frozen=True, slots=True)
class ContractReport:
    """Serializable result of raw-catalog contract validation."""

    passed: bool
    checks: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _logical_type_family(type_name: str) -> str:
    """Map Arrow physical types to stable modeling-level type families."""
    value = type_name.lower().replace(" ", "")

    if value == "null":
        return "null"

    if value.startswith(
        (
            "int",
            "uint",
            "float",
            "double",
            "halffloat",
            "decimal",
        )
    ):
        return "numeric"

    if "dictionary<" in value or "string" in value:
        return "categorical"

    if value.startswith("bool"):
        return "boolean"

    if value.startswith(
        (
            "date",
            "timestamp",
            "time",
            "duration",
        )
    ):
        return "temporal"

    if "binary" in value:
        return "binary"

    if value.startswith(
        (
            "list<",
            "large_list<",
            "fixed_size_list<",
        )
    ):
        return "list"

    if value.startswith("struct<"):
        return "struct"

    if value.startswith("map<"):
        return "map"

    return f"exact:{value}"


def _semantic_type_family(
    column_name: str,
    type_name: str,
) -> str:
    """Return the modeling-level type family for a Home Credit feature.

    Home Credit predictor names ending in ``D`` are transformed date
    features. Their physical Arrow representation may differ between the
    large training data and tiny competition test sample, so the semantic
    feature contract takes precedence over storage representation.
    """
    if column_name.endswith("D"):
        return "transformed_date"

    return _logical_type_family(type_name)


def _types_compatible(
    column_name: str,
    left: str,
    right: str,
) -> bool:
    """Return whether two physical types represent compatible features."""
    if left == right:
        return True

    left_family = _semantic_type_family(column_name, left)
    right_family = _semantic_type_family(column_name, right)

    if "null" in {left_family, right_family}:
        return True

    return left_family == right_family


def _schema_map(
    entry: CatalogEntry,
    *,
    exclude: frozenset[str] | None = None,
) -> dict[str, str]:
    """Return the ordered catalog schema as a name-to-type mapping."""
    excluded = frozenset() if exclude is None else exclude

    return {
        name: type_name
        for name, type_name in zip(
            entry.column_names,
            entry.column_types,
            strict=True,
        )
        if name not in excluded
    }


def _schema_differences(
    left: CatalogEntry,
    right: CatalogEntry,
    *,
    exclude: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Describe genuine logical incompatibilities between two schemas."""
    left_schema = _schema_map(left, exclude=exclude)
    right_schema = _schema_map(right, exclude=exclude)

    differences: list[str] = []

    left_names = set(left_schema)
    right_names = set(right_schema)

    for name in sorted(left_names - right_names):
        differences.append(f"missing_right:{name}")

    for name in sorted(right_names - left_names):
        differences.append(f"missing_left:{name}")

    for name in sorted(left_names & right_names):
        left_type = left_schema[name]
        right_type = right_schema[name]

        if not _types_compatible(name, left_type, right_type):
            differences.append(f"type:{name}:{left_type}->{right_type}")

    return tuple(differences)


def _physical_signature(
    entry: CatalogEntry,
    *,
    exclude: frozenset[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return deterministic physical column/type signature."""
    return tuple(sorted(_schema_map(entry, exclude=exclude).items()))


def validate_catalog(
    entries: Sequence[CatalogEntry],
) -> ContractReport:
    """Validate raw-data structure before feature engineering.

    Physical Arrow differences are retained as provenance warnings.
    Modeling contracts fail only for genuine structural incompatibility.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1

        if not condition:
            errors.append(message)

    require(
        bool(entries),
        "catalog must contain at least one Parquet file",
    )

    require(
        len({entry.file for entry in entries}) == len(entries),
        "catalog files must be unique",
    )

    require(
        all(
            entry.columns == len(entry.column_names) == len(entry.column_types) for entry in entries
        ),
        "catalog column counts must match recorded schemas",
    )

    require(
        all(len(set(entry.column_names)) == len(entry.column_names) for entry in entries),
        "Parquet schemas must not contain duplicate columns",
    )

    require(
        all(entry.case_id_present for entry in entries),
        "every Parquet source must contain case_id",
    )

    require(
        all(entry.case_id_null_count == 0 for entry in entries),
        "case_id must never be null",
    )

    require(
        all(entry.columns > 0 for entry in entries),
        "every Parquet source must expose a schema",
    )

    require(
        all(entry.rows > 0 for entry in entries if entry.split == "train"),
        "all train Parquet sources must be non-empty",
    )

    unexpected_empty_test = [
        entry.file
        for entry in entries
        if entry.split == "test" and entry.rows == 0 and entry.file not in _ALLOWED_EMPTY_TEST_FILES
    ]

    require(
        not unexpected_empty_test,
        ("unexpected empty test Parquet source(s): " + ", ".join(sorted(unexpected_empty_test))),
    )

    for entry in entries:
        if entry.file in _ALLOWED_EMPTY_TEST_FILES:
            require(
                entry.rows == 0,
                (f"known empty official test table unexpectedly contains rows: {entry.file}"),
            )

            warnings.append(f"known empty official test Parquet accepted: {entry.file}")

    train_base = [entry for entry in entries if entry.split == "train" and entry.family == "base"]

    test_base = [entry for entry in entries if entry.split == "test" and entry.family == "base"]

    require(
        len(train_base) == 1,
        "catalog must contain exactly one train base table",
    )

    require(
        len(test_base) == 1,
        "catalog must contain exactly one test base table",
    )

    if len(train_base) == 1:
        require(
            "target" in train_base[0].column_names,
            "train base table must contain target",
        )

        require(
            "WEEK_NUM" in train_base[0].column_names,
            "train base table must contain WEEK_NUM",
        )

    if len(test_base) == 1:
        require(
            "target" not in test_base[0].column_names,
            "test base table must not contain target",
        )

        require(
            "WEEK_NUM" in test_base[0].column_names,
            "test base table must contain WEEK_NUM",
        )

    depth_zero = [entry for entry in entries if entry.depth == 0]

    require(
        all(entry.max_rows_per_case <= 1 for entry in depth_zero),
        ("depth-0 tables must contain at most one row per case_id within each source file"),
    )

    grouped: dict[
        tuple[str, str, int],
        list[CatalogEntry],
    ] = {}

    for entry in entries:
        grouped.setdefault(
            (
                entry.split,
                entry.family,
                entry.depth,
            ),
            [],
        ).append(entry)

    for (split, family, depth), group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda entry: entry.file,
        )

        reference = ordered[0]

        for candidate in ordered[1:]:
            differences = _schema_differences(
                reference,
                candidate,
            )

            require(
                not differences,
                (
                    "shards are logically schema-incompatible within "
                    f"{split}_{family}_depth{depth}: "
                    f"{reference.file} vs {candidate.file}: " + ", ".join(differences)
                ),
            )

        physical_fingerprints = {entry.schema_sha256 for entry in ordered}

        if len(physical_fingerprints) > 1:
            warnings.append(
                "physical schema variants observed within "
                f"{split}_{family}_depth{depth}; "
                "logical schema is compatible"
            )

    by_logical: dict[
        tuple[str, int],
        dict[str, list[CatalogEntry]],
    ] = {}

    for entry in entries:
        by_logical.setdefault(
            (
                entry.family,
                entry.depth,
            ),
            {},
        ).setdefault(
            entry.split,
            [],
        ).append(entry)

    for (family, depth), split_groups in sorted(by_logical.items()):
        train_group = sorted(
            split_groups.get("train", []),
            key=lambda entry: entry.file,
        )

        test_group = sorted(
            split_groups.get("test", []),
            key=lambda entry: entry.file,
        )

        if not train_group or not test_group:
            warnings.append(f"logical table missing one split: {family}_depth{depth}")
            continue

        excluded = frozenset({"target"}) if family == "base" else frozenset()

        train_reference = train_group[0]
        test_reference = test_group[0]

        differences = _schema_differences(
            train_reference,
            test_reference,
            exclude=excluded,
        )

        require(
            not differences,
            (
                "train/test logical schema mismatch for "
                f"{family}_depth{depth}: " + ", ".join(differences)
            ),
        )

        if not differences and _physical_signature(
            train_reference,
            exclude=excluded,
        ) != _physical_signature(
            test_reference,
            exclude=excluded,
        ):
            warnings.append(
                "train/test physical type drift observed for "
                f"{family}_depth{depth}; "
                "logical schema is compatible"
            )

    summaries = summarize_tables(entries)

    require(
        bool(summaries),
        "table summaries must not be empty",
    )

    return ContractReport(
        passed=not errors,
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
