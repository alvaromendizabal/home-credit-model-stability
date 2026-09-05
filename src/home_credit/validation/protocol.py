"""Frozen temporal validation protocol construction and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """One expanding-window temporal cross-validation fold."""

    fold: int
    train_week_min: int
    train_week_max: int
    validation_week_min: int
    validation_week_max: int

    @property
    def train_weeks(self) -> int:
        """Number of whole weeks in the training window."""
        return self.train_week_max - self.train_week_min + 1

    @property
    def validation_weeks(self) -> int:
        """Number of whole weeks in the validation window."""
        return self.validation_week_max - self.validation_week_min + 1


def build_expanding_folds(
    *,
    development_week_min: int,
    development_week_max: int,
    n_splits: int = 5,
    validation_window_weeks: int = 8,
    min_initial_train_weeks: int = 20,
) -> tuple[TemporalFold, ...]:
    """Build deterministic expanding-window temporal folds."""
    if development_week_max < development_week_min:
        raise ValueError("development week range is invalid")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if validation_window_weeks < 2:
        raise ValueError("validation_window_weeks must be at least 2")

    if min_initial_train_weeks < 1:
        raise ValueError("min_initial_train_weeks must be positive")

    development_weeks = development_week_max - development_week_min + 1

    validation_weeks = n_splits * validation_window_weeks

    initial_train_weeks = development_weeks - validation_weeks

    if initial_train_weeks < min_initial_train_weeks:
        raise ValueError("development range is too short for the requested temporal folds")

    folds: list[TemporalFold] = []

    for fold_index in range(n_splits):
        validation_week_min = (
            development_week_min + initial_train_weeks + fold_index * validation_window_weeks
        )

        validation_week_max = validation_week_min + validation_window_weeks - 1

        folds.append(
            TemporalFold(
                fold=fold_index + 1,
                train_week_min=development_week_min,
                train_week_max=validation_week_min - 1,
                validation_week_min=validation_week_min,
                validation_week_max=validation_week_max,
            )
        )

    result = tuple(folds)

    validate_expanding_folds(
        result,
        development_week_min=development_week_min,
        development_week_max=development_week_max,
    )

    return result


def validate_expanding_folds(
    folds: tuple[TemporalFold, ...],
    *,
    development_week_min: int,
    development_week_max: int,
) -> None:
    """Reject overlap or malformed temporal folds."""
    if not folds:
        raise ValueError("at least one temporal fold is required")

    validation_weeks_seen: set[int] = set()
    previous_train_week_max: int | None = None
    previous_validation_week_max: int | None = None

    for fold in folds:
        if fold.train_week_min != development_week_min:
            raise ValueError("all expanding folds must share the development start week")

        if fold.train_week_max >= fold.validation_week_min:
            raise ValueError("training and validation weeks must not overlap")

        if fold.validation_week_max > development_week_max:
            raise ValueError("validation extends beyond the development range")

        if previous_train_week_max is not None and fold.train_week_max <= previous_train_week_max:
            raise ValueError("training windows must expand monotonically")

        if (
            previous_validation_week_max is not None
            and fold.validation_week_min != previous_validation_week_max + 1
        ):
            raise ValueError("validation windows must be contiguous and non-overlapping")

        current_validation = set(
            range(
                fold.validation_week_min,
                fold.validation_week_max + 1,
            )
        )

        if validation_weeks_seen.intersection(current_validation):
            raise ValueError("validation weeks must not appear in multiple folds")

        validation_weeks_seen.update(current_validation)

        previous_train_week_max = fold.train_week_max

        previous_validation_week_max = fold.validation_week_max

    if folds[-1].validation_week_max != development_week_max:
        raise ValueError("the last validation fold must end at the development boundary")


def fold_payload(
    folds: tuple[TemporalFold, ...],
) -> list[dict[str, int]]:
    """Serialize temporal folds deterministically."""
    return [cast(dict[str, int], asdict(fold)) for fold in folds]


def protocol_sha256(
    payload: Mapping[str, object],
) -> str:
    """Hash canonical protocol content."""
    canonical_payload = dict(payload)
    canonical_payload.pop(
        "protocol_sha256",
        None,
    )

    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def attach_protocol_sha256(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Attach deterministic protocol SHA-256."""
    result = dict(payload)

    result["protocol_sha256"] = protocol_sha256(result)

    return result


def verify_protocol_sha256(
    payload: Mapping[str, object],
) -> bool:
    """Verify the embedded protocol SHA-256."""
    embedded = payload.get("protocol_sha256")

    return isinstance(embedded, str) and embedded == protocol_sha256(payload)


def write_protocol(
    payload: Mapping[str, object],
    output: Path,
) -> None:
    """Atomically write deterministic protocol JSON."""
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_suffix(output.suffix + ".tmp")

    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            output,
        )

    finally:
        temporary.unlink(missing_ok=True)
