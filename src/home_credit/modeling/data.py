"""Feature-snapshot indexing and leakage-safe fold materialization."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

CASE_ID = "case_id"
TARGET = "target"
WEEK_NUM = "WEEK_NUM"


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """One model candidate feature and its source block."""

    name: str
    block: str
    family: str
    depth: int
    dtype: str
    categorical: bool


@dataclass(frozen=True, slots=True)
class FeatureBlockRef:
    """One local case-level feature block from the frozen Phase 4 snapshot."""

    split: str
    family: str
    depth: int
    path: Path
    output_sha256: str
    rows: int
    feature_columns: int

    @property
    def name(self) -> str:
        """Return canonical logical block name."""
        return f"{self.family}_depth{self.depth}"


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Validated local view of the immutable case-level feature snapshot."""

    root: Path
    manifest_sha256: str
    protocol_sha256: str
    recipe_sha256: str
    execution_sha256: str
    feature_git_commit: str
    train_blocks: tuple[FeatureBlockRef, ...]
    test_blocks: tuple[FeatureBlockRef, ...]
    features: tuple[FeatureRef, ...]

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_manifest_sha256: str,
        expected_protocol_sha256: str,
        verify_hashes: bool = True,
    ) -> FeatureSnapshot:
        """Load and verify one local Phase 4 snapshot."""
        resolved_root = root.resolve()
        manifest_path = resolved_root / "feature_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature manifest missing: {manifest_path}")

        manifest_bytes = manifest_path.read_bytes()
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "feature manifest SHA-256 mismatch: "
                f"expected={expected_manifest_sha256} "
                f"actual={actual_manifest_sha256}"
            )

        raw = json.loads(manifest_bytes)
        if not isinstance(raw, dict):
            raise ValueError("feature manifest must be a JSON object")
        manifest = cast(dict[str, Any], raw)

        protocol_sha256 = _required_str(manifest, "validation_protocol_sha256")
        if protocol_sha256 != expected_protocol_sha256:
            raise ValueError(
                "feature snapshot validation protocol mismatch: "
                f"expected={expected_protocol_sha256} actual={protocol_sha256}"
            )

        block_payloads = manifest.get("blocks")
        if not isinstance(block_payloads, list):
            raise ValueError("feature manifest blocks must be a list")

        train_blocks: list[FeatureBlockRef] = []
        test_blocks: list[FeatureBlockRef] = []

        for raw_block in block_payloads:
            if not isinstance(raw_block, dict):
                raise ValueError("feature manifest block must be an object")
            block = cast(dict[str, Any], raw_block)
            split = _required_str(block, "split")
            family = _required_str(block, "family")
            depth = _required_int(block, "depth")
            output_sha256 = _required_str(block, "output_sha256")
            rows = _required_int(block, "rows")
            feature_columns = _required_int(block, "feature_columns")

            path = resolved_root / "blocks" / split / f"{family}_depth{depth}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"feature block missing: {path}")
            if verify_hashes:
                actual = _sha256_file(path)
                if actual != output_sha256:
                    raise ValueError(
                        "feature block SHA-256 mismatch: "
                        f"path={path} expected={output_sha256} actual={actual}"
                    )

            ref = FeatureBlockRef(
                split=split,
                family=family,
                depth=depth,
                path=path,
                output_sha256=output_sha256,
                rows=rows,
                feature_columns=feature_columns,
            )
            if split == "train":
                train_blocks.append(ref)
            elif split == "test":
                test_blocks.append(ref)
            else:
                raise ValueError(f"unsupported feature split: {split}")

        train_blocks.sort(key=lambda item: (item.family, item.depth))
        test_blocks.sort(key=lambda item: (item.family, item.depth))
        if len(train_blocks) != 17 or len(test_blocks) != 17:
            raise ValueError(
                "feature snapshot must contain 17 train and 17 test blocks: "
                f"train={len(train_blocks)} test={len(test_blocks)}"
            )

        train_keys = {(item.family, item.depth) for item in train_blocks}
        test_keys = {(item.family, item.depth) for item in test_blocks}
        if train_keys != test_keys:
            raise ValueError("train/test logical feature blocks do not match")

        features = _index_train_features(tuple(train_blocks))
        return cls(
            root=resolved_root,
            manifest_sha256=actual_manifest_sha256,
            protocol_sha256=protocol_sha256,
            recipe_sha256=_required_str(manifest, "feature_recipe_sha256"),
            execution_sha256=_required_str(manifest, "feature_execution_sha256"),
            feature_git_commit=_required_str(manifest, "git_commit"),
            train_blocks=tuple(train_blocks),
            test_blocks=tuple(test_blocks),
            features=features,
        )

    def candidate_features(self, *, excluded: frozenset[str]) -> tuple[FeatureRef, ...]:
        """Return deterministic predictor candidates after explicit exclusions."""
        return tuple(feature for feature in self.features if feature.name not in excluded)

    def train_block(self, family: str, depth: int) -> FeatureBlockRef:
        """Return one train block by logical source."""
        for block in self.train_blocks:
            if block.family == family and block.depth == depth:
                return block
        raise KeyError((family, depth))


def load_feature_frame(
    snapshot: FeatureSnapshot,
    features: tuple[FeatureRef, ...],
    *,
    week_min: int,
    week_max: int,
    max_rows: int | None,
    seed: int,
) -> pl.DataFrame:
    """Materialize selected train features for a leakage-safe week interval."""
    if week_min < 0 or week_max < week_min:
        raise ValueError("invalid week interval")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when supplied")

    selected_by_block: dict[str, list[FeatureRef]] = defaultdict(list)
    seen_names: set[str] = set()
    for feature in features:
        if feature.name in seen_names:
            raise ValueError(f"duplicate selected feature: {feature.name}")
        seen_names.add(feature.name)
        selected_by_block[feature.block].append(feature)

    base_block = snapshot.train_block("base", 0)
    base_features = selected_by_block.pop(base_block.name, [])
    base_columns = [CASE_ID, TARGET, WEEK_NUM]
    base_columns.extend(feature.name for feature in base_features)

    frame = (
        pl.scan_parquet(base_block.path)
        .select(base_columns)
        .filter(pl.col(WEEK_NUM).is_between(week_min, week_max, closed="both"))
    )

    if max_rows is not None:
        # Deterministic pseudo-random ordering without depending on global RNG state.
        sample_key = (
            (pl.col(CASE_ID).cast(pl.Int64) * 1_103_515_245 + int(seed)) % 2_147_483_647
        ).alias("__sample_key")
        frame = (
            frame.with_columns(sample_key).sort("__sample_key").head(max_rows).drop("__sample_key")
        )

    for block in snapshot.train_blocks:
        if block.family == "base":
            continue
        block_features = selected_by_block.get(block.name)
        if not block_features:
            continue
        columns = [CASE_ID]
        columns.extend(feature.name for feature in block_features)
        right = pl.scan_parquet(block.path).select(columns)
        frame = frame.join(right, on=CASE_ID, how="left", validate="1:1")

    result = frame.sort(CASE_ID).collect(engine="streaming")
    _validate_materialized_frame(result, week_min=week_min, week_max=week_max)
    return result


def load_fold_frames(
    snapshot: FeatureSnapshot,
    features: tuple[FeatureRef, ...],
    *,
    train_week_min: int,
    train_week_max: int,
    validation_week_min: int,
    validation_week_max: int,
    seed: int,
    train_row_cap: int | None = None,
    validation_row_cap: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one frozen temporal fold while never crossing the validation boundary."""
    if train_week_max >= validation_week_min:
        raise ValueError("train and validation weeks overlap")

    if train_row_cap is not None or validation_row_cap is not None:
        train = load_feature_frame(
            snapshot,
            features,
            week_min=train_week_min,
            week_max=train_week_max,
            max_rows=train_row_cap,
            seed=seed,
        )
        validation = load_feature_frame(
            snapshot,
            features,
            week_min=validation_week_min,
            week_max=validation_week_max,
            max_rows=validation_row_cap,
            seed=seed + 1,
        )
        return train, validation

    combined = load_feature_frame(
        snapshot,
        features,
        week_min=train_week_min,
        week_max=validation_week_max,
        max_rows=None,
        seed=seed,
    )
    train = combined.filter(pl.col(WEEK_NUM) <= train_week_max)
    validation = combined.filter(
        pl.col(WEEK_NUM).is_between(
            validation_week_min,
            validation_week_max,
            closed="both",
        )
    )
    _validate_materialized_frame(
        train,
        week_min=train_week_min,
        week_max=train_week_max,
    )
    _validate_materialized_frame(
        validation,
        week_min=validation_week_min,
        week_max=validation_week_max,
    )
    return train, validation


def _index_train_features(blocks: tuple[FeatureBlockRef, ...]) -> tuple[FeatureRef, ...]:
    features: list[FeatureRef] = []
    names: set[str] = set()

    for block in blocks:
        schema = pq.ParquetFile(block.path).schema_arrow
        for field in schema:
            if field.name in {CASE_ID, TARGET}:
                continue
            if field.name in names:
                raise ValueError(f"feature name collision: {field.name}")
            names.add(field.name)
            features.append(
                FeatureRef(
                    name=field.name,
                    block=block.name,
                    family=block.family,
                    depth=block.depth,
                    dtype=str(field.type),
                    categorical=_is_categorical(field.type),
                )
            )

    return tuple(sorted(features, key=lambda item: (item.block, item.name)))


def _is_categorical(data_type: Any) -> bool:
    return bool(
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_dictionary(data_type)
    )


def _validate_materialized_frame(
    frame: pl.DataFrame,
    *,
    week_min: int,
    week_max: int,
) -> None:
    required = {CASE_ID, TARGET, WEEK_NUM}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"materialized model frame is missing columns: {missing}")
    if frame.height == 0:
        raise ValueError("materialized model frame is empty")
    if frame.get_column(CASE_ID).null_count() != 0:
        raise ValueError("materialized model frame contains null case_id")
    if frame.get_column(CASE_ID).n_unique() != frame.height:
        raise ValueError("materialized model frame contains duplicate case_id")
    weeks = frame.get_column(WEEK_NUM)

    if weeks.null_count() != 0:
        raise ValueError("materialized model frame contains null WEEK_NUM")

    observed_min_value = weeks.min()
    observed_max_value = weeks.max()

    if not isinstance(observed_min_value, int) or not isinstance(observed_max_value, int):
        raise ValueError("materialized model frame WEEK_NUM extrema must be integers")

    observed_min = observed_min_value
    observed_max = observed_max_value
    if observed_min < week_min or observed_max > week_max:
        raise ValueError(
            "materialized model frame escaped week interval: "
            f"observed={observed_min}-{observed_max} expected={week_min}-{week_max}"
        )
    target = frame.get_column(TARGET)
    if target.null_count() != 0:
        raise ValueError("materialized model frame contains null target")
    target_values = set(int(value) for value in target.unique().to_list())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"target must be binary: {sorted(target_values)}")


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"manifest field {key!r} must be an integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
