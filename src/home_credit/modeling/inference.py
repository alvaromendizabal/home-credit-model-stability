"""Case-bounded raw inference and explicitly owner-authorized submission export.

Uses the same feature operations as training. A ten-row public sample is only a
parity fixture; the scoring population is always read from the supplied raw base.
"""

from __future__ import annotations

import fcntl
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from home_credit.data.loader import RawManifestRecord, S3RawStore
from home_credit.features.builder import (
    FeatureRecipe,
    _base_source,
    _decision_frame,
    _depth_zero_block,
    _history_block,
    _normalized_scan,
    _scan_base,
    group_logical_sources,
)
from home_credit.features.semantics import SemanticColumns
from home_credit.modeling.checkpoints import (
    atomic_write,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from home_credit.modeling.data import FeatureRef
from home_credit.modeling.release import (
    checked_member,
    predict_components,
    read_object,
    save_json,
    transform,
    validate_features,
    validate_probabilities,
    verify_file,
)
from home_credit.modeling.selection import require
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def local_test_records(raw_directory: Path) -> tuple[RawManifestRecord, ...]:
    """Fingerprint every supplied test shard, without accidentally reading training targets."""
    require(raw_directory.is_dir(), "raw test directory does not exist")
    paths = sorted(p for p in raw_directory.rglob("test_*.parquet") if p.is_file())
    require(bool(paths), "no test Parquet files found")
    require(len({p.name for p in paths}) == len(paths), "duplicated raw test filenames")
    return tuple(
        RawManifestRecord(
            p.name, p.relative_to(raw_directory).as_posix(), p.stat().st_size, sha256_file(p)
        )
        for p in paths
    )


def raw_test_frame(
    raw_directory: Path,
    records: tuple[RawManifestRecord, ...],
    features: tuple[FeatureRef, ...],
    schema: Mapping[str, Any],
    recipe_path: Path,
    *,
    case_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Build one bounded scoring batch using the original, frozen feature operations."""
    validate_features(features)
    recipe, digest = FeatureRecipe.load(recipe_path)
    require(digest == schema["recipe_sha256"], "inference feature recipe changed")
    store = S3RawStore(str(raw_directory.resolve()), "local", _filesystem=pafs.LocalFileSystem())
    sources = group_logical_sources(records)
    require(all(source.split == "test" for source in sources), "training input in scoring path")
    base_source = _base_source(sources, "test")
    base = _scan_base(base_source, store, recipe)
    require("target" not in base.collect_schema().names(), "test base contains target")
    require(base.collect_schema()["case_id"].is_integer(), "raw case IDs must be integers")
    base = base.with_columns(pl.col("case_id").cast(pl.Int64, strict=True))
    if case_ids is not None:
        require(bool(case_ids) and len(set(case_ids)) == len(case_ids), "invalid case batch")
        base = base.filter(pl.col("case_id").is_in(pl.Series(case_ids).implode()))
    ids = base.select("case_id").collect()
    require(ids.height > 0 and ids["case_id"].null_count() == 0, "empty or null scoring IDs")
    require(ids["case_id"].n_unique() == ids.height, "duplicate base case IDs")
    require(bool((ids["case_id"] >= 0).all()), "negative scoring case ID")
    if case_ids is not None:
        require(set(ids["case_id"].to_list()) == set(case_ids), "case batch coverage mismatch")
    decision = _decision_frame(base)
    require(
        decision.select(pl.col("_decision_date").null_count()).collect().item() == 0,
        "missing or invalid decision date",
    )
    bounds = (cast(int, ids["case_id"].min()), cast(int, ids["case_id"].max()))
    needed: dict[str, list[str]] = {}
    for feature in features:
        needed.setdefault(feature.block, []).append(feature.name)
    result = ids.lazy()
    found = set()
    for source in sources:
        name = f"{source.family}_depth{source.depth}"
        if name not in needed:
            continue
        require(source.family != "base", "absolute-time base predictors are not supported")
        specification = schema["blocks"][name]
        semantic = SemanticColumns(**{k: tuple(v) for k, v in specification["semantic"].items()})
        observed = set()
        for record in source.records:
            observed.update(
                ds.dataset(str(raw_directory / record.s3_key), format="parquet").schema.names
            )
        required = set(semantic.predictors) - set(specification["allowed_absent_columns"])
        require(required <= observed, f"required raw columns absent from {name}")
        normalized, _ = _normalized_scan(
            source,
            store=store,
            decision_frame=decision,
            recipe=recipe,
            semantic=semantic,
            case_id_bounds=bounds,
        )
        if source.depth == 0:
            counts = normalized.select(
                pl.len().alias("rows"), pl.col("case_id").n_unique().alias("ids")
            ).collect()
            require(counts["rows"][0] == counts["ids"][0], f"duplicate static case IDs: {name}")
            block = _depth_zero_block(normalized, semantic, source=source)
        else:
            block = _history_block(normalized, semantic, source=source, recipe=recipe)
        require(
            set(needed[name]) <= set(block.collect_schema().names()),
            f"feature schema missing: {name}",
        )
        result = result.join(
            block.select(["case_id", *needed[name]]), on="case_id", how="left", validate="1:1"
        )
        found.add(name)
    require(found == set(needed), "required raw table is missing, not merely an empty history")
    output = (
        result.select(["case_id", *[f.name for f in features]])
        .sort("case_id")
        .collect(engine="streaming")
    )
    require(output.height == ids.height, "raw inference silently changed case coverage")
    return output


def inference_code_identity() -> str:
    """Bind batch caches and native bundles to the code that produces their features."""
    package = Path(__file__).resolve().parents[1]
    paths = [
        "modeling/inference.py",
        "modeling/release.py",
        "modeling/encoding.py",
        "modeling/data.py",
        "features/builder.py",
        "features/aggregation.py",
        "features/semantics.py",
    ]
    return sha256_bytes(canonical_json_bytes({name: sha256_file(package / name) for name in paths}))


def load_bundle(
    directory: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], tuple[FeatureRef, ...]]:
    """Verify every portable artifact before loading a native model or encoder."""
    manifest_path = directory / "bundle.json"
    verify_file(manifest_path, expected_sha256)
    bundle = read_object(manifest_path)
    require(bundle["schema_version"] == 1, "unsupported model bundle")
    require(
        bundle["inference_code_sha256"] == inference_code_identity(),
        "inference implementation changed",
    )
    require(bundle["phase"] == "all_labels", "a development model is not the inference release")
    require(bundle["fit_weeks"] == [0, 91], "all-label training scope mismatch")
    require(bundle["calibration"] == "none", "unsupported calibration policy")
    members = {member["path"]: member for member in bundle["files"]}
    require(len(members) == len(bundle["files"]), "duplicate bundle member")
    required = {"features.json", "encoder.json", "raw_schema.json", "feature_recipe.json"}
    required.update(model["path"] for model in bundle["models"].values())
    require(required <= set(members), "required inference artifact is not listed")
    for model in bundle["models"].values():
        require(model["sha256"] == members[model["path"]]["sha256"], "model member digest mismatch")
    for member in bundle["files"]:
        path = checked_member(directory, member["path"])
        verify_file(path, member["sha256"])
        require(path.stat().st_size == member["bytes"], "bundle artifact size mismatch")
    features = tuple(
        FeatureRef(**value) for value in read_object(directory / "features.json")["features"]
    )
    validate_features(features)
    require(len(features) == bundle["feature_count"], "bundle feature count mismatch")
    return bundle, features


def predict_raw(
    bundle_directory: Path,
    raw_directory: Path,
    cache_directory: Path,
    *,
    expected_bundle_sha256: str,
    logger: RunLogger,
    batch_rows: int = 20000,
) -> pd.DataFrame:
    """Resume hash-verified prediction batches. This function never writes a submission CSV."""
    require(batch_rows > 0, "batch size must be positive")
    bundle, features = load_bundle(bundle_directory, expected_sha256=expected_bundle_sha256)
    with StageTimer(logger, "fingerprint_raw_scoring_inputs", heartbeat_seconds=15):
        records = local_test_records(raw_directory)
    base_source = _base_source(group_logical_sources(records), "test")
    ids = pl.read_parquet(raw_directory / base_source.records[0].s3_key, columns=["case_id"])[
        "case_id"
    ]
    require(ids.dtype.is_integer(), "raw case IDs must be integers")
    require(ids.null_count() == 0 and ids.n_unique() == len(ids), "null or duplicate raw case ID")
    require(len(ids) > 0, "empty test population")
    ordered = ids.sort().to_list()
    identity = {
        "bundle_sha256": expected_bundle_sha256,
        "inputs": {record.file: record.sha256 for record in records},
        "batch_rows": batch_rows,
    }
    key = sha256_bytes(canonical_json_bytes(identity))
    work = cache_directory / key
    work.mkdir(parents=True, exist_ok=True)
    encoder = read_object(bundle_directory / "encoder.json")
    schema = read_object(bundle_directory / "raw_schema.json")
    models = {
        name: checked_member(bundle_directory, member["path"])
        for name, member in bundle["models"].items()
    }
    with (work / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _predict_batches(
            bundle_directory, raw_directory, records, features, schema, encoder,
            models, bundle["weights"], work, key, ordered, batch_rows, logger,
        )
    with StageTimer(logger, "verify_raw_inputs_unchanged", heartbeat_seconds=15):
        require(
            all(sha256_file(raw_directory / r.s3_key) == r.sha256 for r in records),
            "raw inputs changed during inference",
        )
    return result


def _predict_batches(
    bundle_directory: Path,
    raw_directory: Path,
    records: tuple[RawManifestRecord, ...],
    features: tuple[FeatureRef, ...],
    schema: dict[str, Any],
    encoder: dict[str, Any],
    models: dict[str, Path],
    weights: dict[str, float],
    work: Path,
    key: str,
    ordered: list[int],
    batch_rows: int,
    logger: RunLogger,
) -> pd.DataFrame:
    """The caller holds the local writer lock throughout checkpoint publication."""
    outputs = []
    total = math_ceil_div(len(ordered), batch_rows)
    for number, start in enumerate(range(0, len(ordered), batch_rows), 1):
        cases = ordered[start : start + batch_rows]
        path = work / f"part_{number:06d}.parquet"
        receipt_path = work / f"part_{number:06d}.json"
        if path.is_file() and receipt_path.is_file():
            receipt = read_object(receipt_path)
            require(receipt["identity"] == key, "prediction checkpoint lineage mismatch")
            verify_file(path, receipt["sha256"])
            predicted = pd.read_parquet(path)
            logger.event("inference_batch_reused", completed=number, total=total)
        else:
            with StageTimer(logger, f"inference_batch_{number}", heartbeat_seconds=15):
                frame = raw_test_frame(
                    raw_directory, records, features, schema,
                    bundle_directory / "feature_recipe.json", case_ids=cases,
                )
                prediction = predict_components(
                    transform(frame, features, encoder), features, models, weights, threads=2
                )
                predicted = pd.DataFrame({"case_id": frame["case_id"].to_numpy(), "score": prediction})
                temporary = path.with_suffix(".parquet.download")
                predicted.to_parquet(temporary, index=False)
                temporary.replace(path)
                save_json(receipt_path, {"identity": key, "sha256": sha256_file(path), "rows": len(cases)})
            logger.event("inference_batch_completed", completed=number, total=total, rows=len(cases))
        require(predicted["case_id"].tolist() == cases, "cached prediction order or coverage changed")
        validate_probabilities(predicted["score"].to_numpy(dtype=np.float64), len(cases))
        outputs.append(predicted)
    return pd.concat(outputs, ignore_index=True)


def math_ceil_div(numerator: int, denominator: int) -> int:
    """Integer-only batch counting."""
    return (numerator + denominator - 1) // denominator


def submission_frame(sample: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Align explicitly to the supplied sample, with no silently discarded or duplicated cases."""
    require(sample.columns.tolist() == ["case_id", "score"], "sample schema must be case_id,score")
    require(predictions.columns.tolist() == ["case_id", "score"], "prediction schema mismatch")
    for frame in (sample, predictions):
        require(len(frame) > 0, "empty submission population")
        require(pd.api.types.is_integer_dtype(frame["case_id"].dtype), "case IDs must be integer")
        require(bool(frame["case_id"].notna().all()) and frame["case_id"].is_unique, "null or duplicate case IDs")
        require(bool((frame["case_id"] >= 0).all()), "negative case ID")
    require(len(sample) == len(predictions), "submission row count mismatch")
    require(set(sample["case_id"]) == set(predictions["case_id"]), "submission case coverage mismatch")
    values = predictions.set_index("case_id")["score"].reindex(sample["case_id"]).to_numpy(dtype=np.float64)
    validate_probabilities(values, len(sample))
    return pd.DataFrame({"case_id": sample["case_id"].to_numpy(), "score": values})


def export_submission(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    destination: Path,
    *,
    lineage: Mapping[str, Any],
    owner_confirmed: bool = False,
) -> Path:
    """Atomically save and read back an owner-requested CSV; never contact Kaggle."""
    require(owner_confirmed is True, "submission generation requires explicit owner confirmation")
    require(destination.suffix == ".csv", "submission destination must be a CSV")
    require(
        all(isinstance(lineage.get(name), str) and len(lineage[name]) == 64
            and set(lineage[name]) <= set("0123456789abcdef")
            for name in ("bundle_sha256", "input_sha256")),
        "submission lineage missing",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _save_submission(sample, predictions, destination, lineage)


def _save_submission(
    sample: pd.DataFrame, predictions: pd.DataFrame, destination: Path,
    lineage: Mapping[str, Any],
) -> Path:
    """Write under the caller's exclusive export lock and preserve any different prior CSV."""
    result = submission_frame(sample, predictions)
    payload = result.to_csv(index=False, float_format="%.17g").encode("utf-8")
    if destination.exists():
        require(destination.read_bytes() == payload, "refuse to overwrite a different existing submission")
    else:
        atomic_write(destination, payload)
    restored = pd.read_csv(destination, float_precision="round_trip")
    checked = submission_frame(sample, restored)
    require(checked["case_id"].tolist() == sample["case_id"].tolist(), "saved CSV order mismatch")
    require(bool(np.allclose(checked["score"], result["score"], rtol=0, atol=1e-15)), "CSV probability roundtrip mismatch")
    save_json(destination.with_suffix(".json"), {
        "schema_version": 1, "rows": len(result), "columns": result.columns.tolist(),
        "sha256": sha256_file(destination), "lineage": dict(lineage), "kaggle_submitted": False,
    })
    return destination
