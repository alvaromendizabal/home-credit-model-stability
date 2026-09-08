"""Frozen refitting and portable inference contracts, independent of cloud orchestration.

The outer holdout is never an early-stopping set. Encoder state is learned solely
from the explicitly supplied fit population and saved next to the native model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.data import FeatureRef
from home_credit.modeling.encoding import _fit_frequency_maps, _frequency_frame
from home_credit.modeling.selection import require
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject nonportable numbers or unexpected top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {path.name}")
    json.dumps(value, allow_nan=False)
    return dict(value)


def verify_file(path: Path, digest: str) -> None:
    """A filename or completion flag alone is never sufficient evidence of validity."""
    require(path.is_file(), f"required artifact missing: {path.name}")
    require(len(digest) == 64 and sha256_file(path) == digest, f"digest mismatch: {path.name}")


def checked_member(root: Path, name: str) -> Path:
    """Reject absolute names, traversal and symlinks outside a portable model bundle."""
    candidate = Path(name)
    require(not candidate.is_absolute() and ".." not in candidate.parts, "unsafe artifact path")
    resolved = (root / candidate).resolve()
    require(resolved.is_relative_to(root.resolve()), "artifact escapes its bundle")
    return resolved


def validate_features(features: tuple[FeatureRef, ...]) -> None:
    """Reject target, identifiers, absolute time and duplicated predictors."""
    names = [f.name for f in features]
    require(bool(names) and len(names) == len(set(names)), "empty or duplicate feature plan")
    excluded = {"case_id", "target", "score", "prediction", "WEEK_NUM", "MONTH"}
    require(not excluded.intersection(names), "forbidden predictor in release")
    require(not any(n.startswith("base__decision_") for n in names), "absolute time predictor")


def frozen_iterations(state: Mapping[str, Any]) -> int:
    """Derive the predeclared tree count from all five development folds, not the holdout."""
    require(state.get("schema_version") == 1, "unsupported source model state")
    require(state["identity"]["smoke"] is False, "smoke models cannot define a release")
    folds = state["folds"]
    values = []
    for number in range(1, 6):
        record = folds[f"lightgbm:fold_{number}"]
        require(record["fold"] == number, "source fold identity mismatch")
        count = record["metrics"]["best_iteration"]
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0, "bad rounds")
        values.append(count)
    return int(median(values))


def fit_encoder(train: pl.DataFrame, features: tuple[FeatureRef, ...]) -> dict[str, Any]:
    """Save the original training-only frequency representation without refitting at inference."""
    validate_features(features)
    require(train.height > 0, "cannot fit an encoder to an empty population")
    require(set(f.name for f in features) <= set(train.columns), "training feature schema missing")
    for feature in features:
        if feature.categorical:
            require(
                not bool(train.select((pl.col(feature.name).cast(pl.String) == "__HC_MISSING__").any()).item()),
                "reserved missing-category token appears as a real category",
            )
    return {
        "schema_version": 1,
        "method": "training_frequency",
        "fit_rows": train.height,
        "feature_names": [f.name for f in features],
        "frequency_maps": _fit_frequency_maps(train, features),
    }


def transform(
    frame: pl.DataFrame, features: tuple[FeatureRef, ...], encoder: Mapping[str, Any]
) -> NDArray[np.float32]:
    """Apply saved mappings in exact model order; unseen categories get the declared zero code."""
    validate_features(features)
    require(encoder["schema_version"] == 1, "unsupported encoder")
    require(encoder["method"] == "training_frequency", "unknown encoding method")
    require(encoder["feature_names"] == [f.name for f in features], "encoder feature order changed")
    require(set(f.name for f in features) <= set(frame.columns), "inference feature schema missing")
    maps = encoder["frequency_maps"]
    require(set(maps) == {f.name for f in features if f.categorical}, "category mapping mismatch")
    for mapping in maps.values():
        require(
            bool(mapping) and all(math.isfinite(v) and 0 < v <= 1 for v in mapping.values()),
            "invalid saved frequency map",
        )
        require(math.isclose(sum(mapping.values()), 1.0, abs_tol=1e-8), "frequency mass mismatch")
    result = np.asarray(_frequency_frame(frame, features, maps).to_numpy(), dtype=np.float32)
    require(result.shape == (frame.height, len(features)), "encoded shape changed")
    require(not bool(np.isinf(result).any()), "feature value overflows float32")
    return result


def validate_probabilities(values: NDArray[np.float64], rows: int) -> None:
    """Reject ambiguous arrays, truncation, NaNs and out-of-range probabilities."""
    require(values.shape == (rows,), "prediction row count or shape mismatch")
    require(bool(np.isfinite(values).all()), "nonfinite prediction")
    require(bool(((values >= 0) & (values <= 1)).all()), "probability outside [0, 1]")


def fit_frozen_model(
    matrix: NDArray[np.float32],
    target: NDArray[np.int8],
    features: tuple[FeatureRef, ...],
    *,
    params: Mapping[str, Any],
    rounds: int,
    seed: int,
    threads: int,
    path: Path,
    logger: RunLogger,
    check_lease: Callable[[], None],
) -> dict[str, Any]:
    """Fit a fixed-round booster with no validation labels or early-stopping interface."""
    validate_features(features)
    require(matrix.shape == (len(target), len(features)), "training matrix shape mismatch")
    require(set(np.unique(target)) == {0, 1}, "training requires both binary classes")
    require(not bool(np.isinf(matrix).any()), "infinite training feature")
    require(rounds > 0 and threads > 0, "round and thread budgets must be positive")
    keys = (
        "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2", "max_bin",
    )
    parameters = {key: params[key] for key in keys}
    parameters.update(
        objective="binary", metric="None", verbosity=-1, seed=seed,
        feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed,
        num_threads=threads, deterministic=True, force_col_wise=True,
    )
    names = [f.name for f in features]
    dataset = lgb.Dataset(matrix, label=target, feature_name=names, free_raw_data=True)

    def progress(environment: Any) -> None:
        check_lease()
        completed = int(environment.iteration) + 1
        if completed == 1 or completed % 100 == 0 or completed == rounds:
            logger.event("release_tree_progress", completed=completed, total=rounds, model=path.stem)

    with StageTimer(logger, f"fit_{path.stem}", heartbeat_seconds=15):
        model = lgb.train(parameters, dataset, num_boost_round=rounds, callbacks=[progress])
    check_lease()
    atomic_write(path, model.model_to_string().encode("utf-8"))
    restored = lgb.Booster(model_file=str(path))
    require(restored.feature_name() == names, "native model feature order changed")
    probe = matrix[: min(257, len(matrix))]
    expected = np.asarray(model.predict(probe, num_threads=threads), dtype=np.float64)
    actual = np.asarray(restored.predict(probe, num_threads=threads), dtype=np.float64)
    validate_probabilities(actual, len(probe))
    require(bool(np.allclose(expected, actual, rtol=0, atol=1e-12)), "model reload prediction mismatch")
    return {
        "requested_rounds": rounds,
        "actual_rounds": int(restored.current_iteration()),
        "fit_rows": len(target),
        "features": len(features),
        "model_sha256": sha256_file(path),
        "reload_max_absolute_error": float(np.max(np.abs(expected - actual))),
        "early_stopping_used": False,
    }


def predict_components(
    matrix: NDArray[np.float32],
    features: tuple[FeatureRef, ...],
    paths: Mapping[str, Path],
    weights: Mapping[str, float],
    *,
    threads: int = 2,
) -> NDArray[np.float64]:
    """Load verified native components and combine probabilities in a stable order."""
    require(set(paths) == set(weights) and bool(paths), "ensemble component mismatch")
    require(all(math.isfinite(w) and 0 < w <= 1 for w in weights.values()), "invalid blend weight")
    require(math.isclose(sum(weights.values()), 1, abs_tol=1e-12), "blend weights do not sum to one")
    prediction = np.zeros(len(matrix), dtype=np.float64)
    for name in sorted(paths):
        model = lgb.Booster(model_file=str(paths[name]))
        require(model.feature_name() == [f.name for f in features], "model feature order mismatch")
        value = np.asarray(model.predict(matrix, num_threads=threads), dtype=np.float64)
        validate_probabilities(value, len(matrix))
        prediction += weights[name] * value
    validate_probabilities(prediction, len(matrix))
    return prediction


def evaluation_report(
    frame: pl.DataFrame, prediction: NDArray[np.float64], *, expected_weeks: list[int]
) -> dict[str, Any]:
    """Evaluate one frozen population; every expected week must have both classes."""
    validate_probabilities(prediction, frame.height)
    weeks = frame["WEEK_NUM"].to_numpy()
    target = frame["target"].to_numpy()
    require(set(np.unique(weeks)) == set(expected_weeks), "evaluation week coverage changed")
    weekly = []
    for week in expected_weeks:
        mask = weeks == week
        require(set(np.unique(target[mask])) == {0, 1}, f"single-class evaluation week: {week}")
        weekly.append({
            "week": week, "rows": int(mask.sum()), "positives": int(target[mask].sum()),
            "gini": float(2 * roc_auc_score(target[mask], prediction[mask]) - 1),
        })
    values = pd.DataFrame({"prediction": prediction, "target": target})
    values["bin"] = pd.qcut(values["prediction"], 10, duplicates="drop")
    reliability = [
        {"rows": len(group), "mean_prediction": float(group["prediction"].mean()),
         "observed_default_rate": float(group["target"].mean())}
        for _, group in values.groupby("bin", observed=True)
    ]
    if not reliability:
        reliability = [{"rows": len(values), "mean_prediction": float(np.mean(prediction)),
            "observed_default_rate": float(np.mean(target))}]
    return {
        "metrics": evaluate_probabilities(target, prediction, weeks),
        "rows": frame.height, "positive_rate": float(np.mean(target)),
        "weekly": weekly, "reliability": reliability,
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Use one canonical serialization for durable receipts and fingerprints."""
    atomic_write(path, canonical_json_bytes(dict(payload)))
