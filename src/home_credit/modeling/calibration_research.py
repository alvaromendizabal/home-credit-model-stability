"""Earlier-to-later development calibration with portable parameter checkpoints."""

from __future__ import annotations

import copy
import fcntl
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.special import expit, logit  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.acceptance import require
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.release import read_object, save_json
from home_credit.modeling.release_workflow import ReleaseLogger, publish_verified, restore_member
from home_credit.modeling.selection import AlignedPredictions, align_predictions
from home_credit.observability.runtime import StageTimer


def validate_policy(plan: dict[str, Any]) -> None:
    require(plan["schema_version"] == 1, "unsupported calibration policy")
    require(plan["methods"] == ["uncalibrated", "sigmoid", "isotonic"], "method budget changed")
    require(plan["evaluation_folds"] == [2, 3, 4, 5], "calibration window changed")
    require(
        plan["fit_policy"] == "strictly_earlier_development_oof_folds", "unsafe calibration fit"
    )
    require(
        plan["max_calibrator_fits"] == 8 and plan["new_base_model_fits"] == 0, "fit budget changed"
    )
    require(
        plan["holdout_evaluation_allowed"] is False and plan["release_promotion_allowed"] is False,
        "observed holdout and release are outside calibration research",
    )
    require(plan["weights"] == {"tuned": 0.9, "original": 0.1}, "frozen reference blend changed")
    require(plan["probability_clip"] == 1e-7, "numeric policy changed")


def probability(values: Any) -> NDArray[np.float64]:
    p = np.asarray(values, dtype=np.float64)
    require(p.ndim == 1 and len(p) > 0, "empty or non-vector probabilities")
    require(bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()), "invalid probabilities")
    return p


def fit_calibrator(p: Any, target: Any, method: str) -> dict[str, Any]:
    """Fit only the supplied past predictions; save JSON, never executable pickle."""
    values = probability(p)
    y = np.asarray(target)
    require(y.shape == values.shape and set(np.unique(y)) == {0, 1}, "invalid calibration labels")
    if method == "uncalibrated":
        return {"method": method}
    if method == "sigmoid":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, tol=1e-9)
        model.fit(logit(np.clip(values, 1e-7, 1 - 1e-7)).reshape(-1, 1), y)
        coefficient = float(model.coef_[0, 0])
        require(coefficient > 0, "sigmoid calibration must preserve ranking")
        return {
            "method": method,
            "coefficient": coefficient,
            "intercept": float(model.intercept_[0]),
        }
    if method == "isotonic":
        fitted = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(values, y)
        return {
            "method": method,
            "x": fitted.X_thresholds_.tolist(),
            "y": fitted.y_thresholds_.tolist(),
        }
    raise ValueError("unknown calibration method")


def apply_calibrator(p: Any, model: dict[str, Any]) -> NDArray[np.float64]:
    values = probability(p)
    if model["method"] == "uncalibrated":
        return values.copy()
    if model["method"] == "sigmoid":
        require(
            np.isfinite(model["coefficient"])
            and model["coefficient"] > 0
            and np.isfinite(model["intercept"]),
            "invalid sigmoid parameters",
        )
        return probability(
            expit(
                model["coefficient"] * logit(np.clip(values, 1e-7, 1 - 1e-7)) + model["intercept"]
            )
        )
    require(model["method"] == "isotonic", "unknown calibration parameters")
    x, y = np.asarray(model["x"], dtype=float), np.asarray(model["y"], dtype=float)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0, "invalid isotonic shape")
    require(
        bool(
            np.isfinite(x).all()
            and np.isfinite(y).all()
            and (np.diff(x) > 0).all()
            and (np.diff(y) >= 0).all()
            and ((y >= 0) & (y <= 1)).all()
        ),
        "invalid isotonic thresholds",
    )
    return probability(np.interp(values, x, y))


def fit_prior_folds(
    data: AlignedPredictions, fold: int, method: str
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """The calibrator never receives current/future labels or feature distributions."""
    require(fold in (2, 3, 4, 5), "calibration requires an earlier development fold")
    require(int(data.week.max()) <= 72, "calibration cannot access the observed holdout")
    train, valid = data.fold < fold, data.fold == fold
    require(bool(train.any() and valid.any()), "missing calibration population")
    require(int(data.week[train].max()) < int(data.week[valid].min()), "calibration time overlap")
    blend = 0.9 * data.predictions["tuned"] + 0.1 * data.predictions["original"]
    model = fit_calibrator(blend[train], data.target[train], method)
    prediction = apply_calibrator(blend[valid], model)
    record = {
        "method": method,
        "fold": fold,
        "fit_rows": int(train.sum()),
        "evaluation_rows": int(valid.sum()),
        "fit_week_max": int(data.week[train].max()),
        "evaluation_week_min": int(data.week[valid].min()),
        "evaluation_week_max": int(data.week[valid].max()),
        "model": model,
        "metrics": evaluate_probabilities(data.target[valid], prediction, data.week[valid]),
    }
    return record, prediction


def reliability(frame: pl.DataFrame, edges: list[float]) -> list[dict[str, Any]]:
    rows = []
    for number, (lower, upper) in enumerate(pairwise(edges)):
        mask = (pl.col("prediction") >= lower) & (
            (pl.col("prediction") <= upper)
            if number == len(edges) - 2
            else (pl.col("prediction") < upper)
        )
        part = frame.filter(mask)
        if len(part):
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "rows": len(part),
                    "mean_prediction": float(part["prediction"].to_numpy().mean()),
                    "observed_default_rate": float(part["target"].to_numpy().mean()),
                }
            )
    require(sum(r["rows"] for r in rows) == len(frame), "reliability coverage mismatch")
    return rows


def run(root: Path, bucket: str) -> dict[str, Any]:
    plan = read_object(root / "configs/calibration_research.json")
    validate_policy(plan)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    identity = {
        "source_commit": source,
        "plan_sha256": sha256_file(root / "configs/calibration_research.json"),
        "lock_sha256": sha256_file(root / "uv.lock"),
    }
    key = sha256_bytes(canonical_json_bytes(identity))
    work = root / "artifacts/calibration_research" / key
    work.mkdir(parents=True, exist_ok=True)
    logger = ReleaseLogger("calibration-research", root / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    store = ExperimentStore(
        client, bucket, f"home-credit-model-stability/calibration-research/{key}", work, logger
    )
    with (work / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with WriterLease(store) as lease:
            state = store.restore(identity)
            if state is None:
                state = {
                    "schema_version": 1,
                    "identity": identity,
                    "revision": 0,
                    "trials": [],
                    "complete": False,
                }
                store.commit(state)
            for record in state["trials"]:
                restore_member(store, record["checkpoint"])
                restore_member(store, record["predictions"])
            if state["complete"]:
                report = read_object(restore_member(store, state["report"]))
                logger.event("calibration_research_reused", new_calibrator_fits=0)
                return report
            frames = {}
            for name, member in plan["sources"].items():
                path = work / "inputs" / f"{name}.parquet"
                store.download(member["object_key"], path, member["sha256"])
                require(path.stat().st_size == member["bytes"], "input byte length mismatch")
                frames[name] = pd.read_parquet(path)
            windows = read_object(root / "configs/validation_protocol.json")["inner_temporal_cv"][
                "folds"
            ]
            data = align_predictions(frames, windows, expected_rows=727187)
            completed = {(r["fold"], r["method"]) for r in state["trials"]}
            require(len(completed) == len(state["trials"]), "duplicate calibration checkpoint")
            for fold in plan["evaluation_folds"]:
                for method in plan["methods"]:
                    lease.check()
                    if (fold, method) in completed:
                        continue
                    with StageTimer(logger, f"calibrate_{method}_{fold}", heartbeat_seconds=15):
                        record, prediction = fit_prior_folds(data, fold, method)
                        path = work / "folds" / f"{method}_{fold}.json"
                        save_json(path, record)
                        predictions = path.with_suffix(".parquet")
                        valid = data.fold == fold
                        pl.DataFrame(
                            {
                                "case_id": data.case_id[valid],
                                "WEEK_NUM": data.week[valid],
                                "target": data.target[valid],
                                "prediction": prediction,
                            }
                        ).write_parquet(predictions)
                        # Reload portable parameters before accepting this fold.
                        loaded = read_object(path)
                        blend = (
                            0.9 * data.predictions["tuned"][valid]
                            + 0.1 * data.predictions["original"][valid]
                        )
                        require(
                            bool(
                                np.array_equal(apply_calibrator(blend, loaded["model"]), prediction)
                            ),
                            "calibrator parameter replay mismatch",
                        )
                        record["checkpoint"] = publish_verified(
                            store, path, path.relative_to(work).as_posix()
                        )
                        record["predictions"] = publish_verified(
                            store, predictions, predictions.relative_to(work).as_posix()
                        )
                    updated = copy.deepcopy(state)
                    updated["trials"].append(record)
                    updated["revision"] += 1
                    lease.check()
                    store.commit(updated)
                    state = updated
            rows, bins = [], {}
            for method in plan["methods"]:
                records = [r for r in state["trials"] if r["method"] == method]
                frame = pl.concat(
                    [pl.read_parquet(restore_member(store, r["predictions"])) for r in records]
                )
                metrics = evaluate_probabilities(
                    frame["target"].to_numpy(),
                    frame["prediction"].to_numpy(),
                    frame["WEEK_NUM"].to_numpy(),
                )
                rows.append(
                    {
                        "method": method,
                        "mean_fold_stability": float(
                            np.mean([r["metrics"]["stability_score"] for r in records])
                        ),
                        "worst_fold_stability": min(
                            r["metrics"]["stability_score"] for r in records
                        ),
                        "evaluation_rows": len(frame),
                        **{f"pooled_{k}": v for k, v in metrics.items()},
                    }
                )
                bins[method] = reliability(frame, plan["reliability_edges"])
            result = {
                "schema_version": 1,
                "identity": identity,
                "study_key": key,
                "scope": plan["scope"],
                "new_base_model_fits": 0,
                "new_calibrator_fits": 8,
                "holdout_used": False,
                "release_changed": False,
                "evaluation_weeks": [41, 72],
                "rows": rows,
                "folds": state["trials"],
                "reliability": bins,
            }
            path = work / "comparison.json"
            save_json(path, result)
            updated = copy.deepcopy(state)
            updated.update(complete=True, report=publish_verified(store, path, "comparison.json"))
            updated["revision"] += 1
            lease.check()
            store.commit(updated)
            logger.event(
                "calibration_research_completed",
                **{k: v for k, v in result.items() if k not in ("folds", "reliability", "rows")},
            )
            return result
