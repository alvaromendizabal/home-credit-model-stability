"""Recheck saved engineered models and correct SHAP sampling without model fits."""

from __future__ import annotations

import copy
import fcntl
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.features.research import apply_peer_references, case_features, required_sources
from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.acceptance import compare_number, require
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import FeatureRef, FeatureSnapshot, load_feature_frame
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.feature_research import hypothesis_from_dict, shap_diagnostics
from home_credit.modeling.release import read_object, save_json, transform
from home_credit.modeling.release_workflow import (
    ReleaseLogger,
    publish_verified,
    restore_member,
    restore_snapshot,
)
from home_credit.observability.runtime import StageTimer


def interpret_fold(
    record: dict[str, Any],
    fold: dict[str, int],
    snapshot: FeatureSnapshot,
    config: BenchmarkConfig,
    plan: dict[str, Any],
    store: ExperimentStore,
) -> dict[str, Any]:
    """Rebuild only validation features from saved maps and verify all predictions."""
    require(record["experiment"] == "engineered", "unexpected interpretation model")
    require(fold["validation_week_max"] <= 72, "interpretation cannot read the observed holdout")
    files = {name: restore_member(store, member) for name, member in record["artifacts"].items()}
    saved = read_object(files["features"])
    original = tuple(FeatureRef(**f) for f in saved["original"])
    hypotheses = tuple(hypothesis_from_dict(h) for h in saved["hypotheses"])
    features = (*original, *(h.ref for h in hypotheses))
    required = {f.name for f in original} | required_sources(hypotheses)
    sources = tuple(f for f in snapshot.features if f.name in required)
    require({f.name for f in sources} == required, "interpretation feature source missing")
    valid = load_feature_frame(
        snapshot,
        sources,
        week_min=fold["validation_week_min"],
        week_max=fold["validation_week_max"],
        max_rows=None,
        seed=config.seed + record["fold"] * 101,
    )
    valid = valid.hstack(case_features(valid, hypotheses)).hstack(
        apply_peer_references(valid, hypotheses, read_object(files["peer_references"]))
    )
    x = transform(valid, tuple(features), read_object(files["encoder"]))
    model = lgb.Booster(model_file=str(files["model"]))
    prediction = np.asarray(model.predict(x, num_threads=config.threads), dtype=np.float64)
    expected = pl.read_parquet(files["predictions"]).sort("case_id")
    metadata = ["case_id", "WEEK_NUM", "target"]
    require(
        valid.select(pl.col(metadata).cast(pl.Int64)).equals(
            expected.select(pl.col(metadata).cast(pl.Int64))
        ),
        "saved model validation population changed",
    )
    error = float(np.max(np.abs(prediction - expected["prediction"].to_numpy())))
    require(error <= 1e-12, "saved feature/encoder/native-model replay disagrees")
    metrics = evaluate_probabilities(
        valid["target"].to_numpy(), prediction, valid["WEEK_NUM"].to_numpy()
    )
    for name, value in metrics.items():
        compare_number(value, record["metrics"][name], f"independent model replay {name}")
    result = read_object(files["diagnostics"])
    result.update(
        shap_diagnostics(model, x, valid, tuple(features), plan, config.seed + record["fold"])
    )
    result.update(
        fold=record["fold"],
        native_model_sha256=record["artifacts"]["model"]["sha256"],
        original_diagnostics_sha256=record["artifacts"]["diagnostics"]["sha256"],
        prediction_maximum_absolute_error=error,
        replayed_predictions=len(valid),
        new_model_fits=0,
        peer_references_refitted=False,
    )
    del model, x, valid
    gc.collect()
    return result


def run(root: Path, bucket: str) -> dict[str, Any]:
    policy = read_object(root / "configs/feature_interpretation.json")
    require(
        policy["schema_version"] == 1
        and policy["new_model_fits"] == 0
        and policy["development_week_max"] == 72
        and policy["shap_rows"] == 512
        and policy["seed_offset"] == 104729
        and policy["prediction_absolute_tolerance"] == 1e-12
        and policy["sampling"] == "uniform_without_replacement_from_full_validation_fold",
        "interpretation policy changed",
    )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
        "interpretation requires clean committed source",
    )
    require(
        sha256_file(root / "configs/model_benchmark.json") == policy["benchmark_config_sha256"],
        "reference model configuration changed",
    )
    logger = ReleaseLogger("feature-interpretation", root / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    reader = ExperimentStore(client, bucket, "unused", root / "artifacts", logger)
    remote = reader.read(
        f"home-credit-model-stability/feature-research/{policy['training_study_key']}/study.json"
    )
    require(remote is not None, "training ledger absent")
    assert remote is not None
    training_bytes, _ = remote
    training = json.loads(training_bytes)
    require(training["complete"] is True, "wait for the complete original training study")
    require(
        training["identity"]["source_commit"] == policy["training_source_commit"]
        and len(training["trials"]) == 20,
        "unexpected training lineage",
    )
    identity = {
        "source_commit": source,
        "training_ledger_sha256": sha256_bytes(training_bytes),
        "policy_sha256": sha256_file(root / "configs/feature_interpretation.json"),
        "lock_sha256": sha256_file(root / "uv.lock"),
    }
    require(
        identity["lock_sha256"] == training["identity"]["lock_sha256"], "dependency lock changed"
    )
    key = sha256_bytes(canonical_json_bytes(identity))
    work = root / "artifacts/feature_interpretation" / key
    work.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(
        client, bucket, f"home-credit-model-stability/feature-interpretation/{key}", work, logger
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
            for item in state["trials"]:
                restore_member(store, item["diagnostics"])
                for member in item["verified_inputs"].values():
                    restore_member(store, member)
            if state["complete"]:
                logger.event("feature_interpretation_reused", new_model_fits=0)
                return read_object(restore_member(store, state["report"]))
            plan = read_object(root / "configs/feature_research.json")
            require(
                sha256_file(root / "configs/feature_research.json")
                == training["identity"]["plan_sha256"],
                "training plan changed",
            )
            with StageTimer(logger, "restore_feature_snapshot", heartbeat_seconds=15):
                snapshot = restore_snapshot(root, plan, store)
            config, _ = BenchmarkConfig.load(root / "configs/model_benchmark.json")
            folds = read_object(root / "configs/validation_protocol.json")["inner_temporal_cv"][
                "folds"
            ]
            completed = {item["fold"] for item in state["trials"]}
            for record in training["trials"]:
                if record["experiment"] != "engineered" or record["fold"] in completed:
                    continue
                lease.check()
                with StageTimer(
                    logger, f"replay_and_interpret_fold_{record['fold']}", heartbeat_seconds=15
                ):
                    result = interpret_fold(
                        record, folds[record["fold"] - 1], snapshot, config, plan, store
                    )
                    path = work / f"diagnostics_fold_{record['fold']}.json"
                    save_json(path, result)
                    item = {
                        "fold": record["fold"],
                        "diagnostics": publish_verified(store, path, path.name),
                        "verified_inputs": record["artifacts"],
                    }
                updated = copy.deepcopy(state)
                updated["trials"].append(item)
                updated["revision"] += 1
                lease.check()
                store.commit(updated)
                state = updated
            require(
                {t["fold"] for t in state["trials"]} == set(range(1, 6)),
                "incomplete interpretation",
            )
            report = {
                "schema_version": 1,
                "identity": identity,
                "study_key": key,
                "new_model_fits": 0,
                "holdout_used": False,
                "folds": state["trials"],
            }
            path = work / "interpretation.json"
            save_json(path, report)
            updated = copy.deepcopy(state)
            updated.update(
                complete=True,
                revision=state["revision"] + 1,
                report=publish_verified(store, path, path.name),
            )
            lease.check()
            store.commit(updated)
            logger.event("feature_interpretation_completed", study_key=key, new_model_fits=0)
            return report
