"""Durable frozen release: development refit, one outer evaluation, all-label refit.

No submission file is created. Cloud artifacts are checked by full read-back,
not just by the presence of an S3 key or an upload return code.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import importlib.metadata
import json
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]
from polars.testing import assert_frame_equal

from home_credit.data.loader import S3RawStore, parse_manifest_bytes
from home_credit.features.builder import _dataset_for_record, _resolved_semantic_for_source, group_logical_sources
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.data import FeatureRef, FeatureSnapshot, load_feature_frame
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.inference import inference_code_identity, local_test_records, predict_raw, raw_test_frame
from home_credit.modeling.release import checked_member, evaluation_report, fit_encoder, fit_frozen_model, frozen_iterations, predict_components, read_object, save_json, transform, validate_features, verify_file
from home_credit.modeling.selection import require
from home_credit.observability.logging import RunLogger, iso_utc
from home_credit.observability.runtime import StageTimer
from home_credit.validation.protocol import verify_protocol_sha256


class ReleaseLogger(RunLogger):
    """Attach nested stage and invocation timings to progress and heartbeat events."""

    def __init__(self, name: str, directory: Path) -> None:
        self.started = time.monotonic()
        self.stages: list[tuple[str, float]] = []
        self.guard = threading.RLock()
        super().__init__(name, directory)

    def event(self, event: str, **fields: Any) -> None:
        with self.guard:
            now = time.monotonic()
            if event == "stage_started":
                self.stages.append((str(fields["stage"]), now))
            stage, started = self.stages[-1] if self.stages else ("release", self.started)
            fields.setdefault("stage", stage)
            fields.setdefault("stage_elapsed_seconds", round(now - started, 3))
            fields.setdefault("total_elapsed_seconds", round(now - self.started, 3))
            super().event(event, **fields)
            if event in {"stage_completed", "stage_failed"} and self.stages:
                self.stages.pop()


def publish_verified(store: ExperimentStore, path: Path, relative: str) -> dict[str, Any]:
    """Read the uploaded object back and compare its full digest before committing a stage."""
    key = store.publish(path, "checkpoints")
    remote = store.read(key)
    require(remote is not None and sha256_bytes(remote[0]) == sha256_file(path), "S3 read-back mismatch")
    return {"path": relative, "key": key, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def restore_member(store: ExperimentStore, member: dict[str, Any]) -> Path:
    """Restore only a verified member within the current release directory."""
    path = checked_member(store.root, member["path"])
    store.download(member["key"], path, member["sha256"])
    require(path.stat().st_size == member["bytes"], "restored artifact length mismatch")
    return path


def commit_stage(store: ExperimentStore, lease: WriterLease, state: dict[str, Any], name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Never update local completion state ahead of its successful durable commit."""
    lease.check()
    updated = copy.deepcopy(state)
    updated["stages"][name] = record
    updated["revision"] += 1
    store.commit(updated)
    return updated


def restore_snapshot(root: Path, plan: dict[str, Any], store: ExperimentStore) -> FeatureSnapshot:
    """Validate all 34 frozen blocks; downloads resume one hash-verified object at a time."""
    digest = plan["feature_manifest_sha256"]
    prefix = f"home-credit-model-stability/feature-snapshots/{digest}"
    directory = root / "artifacts/feature_cache"
    manifest_path = directory / "feature_manifest.json"
    store.download(f"{prefix}/feature_manifest.json", manifest_path, digest)
    manifest = read_object(manifest_path)
    for number, block in enumerate(manifest["blocks"], 1):
        relative = f"blocks/{block['split']}/{block['family']}_depth{block['depth']}.parquet"
        destination = checked_member(directory, relative)
        store.download(f"{prefix}/{relative}", destination, block["output_sha256"])
        require(destination.stat().st_size == block["output_bytes"], "feature-block size mismatch")
        store.logger.event("release_feature_verified", completed=number, total=len(manifest["blocks"]))
    return FeatureSnapshot.load(directory, expected_manifest_sha256=digest, expected_protocol_sha256=plan["protocol_sha256"])


def validate_policy(root: Path, plan: dict[str, Any], store: ExperimentStore) -> tuple[tuple[FeatureRef, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify the complete frozen decision and original development-derived training budgets."""
    require(plan["schema_version"] == 1 and plan["calibration"] == "none", "unsupported release")
    require(plan["holdout_tuning_allowed"] is False, "holdout cannot be used for tuning")
    require(plan["automatic_kaggle_submission"] is False and plan["automatic_submission_csv"] is False, "the release cannot perform the owner's export action")
    require(plan["max_new_model_fits"] == 4, "refit budget changed")
    verify_file(root / "uv.lock", plan["lock_sha256"])
    verify_file(root / "configs/benchmark_features.json", plan["feature_plan_sha256"])
    protocol = read_object(root / "configs/validation_protocol.json")
    require(verify_protocol_sha256(protocol), "protocol content changed")
    require(protocol["protocol_sha256"] == plan["protocol_sha256"], "protocol identity changed")
    outer = protocol["outer_holdout"]
    require(outer["locked"] is True, "original holdout protocol must remain immutable")
    require(plan["holdout_fit_weeks"] == [outer["development_week_min"], outer["development_week_max"]], "development scope changed")
    require(plan["holdout_evaluation_weeks"] == [outer["validation_week_min"], outer["validation_week_max"]], "holdout scope changed")
    require(plan["final_refit_weeks"] == [0, 91], "all-label refit scope changed")
    selected = plan["selection"]
    require(set(plan["components"]) == set(selected["weights"]) and len(plan["components"]) == 2, "frozen component budget changed")
    path = store.root / "inputs/selection.json"
    store.download(selected["key"], path, selected["sha256"])
    evidence = read_object(path)
    require(evidence["outer_holdout_touched"] is False and evidence["smoke"] is False, "invalid development selection")
    require(evidence["selected_candidate"] == selected["candidate"], "winner changed")
    require(set(evidence["selected_weights"]) == set(selected["weights"]), "ensemble members changed")
    for name, weight in selected["weights"].items():
        require(math.isclose(weight, evidence["selected_weights"][name], rel_tol=0, abs_tol=1e-12), "ensemble weight changed")
    source = read_object(root / "configs/model_selection.json")
    store.download(source["study"]["key"], store.root / "inputs/tuning.json", source["study"]["sha256"])
    study = read_object(store.root / "inputs/tuning.json")
    require(study["complete"] is True and study["outer_holdout_touched"] is False, "invalid tuning study")
    baseline = read_object(root / "configs/model_benchmark.json")["models"]["lightgbm"]
    parameters = {}
    for name, component in plan["components"].items():
        path = store.root / "inputs" / f"{name}.json"
        store.download(component["state_key"], path, component["state_sha256"])
        state = read_object(path)
        require(state["identity"]["feature_manifest_sha256"] == plan["feature_manifest_sha256"], "source feature lineage changed")
        require(state["identity"]["validation_protocol_sha256"] == plan["protocol_sha256"], "source validation lineage changed")
        require(frozen_iterations(state) == component["num_boost_round"], "tree budget does not match development evidence")
        matches = [row for row in study["trials"] if row["name"] == component["study_trial"]]
        require(len(matches) == 1 and matches[0]["state"] == "complete", "missing source trial")
        parameters[name] = {**baseline, **matches[0]["params"]}
    features = tuple(FeatureRef(**row) for row in read_object(root / "configs/benchmark_features.json")["selected_features"])
    validate_features(features)
    require(len(features) == plan["feature_count"] == 700, "frozen feature count changed")
    return features, parameters, protocol


def fit_phase(phase: str, weeks: list[int], expected_rows: int, snapshot: FeatureSnapshot, features: tuple[FeatureRef, ...], plan: dict[str, Any], parameters: dict[str, dict[str, Any]], state: dict[str, Any], store: ExperimentStore, lease: WriterLease) -> dict[str, Any]:
    """Preserve each completed native fit; only an unfinished component is retrained."""
    require(phase in {"development", "all_labels"}, "unknown training phase")
    expected_weeks = plan["holdout_fit_weeks" if phase == "development" else "final_refit_weeks"]
    require(weeks == expected_weeks, "fit phase temporal scope changed")
    names = sorted(plan["components"])
    specifications = {name: sha256_bytes(canonical_json_bytes({"phase": phase, "weeks": weeks, "rows": expected_rows, "features": [asdict(feature) for feature in features], "parameters": parameters[name], "rounds": plan["components"][name]["num_boost_round"], "seed": plan["seed"], "threads": plan["threads"]})) for name in names}
    pending = []
    for name in names:
        label = f"{phase}/{name}"
        if label in state["stages"]:
            record = state["stages"][label]
            require(record["fit_spec_sha256"] == specifications[name], "cached fit specification changed")
            require(record["fit_weeks"] == weeks and record["fit_rows"] == expected_rows, "cached fit scope changed")
            restore_member(store, record["model"])
            restore_member(store, record["encoder"])
            store.logger.event("release_model_reused", phase=phase, model=name)
        else:
            pending.append(name)
    if not pending:
        return state
    with StageTimer(store.logger, f"materialize_{phase}_fit", heartbeat_seconds=15):
        train = load_feature_frame(snapshot, features, week_min=weeks[0], week_max=weeks[1], max_rows=None, seed=plan["seed"])
        require(train.height == expected_rows, "fit population count changed")
        encoder = fit_encoder(train, features)
        encoder_path = store.root / phase / "encoder.json"
        save_json(encoder_path, encoder)
        encoder_member = publish_verified(store, encoder_path, f"{phase}/encoder.json")
        matrix = transform(train, features, encoder)
        target = train["target"].to_numpy().astype(np.int8)
        del train
        gc.collect()
    for name in pending:
        lease.check()
        component = plan["components"][name]
        path = store.root / phase / f"{name}.txt"
        receipt = fit_frozen_model(matrix, target, features, params=parameters[name], rounds=component["num_boost_round"], seed=plan["seed"], threads=plan["threads"], path=path, logger=store.logger, check_lease=lease.check)
        model_member = publish_verified(store, path, f"{phase}/{name}.txt")
        state = commit_stage(store, lease, state, f"{phase}/{name}", {**receipt, "fit_spec_sha256": specifications[name], "fit_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "fit_weeks": weeks, "model": model_member, "encoder": encoder_member})
        store.logger.event("release_model_completed", phase=phase, model=name, completed=sum("/" in key for key in state["stages"]), total=plan["max_new_model_fits"])
        gc.collect()
    del matrix, target
    gc.collect()
    return state


def evaluate_holdout(snapshot: FeatureSnapshot, features: tuple[FeatureRef, ...], plan: dict[str, Any], protocol: dict[str, Any], state: dict[str, Any], store: ExperimentStore, lease: WriterLease) -> dict[str, Any]:
    """Bind a one-time evaluation to exact development models before reading holdout labels."""
    if "holdout" in state["stages"]:
        record = state["stages"]["holdout"]
        restore_member(store, record["report"])
        restore_member(store, record["predictions"])
        store.logger.event("holdout_evaluation_reused", report_sha256=record["report"]["sha256"])
        return state
    members = {name: state["stages"][f"development/{name}"] for name in plan["components"]}
    intent = {"selection_sha256": plan["selection"]["sha256"], "feature_manifest_sha256": plan["feature_manifest_sha256"], "weights": plan["selection"]["weights"], "models": {name: row["model"]["sha256"] for name, row in members.items()}, "encoders": {name: row["encoder"]["sha256"] for name, row in members.items()}, "fit_weeks": plan["holdout_fit_weeks"], "evaluation_weeks": plan["holdout_evaluation_weeks"]}
    key = f"home-credit-model-stability/holdout/{plan['protocol_sha256']}/intent.json"
    existing = store.read(key)
    if existing is None:
        store.put(key, canonical_json_bytes(intent), IfNoneMatch="*")
    else:
        require(json.loads(existing[0]) == intent, "the holdout is already bound to a different frozen model; no retuning is allowed")
    store.logger.event("holdout_intent_verified", intent_sha256=sha256_bytes(canonical_json_bytes(intent)))
    paths = {name: restore_member(store, row["model"]) for name, row in members.items()}
    encoder_digests = {row["encoder"]["sha256"] for row in members.values()}
    require(len(encoder_digests) == 1, "ensemble encoding representations differ")
    encoder = read_object(restore_member(store, next(iter(members.values()))["encoder"]))
    weeks = plan["holdout_evaluation_weeks"]
    with StageTimer(store.logger, "evaluate_frozen_outer_holdout", heartbeat_seconds=15):
        frame = load_feature_frame(snapshot, features, week_min=weeks[0], week_max=weeks[1], max_rows=None, seed=plan["seed"])
        require(frame.height == protocol["outer_holdout"]["validation_rows"], "holdout row count changed")
        probability = predict_components(transform(frame, features, encoder), features, paths, plan["selection"]["weights"], threads=plan["threads"])
        report = evaluation_report(frame, probability, expected_weeks=list(range(weeks[0], weeks[1] + 1)))
        report.update(schema_version=1, scope="One frozen outer-holdout evaluation; not a leaderboard result", fit_weeks=plan["holdout_fit_weeks"], evaluation_weeks=weeks, calibration="none", intent=intent, release_identity=state["identity"], evaluated_utc=iso_utc(), evaluated_model_phase="development", all_label_model_evaluated=False)
        predictions = store.root / "holdout/predictions.parquet"
        predictions.parent.mkdir(parents=True, exist_ok=True)
        frame.select("case_id", "target", "WEEK_NUM").with_columns(pl.Series("prediction", probability)).write_parquet(predictions)
        path = store.root / "holdout/evaluation.json"
        save_json(path, report)
        predicted_member = publish_verified(store, predictions, "holdout/predictions.parquet")
        report_member = publish_verified(store, path, "holdout/evaluation.json")
    return commit_stage(store, lease, state, "holdout", {"report": report_member, "predictions": predicted_member})


def prepare_raw_schema(root: Path, plan: dict[str, Any], protocol: dict[str, Any], store: ExperimentStore) -> tuple[dict[str, Any], Path]:
    """Freeze original cross-shard semantics and restore only public test raw files for parity."""
    uri = protocol["data_lock"]["manifest_uri"]
    _, _, remainder = uri.partition("s3://")
    bucket, _, key = remainder.partition("/")
    require(bucket == store.bucket, "raw manifest bucket changed")
    path = store.root / "inputs/raw_manifest.jsonl"
    store.download(key, path, protocol["data_lock"]["manifest_sha256"])
    records = parse_manifest_bytes(path.read_bytes())
    raw_store = S3RawStore(store.bucket, plan["region"])
    sources = group_logical_sources(records)
    schema: dict[str, Any] = {"schema_version": 1, "recipe_sha256": sha256_file(root / "configs/feature_recipe.json"), "blocks": {}}
    for source in sources:
        if source.split != "test" or source.family == "base":
            continue
        semantic = _resolved_semantic_for_source(source, all_sources=sources, store=raw_store)
        observed = set()
        for record in source.records:
            observed.update(_dataset_for_record(record, raw_store).schema.names)
        schema["blocks"][f"{source.family}_depth{source.depth}"] = {"semantic": asdict(semantic), "allowed_absent_columns": sorted(set(semantic.predictors) - observed)}
    raw_directory = store.root / "inputs/raw_test"
    for source in sources:
        if source.split == "test":
            for record in source.records:
                store.download(record.s3_key, raw_directory / Path(record.file).name, record.sha256)
    return schema, raw_directory


def verify_raw_parity(root: Path, snapshot: FeatureSnapshot, features: tuple[FeatureRef, ...], plan: dict[str, Any], protocol: dict[str, Any], state: dict[str, Any], store: ExperimentStore, lease: WriterLease) -> dict[str, Any]:
    """Reject raw-inference incompatibility before spending time on a new fit."""
    previous = state["stages"].get("raw_parity")
    code_sha = inference_code_identity()
    if previous is not None and previous["inference_code_sha256"] == code_sha:
        restore_member(store, previous["schema"])
        restore_member(store, previous["features"])
        store.logger.event("raw_test_parity_reused", rows=previous["rows"], features=len(features))
        return state
    directory = store.root / "parity"
    directory.mkdir(parents=True, exist_ok=True)
    with StageTimer(store.logger, "raw_test_feature_parity", heartbeat_seconds=15):
        schema, raw_directory = prepare_raw_schema(root, plan, protocol, store)
        save_json(directory / "raw_schema.json", schema)
        shutil.copyfile(root / "configs/feature_recipe.json", directory / "feature_recipe.json")
        records = local_test_records(raw_directory)
        actual = raw_test_frame(raw_directory, records, features, schema, directory / "feature_recipe.json")
        reference = pl.scan_parquet(next(b.path for b in snapshot.test_blocks if b.family == "base")).select("case_id")
        for block in snapshot.test_blocks:
            names = [f.name for f in features if f.block == block.name]
            if names:
                reference = reference.join(pl.scan_parquet(block.path).select(["case_id", *names]), on="case_id", how="left", validate="1:1")
        expected = reference.select(["case_id", *[f.name for f in features]]).sort("case_id").collect()
        assert_frame_equal(actual, expected, check_exact=False, rel_tol=1e-6, abs_tol=1e-8)
        store.logger.event("raw_test_parity_passed", rows=actual.height, features=len(features), hidden_test_evaluated=False)
    frame_path = directory / "features.parquet"
    actual.write_parquet(frame_path)
    return commit_stage(store, lease, state, "raw_parity", {"inference_code_sha256": code_sha, "rows": actual.height, "schema": publish_verified(store, directory / "raw_schema.json", "parity/raw_schema.json"), "features": publish_verified(store, frame_path, "parity/features.parquet")})


def assemble_bundle(root: Path, snapshot: FeatureSnapshot, features: tuple[FeatureRef, ...], plan: dict[str, Any], protocol: dict[str, Any], state: dict[str, Any], store: ExperimentStore, lease: WriterLease) -> dict[str, Any]:
    """Verify raw-test feature parity and publish a portable all-label native model bundle."""
    directory = store.root / "bundle"
    directory.mkdir(parents=True, exist_ok=True)
    parity = state["stages"]["raw_parity"]
    require(parity["inference_code_sha256"] == inference_code_identity(), "raw parity is stale")
    shutil.copyfile(restore_member(store, parity["schema"]), directory / "raw_schema.json")
    shutil.copyfile(root / "configs/feature_recipe.json", directory / "feature_recipe.json")
    actual = pl.read_parquet(restore_member(store, parity["features"]))
    save_json(directory / "features.json", {"features": [asdict(f) for f in features]})
    models = {}
    encoders = set()
    for name in sorted(plan["components"]):
        record = state["stages"][f"all_labels/{name}"]
        shutil.copyfile(restore_member(store, record["model"]), directory / f"{name}.txt")
        encoder_path = restore_member(store, record["encoder"])
        encoders.add(sha256_file(encoder_path))
        shutil.copyfile(encoder_path, directory / "encoder.json")
        models[name] = {"path": f"{name}.txt", "sha256": record["model"]["sha256"]}
    require(len(encoders) == 1, "all-label ensemble encoders differ")
    encoder = read_object(directory / "encoder.json")
    prediction = predict_components(transform(actual, features, encoder), features, {name: directory / item["path"] for name, item in models.items()}, plan["selection"]["weights"])
    package = root / "src/home_credit"
    for source in package.rglob("*.py"):
        target_path = directory / "source/home_credit" / source.relative_to(package)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_path)
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "bundle.json":
            files.append({"path": path.relative_to(directory).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    bundle = {"schema_version": 1, "phase": "all_labels", "fit_weeks": plan["final_refit_weeks"], "calibration": "none", "feature_count": len(features), "models": models, "weights": plan["selection"]["weights"], "release_identity": state["identity"], "feature_manifest_sha256": plan["feature_manifest_sha256"], "holdout_report_sha256": state["stages"]["holdout"]["report"]["sha256"], "holdout_score_applies_to_this_model": False, "files": files, "inference_code_sha256": inference_code_identity(), "public_test_parity": {"rows": actual.height, "features": len(features), "passed": True, "probability_min": float(prediction.min()), "probability_max": float(prediction.max()), "hidden_test_evaluated": False}, "submission_csv_generated": False, "kaggle_submitted": False}
    save_json(directory / "bundle.json", bundle)
    _, raw_directory = prepare_raw_schema(root, plan, protocol, store)
    with StageTimer(store.logger, "verify_packaged_raw_inference", heartbeat_seconds=15):
        raw_prediction = predict_raw(directory, raw_directory, store.root / "inference_batches", expected_bundle_sha256=sha256_file(directory / "bundle.json"), logger=store.logger, batch_rows=3)
        require(raw_prediction["case_id"].tolist() == actual["case_id"].to_list(), "packaged raw prediction coverage changed")
        require(bool(np.allclose(raw_prediction["score"], prediction, rtol=0, atol=1e-12)), "packaged raw inference differs from frozen feature scoring")
        resumed = predict_raw(directory, raw_directory, store.root / "inference_batches", expected_bundle_sha256=sha256_file(directory / "bundle.json"), logger=store.logger, batch_rows=3)
        require(resumed.equals(raw_prediction), "resumed scoring predictions changed")
    members = [publish_verified(store, directory / item["path"], f"bundle/{item['path']}") for item in files]
    manifest_member = publish_verified(store, directory / "bundle.json", "bundle/bundle.json")
    archive = store.root / "inference_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                item = zipfile.ZipInfo(path.relative_to(directory).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(item, path.read_bytes())
    archive_member = publish_verified(store, archive, "inference_bundle.zip")
    return commit_stage(store, lease, state, "bundle", {"manifest": manifest_member, "files": members, "archive": archive_member})


def execute(root: Path, bucket: str, *, preflight_only: bool = False) -> dict[str, Any]:
    """Run or resume the frozen pipeline without exposing any submission action."""
    plan = read_object(root / "configs/model_release.json")
    require(sys.version_info[:2] == (3, 12), "use the locked Python 3.12 environment")
    dependencies = ["configs/model_release.json", "configs/model_benchmark.json", "configs/model_selection.json", "configs/benchmark_features.json", "configs/validation_protocol.json", "uv.lock", "src/home_credit/modeling/release.py", "src/home_credit/modeling/data.py", "src/home_credit/modeling/encoding.py", "src/home_credit/metrics/classification.py", "src/home_credit/metrics/stability.py"]
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "polars", "pyarrow", "lightgbm", "scikit-learn"]}
    dependency_lock = tomllib.loads((root / "uv.lock").read_text())
    for name, version in versions.items():
        require({p["version"] for p in dependency_lock["package"] if p["name"] == name} == {version}, f"active {name} differs from uv.lock")
    identity = {"schema_version": 1, "inputs": {name: sha256_file(root / name) for name in dependencies}, "versions": versions}
    key = sha256_bytes(canonical_json_bytes(identity))
    work = root / "artifacts/model_release" / key
    work.mkdir(parents=True, exist_ok=True)
    logger = ReleaseLogger("model-release", work / "logs")
    client = boto3.client("s3", region_name=plan["region"], config=Config(retries={"mode": "standard", "total_max_attempts": 3}, connect_timeout=5, read_timeout=60))
    store = ExperimentStore(client, bucket, f"home-credit-model-stability/model-release/{key}", work, logger)
    with (work / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with WriterLease(store) as lease:
                state = store.restore(identity)
                if state is None:
                    state = {"schema_version": 1, "identity": identity, "revision": 0, "trials": [], "stages": {}, "complete": False, "created_utc": iso_utc()}
                    store.commit(state)
                with StageTimer(logger, "validate_frozen_release_policy", heartbeat_seconds=15):
                    features, parameters, protocol = validate_policy(root, plan, store)
                with StageTimer(logger, "restore_frozen_feature_snapshot", heartbeat_seconds=15):
                    snapshot = restore_snapshot(root, plan, store)
                state = verify_raw_parity(root, snapshot, features, plan, protocol, state, store, lease)
                if preflight_only:
                    logger.event("model_release_preflight_completed", new_model_fits=0, holdout_evaluated=False)
                    store.publish(logger.jsonl_path, "logs")
                    return state
                state = fit_phase("development", plan["holdout_fit_weeks"], protocol["outer_holdout"]["development_rows"], snapshot, features, plan, parameters, state, store, lease)
                state = evaluate_holdout(snapshot, features, plan, protocol, state, store, lease)
                state = fit_phase("all_labels", plan["final_refit_weeks"], protocol["observed_temporal_structure"]["train_rows"], snapshot, features, plan, parameters, state, store, lease)
                state = assemble_bundle(root, snapshot, features, plan, protocol, state, store, lease)
                state["complete"] = True
                state["revision"] += 1
                state["completed_utc"] = iso_utc()
                state["reporting_git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                store.commit(state)
                logger.event("model_release_completed", release_key=key, new_model_fit_budget=4, submission_generated=False)
                store.publish(logger.jsonl_path, "logs")
                return state
        except Exception as exc:
            logger.event("model_release_failed", error_type=type(exc).__name__, error=str(exc))
            try:
                store.publish(logger.jsonl_path, "logs")
            except Exception as upload_error:
                logger.event("log_upload_pending", error_type=type(upload_error).__name__)
            raise


def main() -> int:
    """Command-line execution of model preparation only; CSV export is notebook-owned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    require(Path.cwd() == root, "run from the repository root")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs", "uv.lock", "pyproject.toml"], text=True).strip()
    require(not dirty, "commit tested source changes before a model release")
    def terminate(signum: int, frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}; completed fits remain resumable")
    signal.signal(signal.SIGTERM, terminate)
    execute(root, args.bucket, preflight_only=args.preflight_only)
    print("MODEL_RELEASE_PREFLIGHT_COMPLETED" if args.preflight_only else "MODEL_RELEASE_COMPLETED", flush=True)
    return 0
