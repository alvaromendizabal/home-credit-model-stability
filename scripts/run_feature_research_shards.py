#!/usr/bin/env python3
"""Run independent late-fold fits using the unchanged, separately pinned training code.

The orchestration checkout and imported training checkout have distinct identities.
Workers own separate ledgers. Collection requires the original worker to be stopped,
verifies every imported artifact and never fits or substitutes a model.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.modeling.acceptance import require
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import FeatureRef, FeatureSnapshot
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.feature_research import hypothesis_from_dict, run_fold, validate_plan
from home_credit.modeling.release import read_object
from home_credit.modeling.release_workflow import ReleaseLogger, restore_member, restore_snapshot
from home_credit.modeling.screening import feature_refs_from_payload
from home_credit.observability.runtime import StageTimer


def assignment(policy: dict[str, Any], shard: str) -> set[tuple[str, int]]:
    require(
        policy["schema_version"] == 1 and policy["maximum_complete_fits"] == 20,
        "invalid shard budget",
    )
    require(shard in policy["shards"], "unknown shard")
    result = set()
    ownership: set[tuple[str, int]] = set()
    expected = {"wider_original", "engineered", "without_amount_ratios", "without_peer_statistics"}
    for name, item in policy["shards"].items():
        pairs = {(e, f) for e in item["experiments"] for f in item["folds"]}
        require(
            set(item["experiments"]) <= expected and item["folds"] == [4, 5],
            "unplanned shard scope",
        )
        require(len(pairs) == len(item["experiments"]) * len(item["folds"]), "duplicate shard task")
        require(not pairs & ownership, "overlapping shard assignments")
        ownership.update(pairs)
        if name == shard:
            result = pairs
    require(ownership == {(e, f) for e in expected for f in (4, 5)}, "late-fold coverage changed")
    return result


def source_identity(orchestration: Path, training: Path, policy: dict[str, Any]) -> dict[str, str]:
    versions = {}
    for name, root in (
        ("orchestration_source_commit", orchestration),
        ("training_source_commit", training),
    ):
        versions[name] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        require(
            not subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip(),
            f"dirty {name}",
        )
    require(
        versions["training_source_commit"] == policy["training_source_commit"],
        "training source changed",
    )
    require(
        Path(run_fold.__code__.co_filename).resolve().is_relative_to(training.resolve()),
        "training import came from the wrong checkout",
    )
    require(
        sha256_file(orchestration / "uv.lock") == sha256_file(training / "uv.lock"),
        "dependency lock changed",
    )
    versions["orchestration_policy_sha256"] = sha256_file(
        orchestration / "configs/feature_research_shards.json"
    )
    return versions


def worker_identity(
    versions: dict[str, str], shard: str, tasks: set[tuple[str, int]]
) -> dict[str, Any]:
    """Keep the in-memory identity identical to its JSON representation on resume."""
    return {**versions, "shard": shard, "tasks": [list(pair) for pair in sorted(tasks)]}


def conditions(
    training: Path,
    snapshot: FeatureSnapshot,
    screen: dict[str, Any],
    config: BenchmarkConfig,
    plan: dict[str, Any],
) -> dict[str, tuple[tuple[FeatureRef, ...], tuple[Any, ...]]]:
    source = read_object(training / "reports/feature_ablation/feature_screen.json")["result"]
    original = feature_refs_from_payload(source)
    require(len(original) == 700, "original feature plan changed")
    selected = tuple(hypothesis_from_dict(h) for h in screen["selected"])
    names = {f.name for f in original}
    eligible = [
        s
        for s in source["scores"]
        if s["unique_values"] > 1
        and s["missing_fraction"] <= config.screening.max_missing_fraction
        and (not s["categorical"] or s["unique_values"] <= 5000)
        and s["name"] not in names
    ]
    ranked = sorted(eligible, key=lambda s: (-s["selection_score"], -s["target_gain"], s["name"]))
    index = {f.name: f for f in snapshot.features}
    wider = (
        *original,
        *(index[s["name"]] for s in ranked[: plan["wider_original_features"] - 700]),
    )
    return {
        "wider_original": (wider, ()),
        "engineered": (original, selected),
        "without_amount_ratios": (
            original,
            tuple(h for h in selected if h.family != "amount_ratios"),
        ),
        "without_peer_statistics": (
            original,
            tuple(h for h in selected if h.family != "peer_statistics"),
        ),
    }


def parent_state(
    client: Any, bucket: str, policy: dict[str, Any], work: Path, logger: ReleaseLogger
) -> tuple[ExperimentStore, dict[str, Any]]:
    prefix = f"home-credit-model-stability/feature-research/{policy['training_study_key']}"
    store = ExperimentStore(client, bucket, prefix, work, logger)
    raw = store.read(f"{prefix}/study.json")
    require(raw is not None, "original training ledger absent")
    assert raw is not None
    state = json.loads(raw[0])
    require(
        state["identity"]["source_commit"] == policy["training_source_commit"],
        "parent training identity changed",
    )
    return store, state


def run_worker(orchestration: Path, training: Path, bucket: str, shard: str, job: str) -> None:
    policy = read_object(orchestration / "configs/feature_research_shards.json")
    tasks = assignment(policy, shard)
    versions = source_identity(orchestration, training, policy)
    logger = ReleaseLogger(f"feature-shard-{shard}", training / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    identity = worker_identity(versions, shard, tasks)
    key = sha256_bytes(canonical_json_bytes(identity))
    work = training / "artifacts/feature_shards" / key
    work.mkdir(parents=True, exist_ok=True)
    prefix = (
        "home-credit-model-stability/feature-research-shards/"
        f"{policy['training_study_key']}/{shard}"
    )
    store = ExperimentStore(client, bucket, prefix, work, logger)
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
                    "managed_job": job,
                    "instance_type": policy["shards"][shard]["instance_type"],
                }
                store.commit(state)
            for record in state["trials"]:
                for member in record["artifacts"].values():
                    restore_member(store, member)
            if state["complete"]:
                logger.event("feature_shard_reused", shard=shard, new_model_fits=0)
                return
            _, parent = parent_state(client, bucket, policy, work, logger)
            require(
                not tasks & {(t["experiment"], t["fold"]) for t in parent["trials"]},
                "parent already owns an assigned late fold",
            )
            plan = read_object(training / "configs/feature_research.json")
            protocol = read_object(training / "configs/validation_protocol.json")
            validate_plan(plan, protocol)
            require(
                sha256_file(training / "configs/feature_research.json")
                == parent["identity"]["plan_sha256"],
                "training plan changed",
            )
            config, _ = BenchmarkConfig.load(training / "configs/model_benchmark.json")
            with StageTimer(logger, "restore_frozen_snapshot", heartbeat_seconds=15):
                snapshot = restore_snapshot(training, plan, store)
                member = parent["stages"]["screen"]["screen"]
                require(member["sha256"] == policy["screen_sha256"], "screen changed")
                screen = read_object(restore_member(store, member))
            variants = conditions(training, snapshot, screen, config, plan)
            for experiment in policy["shards"][shard]["experiments"]:
                original, hypotheses = variants[experiment]
                payload = {
                    "original": [asdict(f) for f in original],
                    "hypotheses": [asdict(h) for h in hypotheses],
                }
                reference = next(t for t in parent["trials"] if t["experiment"] == experiment)
                require(
                    sha256_bytes(canonical_json_bytes(payload))
                    == reference["artifacts"]["features"]["sha256"],
                    "shard feature recipe differs from original fitted control",
                )
            completed = {(t["experiment"], t["fold"]) for t in state["trials"]}
            for experiment, number in sorted(tasks):
                if (experiment, number) in completed:
                    continue
                lease.check()
                original, hypotheses = variants[experiment]
                record = run_fold(
                    snapshot,
                    original,
                    hypotheses,
                    protocol["inner_temporal_cv"]["folds"][number - 1],
                    experiment,
                    config,
                    plan,
                    store,
                    logger,
                )
                record["execution_provenance"] = {
                    **versions,
                    "managed_job": job,
                    "instance_type": state["instance_type"],
                }
                updated = copy.deepcopy(state)
                updated["trials"].append(record)
                updated["revision"] += 1
                lease.check()
                store.commit(updated)
                state = updated
                logger.event(
                    "feature_shard_fit_completed",
                    shard=shard,
                    completed=len(state["trials"]),
                    total=len(tasks),
                    experiment=experiment,
                    fold=number,
                )
            require(
                {(t["experiment"], t["fold"]) for t in state["trials"]} == tasks, "incomplete shard"
            )
            updated = copy.deepcopy(state)
            updated.update(complete=True, revision=state["revision"] + 1)
            lease.check()
            store.commit(updated)
            logger.event("feature_shard_completed", shard=shard, complete_fits=len(tasks))


def guard_original(orchestration: Path, training: Path, bucket: str) -> None:
    """Stop only the named source driver after its first twelve fits are durable."""
    policy = read_object(orchestration / "configs/feature_research_shards.json")
    source_identity(orchestration, training, policy)
    logger = ReleaseLogger("feature-shard-guard", training / "logs")
    client = boto3.client("s3", region_name="us-west-2")
    managed = boto3.client("sagemaker", region_name="us-west-2")
    plan = read_object(training / "configs/feature_research.json")
    expected = {(name, fold) for name in plan["experiments"] for fold in (1, 2, 3)}
    with StageTimer(logger, "wait_for_twelve_durable_fits", heartbeat_seconds=15):
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            _, state = parent_state(client, bucket, policy, training / "artifacts", logger)
            status = managed.describe_processing_job(ProcessingJobName=policy["original_job"])[
                "ProcessingJobStatus"
            ]
            if status not in {"InProgress", "Stopping"}:
                logger.event(
                    "original_worker_already_finished",
                    status=status,
                    durable_fits=len(state["trials"]),
                )
                return
            completed = {(r["experiment"], r["fold"]) for r in state["trials"]}
            if expected <= completed:
                require(
                    completed == expected, "source driver already advanced into assigned late folds"
                )
                if status == "InProgress":
                    managed.stop_processing_job(ProcessingJobName=policy["original_job"])
                logger.event(
                    "original_worker_stop_requested", durable_fits=12, job=policy["original_job"]
                )
                return
            time.sleep(15)
    raise RuntimeError("source-driver guard exceeded its bounded wait")


def collect(orchestration: Path, training: Path, bucket: str) -> None:
    policy = read_object(orchestration / "configs/feature_research_shards.json")
    versions = source_identity(orchestration, training, policy)
    logger = ReleaseLogger("feature-shard-collection", training / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    managed = boto3.client("sagemaker", region_name="us-west-2").describe_processing_job(
        ProcessingJobName=policy["original_job"]
    )
    require(
        managed["ProcessingJobStatus"] in {"Stopped", "Failed", "Completed"},
        "original worker is still active",
    )
    work = training / "artifacts/feature_shard_collection"
    work.mkdir(parents=True, exist_ok=True)
    parent, initial = parent_state(client, bucket, policy, work, logger)
    with StageTimer(logger, "wait_for_original_lease_expiry", heartbeat_seconds=15):
        for _ in range(24):
            lease_record = parent.read(f"{parent.prefix}/writer.json")
            remaining = (
                0.0
                if lease_record is None
                else float(json.loads(lease_record[0])["expires_epoch"]) - time.time()
            )
            if remaining <= 0:
                break
            time.sleep(min(15, remaining))
        else:
            raise RuntimeError("original writer lease did not expire")
    with (work / "collector.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with WriterLease(parent) as lease:
            state = parent.restore(initial["identity"])
            require(
                state is not None and not state["complete"],
                "collection requires incomplete original study",
            )
            assert state is not None
            receipts, imported = [], []
            for shard in policy["shards"]:
                tasks = assignment(policy, shard)
                key = (
                    "home-credit-model-stability/feature-research-shards/"
                    f"{policy['training_study_key']}/{shard}/study.json"
                )
                raw = parent.read(key)
                require(raw is not None, "shard ledger absent")
                assert raw is not None
                worker = json.loads(raw[0])
                require(
                    worker["complete"]
                    and all(worker["identity"][k] == v for k, v in versions.items()),
                    "shard lineage changed",
                )
                require(
                    {(t["experiment"], t["fold"]) for t in worker["trials"]} == tasks,
                    "shard coverage changed",
                )
                for record in worker["trials"]:
                    for member in record["artifacts"].values():
                        restore_member(parent, member)
                imported.extend(worker["trials"])
                receipts.append(
                    {
                        "shard": shard,
                        "ledger_key": key,
                        "ledger_sha256": sha256_bytes(raw[0]),
                        "managed_job": worker["managed_job"],
                        "instance_type": worker["instance_type"],
                    }
                )
            occupied = {(t["experiment"], t["fold"]) for t in state["trials"]}
            for record in imported:
                pair = (record["experiment"], record["fold"])
                if pair in occupied:
                    require(
                        next(t for t in state["trials"] if (t["experiment"], t["fold"]) == pair)
                        == record,
                        "conflicting duplicate fit",
                    )
                else:
                    state["trials"].append(record)
                    occupied.add(pair)
            require(
                len(state["trials"]) <= 20 and len(occupied) == len(state["trials"]),
                "combined fit budget exceeded",
            )
            state["execution_shards"] = receipts
            state["revision"] += 1
            lease.check()
            parent.commit(state)
            logger.event(
                "feature_shards_collected", completed_fits=len(state["trials"]), new_model_fits=0
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "collect", "guard"])
    parser.add_argument("--orchestration-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--shard")
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    if args.mode == "worker":
        if args.shard is None:
            parser.error("worker requires --shard")
        run_worker(args.orchestration_root, args.training_root, args.bucket, args.shard, args.job)
    elif args.mode == "collect":
        collect(args.orchestration_root, args.training_root, args.bucket)
    else:
        guard_original(args.orchestration_root, args.training_root, args.bucket)
