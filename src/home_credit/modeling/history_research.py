"""Checkpointed raw-history research on the original development population."""

from __future__ import annotations

import copy
import fcntl
import gc
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.data.loader import S3Uri, parse_manifest_bytes
from home_credit.features.builder import group_logical_sources
from home_credit.features.distributions import (
    STATISTICS,
    HistoryFeature,
    aggregate_history,
    inspect_source,
    scan_numeric_history,
)
from home_credit.features.research import Hypothesis, prune_candidates
from home_credit.modeling.ablation import compare_predictions
from home_credit.modeling.acceptance import require
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import (
    FeatureBlockRef,
    FeatureRef,
    FeatureSnapshot,
    load_feature_frame,
)
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.feature_research import run_fold
from home_credit.modeling.release import read_object, save_json, verify_file
from home_credit.modeling.release_workflow import (
    ReleaseLogger,
    commit_stage,
    publish_verified,
    restore_member,
    restore_snapshot,
)
from home_credit.modeling.screening import feature_refs_from_payload, screen_features
from home_credit.observability.runtime import StageTimer
from home_credit.validation.protocol import verify_protocol_sha256


def validate_plan(plan: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Reject changes to the bounded contrasts and the already-observed holdout."""
    require(plan["schema_version"] == 1, "unsupported history plan")
    require(
        verify_protocol_sha256(protocol) and protocol["protocol_sha256"] == plan["protocol_sha256"],
        "different or invalid temporal protocol",
    )
    require(
        plan["raw_manifest_sha256"] == protocol["data_lock"]["manifest_sha256"],
        "raw snapshot changed",
    )
    require(plan["development_week_max"] == 72, "history study cannot use the observed holdout")
    require(plan["release_promotion_allowed"] is False, "research cannot promote a release")
    require(plan["source_depths"] == [1, 2], "only relational histories are eligible")
    require(plan["statistics"] == list(STATISTICS), "aggregation contract changed")
    require(plan["quantile_interpolation"] == "linear", "quantile definition changed")
    require(
        plan["minimum_observations"] == {"median": 1, "iqr": 3, "p90": 3, "skew": 5}
        and plan["skew"] == "adjusted_Fisher_Pearson_positive_variance",
        "sample support contract changed",
    )
    require(
        plan["experiments"] == ["history_shape", "without_history_skew"]
        and plan["new_model_fit_budget"] == 10
        and plan["screen_model_fit_budget"] == 2,
        "experimental contrasts or budget changed",
    )
    require(plan["additional_features"] == 96, "selection budget changed")
    require(plan["max_concurrent_fits"] == 2, "worker budget changed")
    require(
        plan["screen_weeks"] == {"train": [0, 24], "validation": [25, 32]},
        "screen overlaps development evaluation",
    )
    require(1 <= plan["case_partition_rows"] <= 100000, "unbounded materialization")
    folds = protocol["inner_temporal_cv"]["folds"]
    require(
        folds
        == [
            {
                "fold": i,
                "train_week_min": 0,
                "train_week_max": 24 + 8 * i,
                "validation_week_min": 25 + 8 * i,
                "validation_week_max": 32 + 8 * i,
            }
            for i in range(1, 6)
        ],
        "frozen five-fold population changed",
    )


def build_history_snapshot(
    snapshot: FeatureSnapshot,
    plan: dict[str, Any],
    protocol: dict[str, Any],
    store: ExperimentStore,
    lease: WriterLease,
    state: dict[str, Any],
) -> tuple[FeatureSnapshot, tuple[HistoryFeature, ...], dict[str, Any]]:
    """Build per-source, per-partition checkpoints with full raw-file hash verification."""
    raw_location = S3Uri.parse(protocol["data_lock"]["manifest_uri"])
    require(raw_location.bucket == store.bucket, "raw snapshot belongs to another bucket")
    raw_manifest = store.root / "raw_manifest.jsonl"
    store.download(raw_location.key, raw_manifest, plan["raw_manifest_sha256"])
    sources = tuple(
        s
        for s in group_logical_sources(parse_manifest_bytes(raw_manifest.read_bytes()))
        if s.split == "train" and s.depth in plan["source_depths"]
    )
    require(bool(sources), "no raw history sources")
    base = next(b for b in snapshot.train_blocks if b.family == "base")
    cases = (
        pl.scan_parquet(base.path)
        .filter(pl.col("WEEK_NUM").is_between(0, 72))
        .select("case_id", "WEEK_NUM")
        .sort("case_id")
        .collect()
    )
    require(len(cases) == protocol["outer_holdout"]["development_rows"], "population changed")
    blocks: list[FeatureBlockRef] = []
    specifications: list[HistoryFeature] = []
    audits: list[dict[str, Any]] = []
    for source in sources:
        label = f"history/{source.family}_depth{source.depth}"
        if label not in state["stages"]:
            with StageTimer(store.logger, f"build_{label}", heartbeat_seconds=15):
                paths = []
                for record in source.records:
                    lease.check()
                    path = store.root / "raw" / Path(record.file).name
                    store.download(record.s3_key, path, record.sha256)
                    require(path.stat().st_size == record.bytes, "raw file length mismatch")
                    paths.append(path)
                specs, audit = inspect_source(tuple(paths), source.family, source.depth)
                audit["raw_records"] = [asdict(r) for r in source.records]
                parts = []
                if specs:
                    columns = tuple(sorted({s.column for s in specs}))
                    history = scan_numeric_history(tuple(paths), columns)
                    for number, population in enumerate(
                        cases.iter_slices(plan["case_partition_rows"])
                    ):
                        stage = f"{label}/part_{number}"
                        if stage not in state["stages"]:
                            lease.check()
                            block = aggregate_history(history, population, specs)
                            path = store.root / f"{stage}.parquet"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            block.write_parquet(path)
                            member = publish_verified(store, path, f"{stage}.parquet")
                            state = commit_stage(store, lease, state, stage, member)
                            del block
                        parts.append(restore_member(store, state["stages"][stage]))
                        store.logger.event(
                            "history_partition_verified",
                            source=source.logical_name,
                            completed=number + 1,
                            total=(len(cases) + plan["case_partition_rows"] - 1)
                            // plan["case_partition_rows"],
                        )
                    path = store.root / f"{label}.parquet"
                    pl.scan_parquet(parts).sort("case_id").sink_parquet(path)
                    member = publish_verified(store, path, f"{label}.parquet")
                else:
                    member = None
                state = commit_stage(
                    store,
                    lease,
                    state,
                    label,
                    {"block": member, "specs": [asdict(s) for s in specs], "audit": audit},
                )
        record = state["stages"][label]
        specs = tuple(HistoryFeature(**s) for s in record["specs"])
        audits.append(record["audit"])
        specifications.extend(specs)
        if record["block"] is not None:
            member = record["block"]
            path = restore_member(store, member)
            require(
                list(pl.read_parquet_schema(path)) == ["case_id", *(s.name for s in specs)],
                "history block schema changed",
            )
            blocks.append(
                FeatureBlockRef(
                    "train",
                    f"history_{source.family}",
                    source.depth,
                    path,
                    member["sha256"],
                    len(cases),
                    len(specs),
                )
            )
        gc.collect()
    identity = {
        "schema_version": 1,
        "scope": "development-only view; no test blocks or deployable history recipe",
        "base_feature_manifest_sha256": snapshot.manifest_sha256,
        "raw_manifest_sha256": plan["raw_manifest_sha256"],
        "plan_sha256": state["identity"]["plan_sha256"],
        "source_commit": state["identity"]["source_commit"],
        "development_rows": len(cases),
        "development_weeks": [0, 72],
        "blocks": [
            {"name": b.name, "sha256": b.output_sha256, "features": b.feature_columns}
            for b in blocks
        ],
        "source_audits": audits,
    }
    path = store.root / "history_manifest.json"
    save_json(path, identity)
    if "history_manifest" not in state["stages"]:
        state = commit_stage(
            store,
            lease,
            state,
            "history_manifest",
            publish_verified(store, path, "history_manifest.json"),
        )
    require(sha256_file(path) == state["stages"]["history_manifest"]["sha256"], "view changed")
    view = replace(
        snapshot,
        manifest_sha256=sha256_file(path),
        train_blocks=(*snapshot.train_blocks, *blocks),
        test_blocks=(),
        features=(*snapshot.features, *(s.ref for s in specifications)),
    )
    return view, tuple(specifications), state


def screen_history(
    snapshot: FeatureSnapshot,
    specs: tuple[HistoryFeature, ...],
    config: BenchmarkConfig,
    plan: dict[str, Any],
    store: ExperimentStore,
) -> dict[str, Any]:
    """Select once on the early windows; keep an explanation for every rejection."""
    frames = []
    for start, end, cap, seed in (
        (0, 24, config.screening.max_train_rows, config.seed),
        (25, 32, config.screening.max_validation_rows, config.seed + 1),
    ):
        frames.append(
            load_feature_frame(
                snapshot,
                tuple(s.ref for s in specs),
                week_min=start,
                week_max=end,
                max_rows=cap,
                seed=seed,
            )
        )
    train, valid = frames
    hypotheses = tuple(
        Hypothesis(s.name, s.statistic, s.statistic, (s.column,), "Within-applicant history shape")
        for s in specs
    )
    pruned, records = prune_candidates(train, hypotheses)
    names = {s.name for s in pruned}
    result = screen_features(
        train,
        valid,
        tuple(s.ref for s in specs if s.name in names),
        config=replace(config.screening, max_features=plan["additional_features"]),
        seed=config.seed,
        threads=config.threads,
        logger=store.logger,
    )
    selected = {s.name for s in result.selected_features}
    scores = {s.name: asdict(s) for s in result.scores}
    for record in records:
        score = scores.get(record["name"])
        reason = record["structural_rejection"]
        if reason is None and record["name"] not in selected:
            reason = (
                "missingness"
                if score and score["missing_fraction"] > config.screening.max_missing_fraction
                else "early_ranking_budget"
            )
        record.update(rejection_reason=reason, screen=score)
    payload = {
        "schema_version": 1,
        "screen_weeks": plan["screen_weeks"],
        "train_rows": len(train),
        "validation_rows": len(valid),
        "generated": len(specs),
        "retained": len(selected),
        "rejected": len(specs) - len(selected),
        "selected": [asdict(s) for s in specs if s.name in selected],
        "catalog": records,
        "rejection_counts": dict(
            Counter(r["rejection_reason"] for r in records if r["rejection_reason"])
        ),
        "result": result.to_payload(),
    }
    path = store.root / "screen.json"
    save_json(path, payload)
    return publish_verified(store, path, "screen.json")


def fold_sensitivity(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive fold omission; overlapping expanding folds are not independent trials."""
    control = {
        r["fold"]: r["stability_score"] for r in comparison["folds"] if r["experiment"] == "control"
    }
    output = []
    for row in comparison["rows"]:
        name = row["experiment"]
        if name == "control":
            continue
        differences = [
            r["stability_score"] - control[r["fold"]]
            for r in comparison["folds"]
            if r["experiment"] == name
        ]
        require(len(differences) == 5, "five fold contrasts required")
        omitted = [(sum(differences) - d) / 4 for d in differences]
        output.append(
            {
                "experiment": name,
                "positive_folds": sum(d > 0 for d in differences),
                "fold_deltas": differences,
                "omission_min": min(omitted),
                "omission_max": max(omitted),
                "interpretation": "Descriptive sensitivity; not a confidence interval or p-value",
            }
        )
    return output


def checkpointed_fold(
    view: FeatureSnapshot,
    features: tuple[FeatureRef, ...],
    fold: dict[str, int],
    name: str,
    config: BenchmarkConfig,
    plan: dict[str, Any],
    parent: ExperimentStore,
    lease: WriterLease,
) -> dict[str, Any]:
    """Recover even a fit finished immediately before a coordinator interruption.

    Workers have separate paths and loggers. Only the coordinator updates the study
    ledger; the shared S3 client is thread-safe. Trial receipts are conditional and
    bind the full fit specification, not just a name and completion flag.
    """
    lease.check()
    specification = {
        "view_sha256": view.manifest_sha256,
        "features": [asdict(f) for f in features],
        "fold": fold,
        "experiment": name,
        "config": asdict(config),
    }
    digest = sha256_bytes(canonical_json_bytes(specification))
    key = f"{parent.prefix}/trials/{name}_{fold['fold']}.json"
    existing = parent.read(key)
    if existing is not None:
        record = dict(json.loads(existing[0]))
        require(record["fit_spec_sha256"] == digest, "completed fit specification changed")
        for member in record["artifacts"].values():
            restore_member(parent, member)
        return record
    logger = ReleaseLogger(f"history-{name}-{fold['fold']}", parent.root / "logs")
    worker = ExperimentStore(parent.client, parent.bucket, parent.prefix, parent.root, logger)
    record = run_fold(view, features, (), fold, name, config, plan, worker, logger)
    record.update(
        model_fit_performed=True,
        fit_spec_sha256=digest,
        history_view_sha256=view.manifest_sha256,
        original_features=700,
        history_features=len(features) - 700,
    )
    lease.check()
    payload = canonical_json_bytes(record)
    parent.put(key, payload, IfNoneMatch="*")
    verified = parent.read(key)
    require(verified is not None and verified[0] == payload, "trial receipt read-back failed")
    return record


def run(root: Path, bucket: str) -> dict[str, Any]:
    """Execute a bounded study, resuming verified source blocks and individual fits."""
    plan = read_object(root / "configs/history_research.json")
    protocol = read_object(root / "configs/validation_protocol.json")
    validate_plan(plan, protocol)
    verify_file(root / "uv.lock", plan["lock_sha256"])
    screen_path = root / "reports/feature_ablation/feature_screen.json"
    verify_file(screen_path, plan["control_feature_sha256"])
    original = feature_refs_from_payload(read_object(screen_path)["result"])
    require(len(original) == 700, "control feature set changed")
    config, _ = BenchmarkConfig.load(root / "configs/model_benchmark.json")
    require(
        config.seed == plan["seed"] and config.threads == plan["threads"], "fit settings changed"
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
        "history research requires clean committed source",
    )
    identity = {
        "source_commit": source_commit,
        "plan_sha256": sha256_file(root / "configs/history_research.json"),
        "benchmark_config_sha256": sha256_file(root / "configs/model_benchmark.json"),
        "raw_manifest_sha256": plan["raw_manifest_sha256"],
        "feature_manifest_sha256": plan["feature_manifest_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "lock_sha256": plan["lock_sha256"],
    }
    study_key = sha256_bytes(canonical_json_bytes(identity))
    directory = root / "artifacts/history_research" / study_key
    directory.mkdir(parents=True, exist_ok=True)
    logger = ReleaseLogger("history-research", root / "logs")
    store = ExperimentStore(
        boto3.client(
            "s3",
            region_name="us-west-2",
            config=Config(
                retries={"mode": "standard", "total_max_attempts": 5},
                max_pool_connections=32,
            ),
        ),
        bucket,
        f"home-credit-model-stability/history-research/{study_key}",
        directory,
        logger,
    )
    with (directory / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with WriterLease(store) as lease, StageTimer(logger, "history_study", heartbeat_seconds=15):
            state = store.restore(identity)
            if state is None:
                state = {
                    "schema_version": 1,
                    "identity": identity,
                    "revision": 0,
                    "trials": [],
                    "stages": {},
                    "complete": False,
                }
                store.commit(state)
            for trial in state["trials"]:
                for member in trial["artifacts"].values():
                    restore_member(store, member)
            if state["complete"]:
                report = read_object(restore_member(store, state["stages"]["report"]))
                logger.event(
                    "history_research_reused", model_fits=0, completed=len(state["trials"])
                )
                return report
            snapshot = restore_snapshot(root, plan, store)
            view, specs, state = build_history_snapshot(
                snapshot, plan, protocol, store, lease, state
            )
            if "screen" not in state["stages"]:
                with StageTimer(logger, "screen_history_candidates", heartbeat_seconds=15):
                    member = screen_history(view, specs, config, plan, store)
                    state = commit_stage(store, lease, state, "screen", member)
            screen = read_object(restore_member(store, state["stages"]["screen"]))
            selected = tuple(HistoryFeature(**s) for s in screen["selected"])
            conditions = {
                "history_shape": (*original, *(s.ref for s in selected)),
                "without_history_skew": (
                    *original,
                    *(s.ref for s in selected if s.statistic != "skew"),
                ),
            }
            completed = {(t["experiment"], t["fold"]) for t in state["trials"]}
            require(len(completed) == len(state["trials"]), "duplicate trial")
            identical = conditions["history_shape"] == conditions["without_history_skew"]
            with ThreadPoolExecutor(max_workers=plan["max_concurrent_fits"]) as pool:
                pending = []
                for fold in protocol["inner_temporal_cv"]["folds"]:
                    for name, features in conditions.items():
                        if (name, fold["fold"]) in completed:
                            logger.event("history_fold_reused", experiment=name, fold=fold["fold"])
                            continue
                        if identical and name == "without_history_skew":
                            continue
                        pending.append(
                            pool.submit(
                                checkpointed_fold,
                                view,
                                features,
                                fold,
                                name,
                                config,
                                plan,
                                store,
                                lease,
                            )
                        )
                for future in as_completed(pending):
                    record = future.result()
                    updated = copy.deepcopy(state)
                    updated["trials"].append(record)
                    updated["trials"].sort(key=lambda t: (t["fold"], t["experiment"]))
                    updated["revision"] += 1
                    lease.check()
                    store.commit(updated)
                    state = updated
                    logger.event(
                        "history_fit_completed",
                        completed=len(state["trials"]),
                        total=10,
                        experiment=record["experiment"],
                        fold=record["fold"],
                        **record["metrics"],
                    )
            if identical:
                for trial in list(state["trials"]):
                    pair = ("without_history_skew", trial["fold"])
                    if trial["experiment"] != "history_shape" or pair in completed:
                        continue
                    record = copy.deepcopy(trial)
                    record.update(
                        experiment="without_history_skew",
                        model_fit_performed=False,
                        reused_from="history_shape",
                    )
                    updated = copy.deepcopy(state)
                    updated["trials"].append(record)
                    updated["trials"].sort(key=lambda t: (t["fold"], t["experiment"]))
                    updated["revision"] += 1
                    lease.check()
                    store.commit(updated)
                    state = updated
            control = directory / "control.parquet"
            store.download(plan["control"]["object_key"], control, plan["control"]["sha256"])
            require(control.stat().st_size == plan["control"]["bytes"], "control length changed")
            frames = {"control": pl.read_parquet(control)}
            for name in conditions:
                frames[name] = pl.concat(
                    [
                        pl.read_parquet(restore_member(store, t["artifacts"]["predictions"]))
                        for t in state["trials"]
                        if t["experiment"] == name
                    ]
                )
            result = compare_predictions(frames, protocol["inner_temporal_cv"]["folds"])
            require(len(state["trials"]) == 10, "incomplete history contrasts")
            result.update(
                scope=plan["scope"],
                identity=identity,
                study_key=study_key,
                holdout_previously_opened=True,
                release_changed=False,
                new_model_fits=sum(t["model_fit_performed"] for t in state["trials"]),
                screen_model_fits=2,
                aligned_oof_cases=len(frames["control"]),
                original_features=700,
                additional_candidates=screen["generated"],
                additional_retained=screen["retained"],
                additional_rejected=screen["rejected"],
                retained_statistics=dict(Counter(s.statistic for s in selected)),
                retained_sources=dict(Counter(s.ref.block for s in selected)),
                rejection_counts=screen["rejection_counts"],
                fit_records=state["trials"],
                history_manifest=state["stages"]["history_manifest"],
                screen=state["stages"]["screen"],
            )
            result["fold_sensitivity"] = fold_sensitivity(result)
            require(
                bool(np.isfinite([r["mean_fold_stability"] for r in result["rows"]]).all()),
                "nonfinite result",
            )
            save_json(directory / "comparison.json", result)
            member = publish_verified(store, directory / "comparison.json", "comparison.json")
            state = commit_stage(store, lease, state, "report", member)
            updated = copy.deepcopy(state)
            updated["complete"] = True
            updated["revision"] += 1
            store.commit(updated)
            logger.event(
                "history_research_completed",
                study_key=study_key,
                new_model_fits=result["new_model_fits"],
            )
            return result
