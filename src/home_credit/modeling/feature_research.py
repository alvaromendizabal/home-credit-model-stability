"""Resumable, development-only feature expansion with controlled comparisons."""

from __future__ import annotations

import copy
import fcntl
import gc
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import polars as pl
from botocore.config import Config  # type: ignore[import-untyped]

from home_credit.features.research import (
    Hypothesis,
    case_features,
    catalog,
    peer_features,
    prune_candidates,
    required_sources,
)
from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.ablation import compare_predictions
from home_credit.modeling.acceptance import require, validate_predictions
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file
from home_credit.modeling.config import BenchmarkConfig
from home_credit.modeling.data import (
    FeatureRef,
    FeatureSnapshot,
    load_feature_frame,
    load_fold_frames,
)
from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
from home_credit.modeling.models import fit_lightgbm
from home_credit.modeling.release import fit_encoder, read_object, save_json, transform, verify_file
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


def hypothesis_from_dict(value: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        value["name"],
        value["family"],
        value["operation"],
        tuple(value["sources"]),
        value["rationale"],
    )


def validate_plan(plan: dict[str, Any], protocol: dict[str, Any]) -> None:
    require(plan["schema_version"] == 1, "unsupported research plan")
    require(plan["development_week_max"] == 72, "research cannot access the observed holdout")
    require(verify_protocol_sha256(protocol), "protocol digest changed")
    require(protocol["protocol_sha256"] == plan["protocol_sha256"], "different protocol")
    folds = protocol["inner_temporal_cv"]["folds"]
    require(len(folds) == 5 and len({f["fold"] for f in folds}) == 5, "five unique folds required")
    require(
        all(
            0
            <= f["train_week_min"]
            <= f["train_week_max"]
            < f["validation_week_min"]
            <= f["validation_week_max"]
            <= 72
            for f in folds
        ),
        "invalid research windows",
    )
    require(
        plan["experiments"]
        == ["wider_original", "engineered", "without_amount_ratios", "without_peer_statistics"],
        "experimental contrasts changed",
    )
    require(plan["new_model_fit_budget"] == 20, "unexpected fit budget")
    require(700 < plan["wider_original_features"] <= 2034, "invalid feature budget")
    require(32 <= plan["additional_features"] <= 512, "invalid extension budget")


def screen_extension(
    snapshot: FeatureSnapshot,
    config: BenchmarkConfig,
    hypotheses: tuple[Hypothesis, ...],
    root: Path,
    store: ExperimentStore,
    logger: ReleaseLogger,
    limit: int,
) -> dict[str, Any]:
    """Screen only in the same early windows as the original feature selection."""
    source_names = required_sources(hypotheses)
    source_refs = tuple(f for f in snapshot.features if f.name in source_names)
    require(len(source_refs) == len(source_names), "hypothesis source absent from frozen snapshot")
    with StageTimer(logger, "materialize_early_feature_hypotheses", heartbeat_seconds=15):
        train = load_feature_frame(
            snapshot,
            source_refs,
            week_min=0,
            week_max=24,
            max_rows=config.screening.max_train_rows,
            seed=config.seed,
        )
        valid = load_feature_frame(
            snapshot,
            source_refs,
            week_min=25,
            week_max=32,
            max_rows=config.screening.max_validation_rows,
            seed=config.seed + 1,
        )
        train_case, valid_case = case_features(train, hypotheses), case_features(valid, hypotheses)
        train_peer, valid_peer, peer_state = peer_features(train, valid, hypotheses)
        train = train.select("case_id", "WEEK_NUM", "target").hstack(train_case).hstack(train_peer)
        valid = valid.select("case_id", "WEEK_NUM", "target").hstack(valid_case).hstack(valid_peer)
        del train_case, valid_case, train_peer, valid_peer
        gc.collect()
    with StageTimer(logger, "prune_structural_duplicates", heartbeat_seconds=15):
        pruned, records = prune_candidates(train, hypotheses)
    with StageTimer(logger, "screen_early_target_and_drift", heartbeat_seconds=15):
        result = screen_features(
            train,
            valid,
            tuple(h.ref for h in pruned),
            config=replace(config.screening, max_features=limit),
            seed=config.seed,
            threads=config.threads,
            logger=logger,
        )
    lookup = {h.name: h for h in hypotheses}
    selected = [asdict(lookup[f.name]) for f in result.selected_features]
    scores = {s.name: asdict(s) for s in result.scores}
    for record in records:
        score = scores.get(record["name"])
        if record["structural_rejection"]:
            record["rejection_reason"] = record["structural_rejection"]
        elif score is not None and score["selected"]:
            record["rejection_reason"] = None
        elif (
            score is not None and score["missing_fraction"] > config.screening.max_missing_fraction
        ):
            record["rejection_reason"] = "missingness"
        elif score is not None and score["categorical"] and score["unique_values"] > 5000:
            record["rejection_reason"] = "cardinality"
        else:
            record["rejection_reason"] = "early_ranking_budget"
        if score:
            record["screen"] = score
    payload = {
        "schema_version": 1,
        "scope": "post-release development exploration",
        "screen_weeks": {"train": [0, 24], "validation": [25, 32]},
        "train_rows": len(train),
        "validation_rows": len(valid),
        "generated": len(hypotheses),
        "retained": len(selected),
        "rejected": len(hypotheses) - len(selected),
        "rejection_counts": dict(
            Counter(r["rejection_reason"] for r in records if r["rejection_reason"])
        ),
        "catalog": records,
        "selected": selected,
        "result": result.to_payload(),
    }
    screen_path = store.root / "screen.json"
    save_json(screen_path, payload)
    peer_path = store.root / "screen_peer_references.json"
    save_json(peer_path, peer_state)
    return {
        "screen": publish_verified(store, screen_path, "screen.json"),
        "peer_references": publish_verified(store, peer_path, "screen_peer_references.json"),
    }


def add_hypotheses(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    hypotheses: tuple[Hypothesis, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    a, b, reference = peer_features(train, valid, hypotheses)
    return (
        train.hstack(case_features(train, hypotheses)).hstack(a),
        valid.hstack(case_features(valid, hypotheses)).hstack(b),
        reference,
    )


def diagnostics(
    model: lgb.Booster,
    x: Any,
    valid: pl.DataFrame,
    train: Any,
    features: tuple[FeatureRef, ...],
    plan: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fold-specific importance, exact TreeSHAP additivity and within-week permutation."""
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(valid), min(plan["permutation_rows"], len(valid)), replace=False))
    sample, frame = x[take], valid[take]
    baseline = np.asarray(model.predict(sample, num_threads=plan["threads"]), dtype=np.float64)
    original = evaluate_probabilities(
        frame["target"].to_numpy(), baseline, frame["WEEK_NUM"].to_numpy()
    )
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(features):
        group = f.family if f.name.startswith("research__") else "original_features"
        groups.setdefault(group, []).append(i)
    permutation = []
    for group, columns in sorted(groups.items()):
        if group == "original_features":
            continue
        for repeat in range(plan["permutation_repeats"]):
            copy_x = sample.copy()
            for week in frame["WEEK_NUM"].unique().to_list():
                rows = np.flatnonzero(frame["WEEK_NUM"].to_numpy() == week)
                copy_x[np.ix_(rows, columns)] = sample[np.ix_(rng.permutation(rows), columns)]
            prediction = np.asarray(
                model.predict(copy_x, num_threads=plan["threads"]), dtype=np.float64
            )
            metrics = evaluate_probabilities(
                frame["target"].to_numpy(), prediction, frame["WEEK_NUM"].to_numpy()
            )
            permutation.append(
                {
                    "family": group,
                    "repeat": repeat,
                    "stability_decrease": original["stability_score"] - metrics["stability_score"],
                    "auc_decrease": original["auc"] - metrics["auc"],
                }
            )
    contributions = np.asarray(
        model.predict(sample[: plan["shap_rows"]], pred_contrib=True, num_threads=plan["threads"]),
        dtype=np.float64,
    )
    margin = model.predict(sample[: plan["shap_rows"]], raw_score=True, num_threads=plan["threads"])
    require(
        bool(np.allclose(contributions.sum(axis=1), margin, atol=1e-6, rtol=1e-6)),
        "TreeSHAP contributions do not sum to the model margin",
    )
    importance = [
        {"name": f.name, "family": f.family, "gain": float(gain), "mean_abs_shap": float(shap)}
        for f, gain, shap in zip(
            features,
            model.feature_importance(importance_type="gain"),
            np.abs(contributions[:, :-1]).mean(axis=0),
            strict=True,
        )
    ]
    # Pairwise redundancy is descriptive; do not remove features using validation diagnostics.
    top = np.argsort(-model.feature_importance(importance_type="gain"))[:80]
    sample_train = train[:: max(1, len(train) // 4000), :][:4000, top].astype(np.float64)
    pairs = []
    for i in range(len(top)):
        for j in range(i):
            a, b = sample_train[:, i], sample_train[:, j]
            keep = np.isfinite(a) & np.isfinite(b)
            if keep.sum() >= 100 and np.std(a[keep]) > 0 and np.std(b[keep]) > 0:
                corr = float(np.corrcoef(a[keep], b[keep])[0, 1])
                if abs(corr) >= 0.98:
                    pairs.append(
                        {
                            "a": features[top[i]].name,
                            "b": features[top[j]].name,
                            "training_correlation": corr,
                            "paired_rows": int(keep.sum()),
                        }
                    )
    return {
        "sample_rows": len(sample),
        "shap_rows": len(contributions),
        "importance": importance,
        "permutation": permutation,
        "redundancy": pairs,
        "interpretation": "Predictive associations, not causal effects. Permutation preserves "
        "weeks but can break dependencies with unpermuted features. SHAP uses tree-path "
        "contributions on deterministic validation samples; not a population fairness audit.",
    }


def run_fold(
    snapshot: FeatureSnapshot,
    original: tuple[FeatureRef, ...],
    hypotheses: tuple[Hypothesis, ...],
    fold: dict[str, int],
    name: str,
    config: BenchmarkConfig,
    plan: dict[str, Any],
    store: ExperimentStore,
    logger: ReleaseLogger,
) -> dict[str, Any]:
    lookup = {f.name: f for f in snapshot.features}
    require(
        0
        <= fold["train_week_min"]
        <= fold["train_week_max"]
        < fold["validation_week_min"]
        <= fold["validation_week_max"]
        <= 72,
        "a research fit cannot access or overlap the observed holdout",
    )
    source_names = {f.name for f in original} | required_sources(hypotheses)
    sources = tuple(lookup[n] for n in sorted(source_names))
    features = (*original, *(h.ref for h in hypotheses))
    directory = store.root / "folds" / name / str(fold["fold"])
    directory.mkdir(parents=True, exist_ok=True)
    with StageTimer(logger, f"materialize_{name}_{fold['fold']}", heartbeat_seconds=15):
        train, valid = load_fold_frames(
            snapshot,
            sources,
            seed=config.seed + fold["fold"] * 101,
            train_week_min=fold["train_week_min"],
            train_week_max=fold["train_week_max"],
            validation_week_min=fold["validation_week_min"],
            validation_week_max=fold["validation_week_max"],
            train_row_cap=None,
            validation_row_cap=None,
        )
        train, valid, peer_state = add_hypotheses(train, valid, hypotheses)
        selected_names = [f.name for f in features]
        train = train.select("case_id", "WEEK_NUM", "target", *selected_names)
        valid = valid.select("case_id", "WEEK_NUM", "target", *selected_names)
        encoder = fit_encoder(train, features)
        x_train, x_valid = transform(train, features, encoder), transform(valid, features, encoder)
        y_train, y_valid = (
            train["target"].to_numpy().astype(np.int8),
            valid["target"].to_numpy().astype(np.int8),
        )
        del train
        gc.collect()
    with StageTimer(logger, f"fit_{name}_{fold['fold']}", heartbeat_seconds=15):
        params = next(m.params for m in config.models if m.name == "lightgbm")
        fit = fit_lightgbm(
            x_train,
            y_train,
            x_valid,
            y_valid,
            params=params,
            seed=config.seed + fold["fold"],
            threads=config.threads,
            artifact_path=directory / "model.txt",
            feature_names=tuple(selected_names),
            logger=logger,
        )
        model = lgb.Booster(model_file=str(fit.artifact_path))
        require(
            bool(
                np.allclose(
                    model.predict(x_valid, num_threads=config.threads),
                    fit.prediction,
                    atol=1e-12,
                    rtol=0,
                )
            ),
            "native model round-trip differs",
        )
    frame = valid.select("case_id", "WEEK_NUM", "target").with_columns(
        pl.Series("prediction", fit.prediction)
    )
    validate_predictions(
        frame, set(range(fold["validation_week_min"], fold["validation_week_max"] + 1)), name
    )
    metrics = evaluate_probabilities(y_valid, fit.prediction, frame["WEEK_NUM"].to_numpy())
    frame.write_parquet(directory / "predictions.parquet")
    save_json(directory / "encoder.json", encoder)
    save_json(directory / "peer_references.json", peer_state)
    save_json(
        directory / "features.json",
        {"original": [asdict(f) for f in original], "hypotheses": [asdict(h) for h in hypotheses]},
    )
    if name == "engineered":
        with StageTimer(logger, f"interpret_{name}_{fold['fold']}", heartbeat_seconds=15):
            report = diagnostics(
                model, x_valid, valid, x_train, tuple(features), plan, config.seed + fold["fold"]
            )
        save_json(directory / "diagnostics.json", report)
    record: dict[str, Any] = {
        "experiment": name,
        "fold": fold["fold"],
        "features": len(features),
        "train_rows": len(x_train),
        "validation_rows": len(valid),
        "best_iteration": fit.best_iteration,
        "metrics": metrics,
        "artifacts": {},
    }
    for path in sorted(directory.iterdir()):
        record["artifacts"][path.stem] = publish_verified(
            store, path, path.relative_to(store.root).as_posix()
        )
    del model, x_train, x_valid, valid
    gc.collect()
    return record


def run(root: Path, bucket: str) -> dict[str, Any]:
    plan = read_object(root / "configs/feature_research.json")
    protocol = read_object(root / "configs/validation_protocol.json")
    validate_plan(plan, protocol)
    verify_file(root / "uv.lock", plan["lock_sha256"])
    config, _ = BenchmarkConfig.load(root / "configs/model_benchmark.json")
    verify_file(
        root / "reports/feature_ablation/feature_screen.json",
        "9cd20cf4a831f3acc8dcb73b8afedb281b8a61cd13758f87a0d87f57322e9c2d",
    )
    source = read_object(root / "reports/feature_ablation/feature_screen.json")["result"]
    original = feature_refs_from_payload(source)
    require(len(original) == 700, "control feature count changed")
    hypotheses = catalog(source["scores"])
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
        "research execution requires a clean committed source tree",
    )
    identity = {
        "source_commit": source_commit,
        "plan_sha256": sha256_file(root / "configs/feature_research.json"),
        "feature_manifest_sha256": plan["feature_manifest_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "lock_sha256": plan["lock_sha256"],
    }
    study_key = sha256_bytes(canonical_json_bytes(identity))
    directory = root / "artifacts/feature_research" / study_key
    directory.mkdir(parents=True, exist_ok=True)
    logger = ReleaseLogger("feature-research", root / "logs")
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(retries={"mode": "standard", "total_max_attempts": 5}),
    )
    store = ExperimentStore(
        client,
        bucket,
        f"home-credit-model-stability/feature-research/{study_key}",
        directory,
        logger,
    )
    with (directory / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with WriterLease(store) as lease:
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
            # Validate durable outputs on every resume before considering a stage complete.
            for record in state["trials"]:
                for member in record["artifacts"].values():
                    restore_member(store, member)
            if state["complete"]:
                report = read_object(restore_member(store, state["stages"]["report"]))
                logger.event(
                    "feature_research_reused", model_fits=0, completed=len(state["trials"])
                )
                return report
            with StageTimer(logger, "restore_frozen_feature_snapshot", heartbeat_seconds=15):
                snapshot = restore_snapshot(root, plan, store)
            if "screen" not in state["stages"]:
                record = screen_extension(
                    snapshot, config, hypotheses, root, store, logger, plan["additional_features"]
                )
                state = commit_stage(store, lease, state, "screen", record)
            screen = read_object(restore_member(store, state["stages"]["screen"]["screen"]))
            selected = tuple(hypothesis_from_dict(h) for h in screen["selected"])
            original_names = {f.name for f in original}
            eligible = [
                s
                for s in source["scores"]
                if s["unique_values"] > 1
                and s["missing_fraction"] <= config.screening.max_missing_fraction
                and (not s["categorical"] or s["unique_values"] <= 5000)
                and s["name"] not in original_names
            ]
            ranked = sorted(
                eligible, key=lambda s: (-s["selection_score"], -s["target_gain"], s["name"])
            )
            index = {f.name: f for f in snapshot.features}
            wider = (
                *original,
                *(index[s["name"]] for s in ranked[: plan["wider_original_features"] - 700]),
            )
            conditions = {
                "wider_original": (tuple(wider), ()),
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
            completed = {(t["experiment"], t["fold"]) for t in state["trials"]}
            require(len(completed) == len(state["trials"]), "duplicate completed fit")
            for fold in protocol["inner_temporal_cv"]["folds"]:
                for name, (features, specs) in conditions.items():
                    lease.check()
                    if (name, fold["fold"]) in completed:
                        logger.event("research_fold_reused", experiment=name, fold=fold["fold"])
                        continue
                    record = run_fold(
                        snapshot, features, specs, fold, name, config, plan, store, logger
                    )
                    updated = copy.deepcopy(state)
                    updated["trials"].append(record)
                    updated["revision"] += 1
                    lease.check()
                    store.commit(updated)
                    state = updated
                    logger.event(
                        "research_fit_completed",
                        completed=len(state["trials"]),
                        total=20,
                        experiment=name,
                        fold=fold["fold"],
                        **record["metrics"],
                    )
            control = directory / "control.parquet"
            store.download(plan["control"]["object_key"], control, plan["control"]["sha256"])
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
            result.update(
                scope=plan["scope"],
                holdout_previously_opened=True,
                release_changed=False,
                new_model_fits=len(state["trials"]),
                original_candidates=2508,
                additional_candidates=screen["generated"],
                additional_retained=screen["retained"],
                additional_rejected=screen["rejected"],
                identity=identity,
                study_key=study_key,
                candidate_families=dict(Counter(h.family for h in hypotheses)),
                retained_families=dict(Counter(h.family for h in selected)),
                rejection_counts=screen["rejection_counts"],
                fit_records=state["trials"],
                screen=state["stages"]["screen"],
            )
            save_json(directory / "comparison.json", result)
            member = publish_verified(store, directory / "comparison.json", "comparison.json")
            state = commit_stage(store, lease, state, "report", member)
            updated = copy.deepcopy(state)
            updated["complete"] = True
            updated["revision"] += 1
            store.commit(updated)
            logger.event(
                "feature_research_completed",
                new_model_fits=len(state["trials"]),
                study_key=study_key,
            )
            return result
