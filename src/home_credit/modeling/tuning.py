"""Replayable Optuna proposals and temporal selection from verified predictions."""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution

from home_credit.modeling.ablation import compare_predictions
from home_credit.modeling.acceptance import read_json, require
from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes, sha256_file


def load_plan(root: Path) -> dict[str, Any]:
    """Validate the bounded study and all inputs before any paid computation."""
    plan = read_json(root / "configs/model_tuning.json")
    require(plan["schema_version"] == 1, "unsupported tuning plan")
    require(type(plan["new_trials"]) is int and 1 <= plan["new_trials"] <= 32, "invalid budget")
    require(2 <= plan["startup_trials"] <= plan["new_trials"], "invalid startup budget")
    require(type(plan["seed"]) is int and plan["seed"] >= 0, "invalid sampler seed")
    for name, digest in plan["input_sha256"].items():
        require((root / name).resolve().is_relative_to(root.resolve()), "input escaped root")
        require(sha256_file(root / name) == digest, f"tuning input hash mismatch: {name}")
    protocol = read_json(root / "configs/validation_protocol.json")
    require(protocol["protocol_sha256"] == plan["protocol_sha256"], "protocol changed")
    require(protocol["outer_holdout"]["locked"] is True, "holdout must remain locked")
    folds = protocol["inner_temporal_cv"]["folds"]
    require(
        len(folds) == 5 and max(f["validation_week_max"] for f in folds) == 72,
        "tuning requires the five locked development folds",
    )
    require(
        set(search_space(plan))
        == {
            "num_leaves",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
        },
        "unexpected tuning parameters",
    )
    baseline = read_json(root / "configs/ablations/control.json")
    require(baseline["feature_selection"]["exclude_blocks"] == [], "retain all selected features")
    require(baseline["threads"] == 6, "thread budget changed")
    require(baseline["outer_holdout_guard_week_min"] == 73, "holdout guard changed")
    reference = read_json(root / "reports/feature_ablation/training_environment.json")
    lock = tomllib.loads((root / "uv.lock").read_text())
    for name, version in reference["training_and_reporting_versions"].items():
        require(
            {p["version"] for p in lock["package"] if p["name"] == name} == {version},
            f"baseline training/reporting package changed: {name}",
        )
    return plan


def search_space(plan: dict[str, Any]) -> dict[str, BaseDistribution]:
    """Build the same explicit distributions for every proposal and replay."""
    distributions: dict[str, BaseDistribution] = {}
    for key, spec in plan["search_space"].items():
        if spec["type"] == "categorical":
            distributions[key] = CategoricalDistribution(spec["choices"])
        else:
            require(spec["type"] == "float", "unknown distribution")
            distributions[key] = FloatDistribution(spec["low"], spec["high"], log=spec["log"])
    return distributions


def propose(plan: dict[str, Any], history: list[dict[str, Any]], slot: int) -> dict[str, Any]:
    """Rebuild completed Optuna history and draw one deterministically seeded proposal.

    The durable JSON ledger is authoritative. No pickled sampler, transient RNG state,
    or live database is needed to resume the exact next proposal.
    """
    require(slot >= 1 and len(history) == slot, "proposal history is not contiguous")
    sampler_seed = (int(plan["seed"]) + slot) % (2**32)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, n_startup_trials=plan["startup_trials"]
        ),
    )
    distributions = search_space(plan)
    for record in history:
        require(record["state"] == "complete", "cannot sample ahead of an incomplete trial")
        require(math.isfinite(record["value"]), "nonfinite trial objective")
        study.add_trial(
            optuna.trial.create_trial(
                params=record["params"],
                distributions=distributions,
                value=record["value"],
            )
        )
    trial = study.ask(fixed_distributions=distributions)
    return {
        "slot": slot,
        "name": f"trial_{slot:03d}",
        "state": "proposed",
        "params": trial.params,
        "sampler_seed": sampler_seed,
        "history_sha256": sha256_bytes(canonical_json_bytes(history)),
    }


def trial_config(base: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """Change model hyperparameters in the canonical benchmark configuration."""
    result = copy.deepcopy(base)
    result["name"] = f"lightgbm_{proposal['name']}"
    result["models"]["lightgbm"].update(proposal["params"])
    # Model API requires an integer count; sampling uses a finite categorical space.
    for key in ("num_leaves", "min_data_in_leaf"):
        result["models"]["lightgbm"][key] = int(result["models"]["lightgbm"][key])
    result["notes"] = "Bounded Optuna development study. Holdout weeks 73-91 remain locked."
    return result


def evaluate_trial(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    folds: list[dict[str, int]],
    name: str,
) -> dict[str, Any]:
    """Recompute official-metric components and probability metrics on aligned cases."""
    comparison = compare_predictions({"control": control, name: candidate}, folds)
    row = next(row for row in comparison["rows"] if row["experiment"] == name)
    components = [row for row in comparison["folds"] if row["experiment"] == name]
    row.update(
        {
            "mean_weekly_gini": float(np.mean([r["mean_gini"] for r in components])),
            "mean_temporal_slope": float(np.mean([r["temporal_slope"] for r in components])),
            "mean_residual_std": float(np.mean([r["residual_std"] for r in components])),
            "mean_brier_score": float(np.mean([r["brier_score"] for r in components])),
        }
    )
    return {"metrics": row, "folds": components, "value": row["mean_fold_stability"]}


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the frozen selection policy, retaining the control on an exact tie."""

    def key(record: dict[str, Any]) -> tuple[float, ...]:
        r = record["metrics"]
        return (
            -r["mean_fold_stability"],
            -r["worst_fold_stability"],
            -r["mean_weekly_gini"],
            -r["mean_temporal_slope"],
            r["mean_residual_std"],
            r["mean_brier_score"],
            float(record["slot"]),
        )

    return sorted([r for r in records if r["state"] == "complete"], key=key)
