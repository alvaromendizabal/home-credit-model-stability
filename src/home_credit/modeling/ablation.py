"""Controlled feature removal and independently recomputed ablation comparisons."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from home_credit.metrics.classification import evaluate_probabilities
from home_credit.modeling.acceptance import read_json, require, validate_predictions
from home_credit.modeling.checkpoints import atomic_write, sha256_file
from home_credit.modeling.data import FeatureRef, FeatureSnapshot
from home_credit.modeling.screening import feature_refs_from_payload

if TYPE_CHECKING:
    from home_credit.modeling.config import BenchmarkConfig


def frozen_features(config: BenchmarkConfig, snapshot: FeatureSnapshot) -> tuple[FeatureRef, ...]:
    """Remove whole blocks from the verified original list, without re-screening."""
    policy = config.feature_selection
    require(policy is not None, "frozen feature policy is missing")
    assert policy is not None
    path = Path(policy["path"])
    require(sha256_file(path) == policy["sha256"], "frozen feature file hash mismatch")
    source = read_json(path)
    identity = source["identity"]
    require(identity["smoke"] is False, "cannot freeze features from a smoke run")
    require(
        identity["feature_manifest_sha256"] == snapshot.manifest_sha256, "feature identity mismatch"
    )
    require(
        identity["validation_protocol_sha256"] == snapshot.protocol_sha256,
        "protocol identity mismatch",
    )
    require(
        source["screening_config"]["validation_week_max"] <= 32,
        "screening overlaps development evaluation",
    )
    features = feature_refs_from_payload(source)
    candidates = set(snapshot.candidate_features(excluded=frozenset(config.excluded_predictors)))
    require(
        all(f in candidates for f in features), "frozen feature absent or excluded from snapshot"
    )
    excluded = set(policy["exclude_blocks"])
    require(excluded <= {f.block for f in features}, "unknown excluded feature block")
    selected = tuple(f for f in features if f.block not in excluded)
    require(bool(selected), "ablation removed every feature")
    return selected


def feature_payload(features: tuple[FeatureRef, ...]) -> dict[str, Any]:
    """Serialize the retained ordered feature list."""
    return {"selected_features": [asdict(f) for f in features]}


def compare_predictions(
    frames: dict[str, pl.DataFrame], folds: list[dict[str, int]]
) -> dict[str, Any]:
    """Compare aligned cases on every frozen fold; never trust saved summary numbers."""
    require("control" in frames, "control predictions are required")
    expected_weeks = {
        w for f in folds for w in range(f["validation_week_min"], f["validation_week_max"] + 1)
    }
    require(bool(expected_weeks) and max(expected_weeks) < 73, "holdout overlap")
    require(len({f["fold"] for f in folds}) == len(folds), "duplicate folds")
    require(
        sum(f["validation_week_max"] - f["validation_week_min"] + 1 for f in folds)
        == len(expected_weeks),
        "overlapping folds",
    )
    control = frames["control"].sort("case_id")
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for name, raw in frames.items():
        frame = raw.sort("case_id")
        validate_predictions(frame, expected_weeks, name)
        require(
            frame.select("case_id", "WEEK_NUM", "target").equals(
                control.select("case_id", "WEEK_NUM", "target")
            ),
            "cross-experiment case alignment mismatch",
        )
        current = []
        for fold in folds:
            subset = frame.filter(
                pl.col("WEEK_NUM").is_between(
                    fold["validation_week_min"], fold["validation_week_max"]
                )
            )
            metrics = evaluate_probabilities(
                subset["target"].to_numpy(),
                subset["prediction"].to_numpy(),
                subset["WEEK_NUM"].to_numpy(),
            )
            current.append(metrics)
            fold_rows.append({"experiment": name, "fold": fold["fold"], **metrics})
        pooled = evaluate_probabilities(
            frame["target"].to_numpy(), frame["prediction"].to_numpy(), frame["WEEK_NUM"].to_numpy()
        )
        rows.append(
            {
                "experiment": name,
                "mean_fold_stability": float(np.mean([r["stability_score"] for r in current])),
                "worst_fold_stability": min(r["stability_score"] for r in current),
                **{f"oof_{k}": pooled[k] for k in ("auc", "pr_auc", "brier_score", "log_loss")},
            }
        )
    control_score = next(r["mean_fold_stability"] for r in rows if r["experiment"] == "control")
    for row in rows:
        row["delta_vs_control"] = row["mean_fold_stability"] - control_score
    rows.sort(
        key=lambda r: (-r["mean_fold_stability"], -r["worst_fold_stability"], r["experiment"])
    )
    return {
        "schema_version": 1,
        "scope": "development feature ablation; final holdout remains locked",
        "rows": rows,
        "folds": fold_rows,
        "outer_holdout_touched": False,
    }


def write_report(result: dict[str, Any], destination: Path) -> None:
    """Write an offline HTML comparison with a score chart and full metric table."""
    import html

    import plotly.graph_objects as go

    rows = result["rows"]
    fig = go.Figure(
        go.Bar(
            x=[r["experiment"] for r in rows],
            y=[r["delta_vs_control"] for r in rows],
            marker_color=["#087f8c" if r["delta_vs_control"] >= 0 else "#be4b45" for r in rows],
            hovertemplate="%{x}<br>Stability change: %{y:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Does removing a feature block improve stability?",
        yaxis_title="Mean fold stability change vs control",
        height=450,
    )
    keys = list(rows[0])
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in keys)
        + "</tr></thead><tbody>"
    )
    for row in rows:
        formatted = [f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in keys]
        table += "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in formatted) + "</tr>"
    table += "</tbody></table>"
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Home Credit | Feature ablation</title><style>
body{font:16px system-ui;margin:48px auto;max-width:1280px;color:#19334a;
background:#f7f9fc;padding:0 24px}h1{font-size:36px}
table{border-collapse:collapse;width:100%;font-size:14px;background:white}
td,th{padding:12px;text-align:left;border-bottom:1px solid #dbe3ea}th{color:#087f8c}
section{overflow:auto}p{max-width:900px;line-height:1.6}</style>
<h1>Home Credit: Feature ablation</h1>
<p>Identical LightGBM settings, frozen temporal folds, and a fixed source feature list.
Only the named blocks are removed. These development results support experiment selection;
weeks 73-91 remain locked. Average precision is reported as PR AUC.
Early stopping uses the development folds.</p>"""
    label = (
        "SMOKE VALIDATION: capped data; not full benchmark results."
        if result.get("smoke")
        else "FULL DEVELOPMENT COMPARISON: final holdout pending."
    )
    page += f"<p><strong>{label}</strong></p>"
    page += (
        fig.to_html(full_html=False, include_plotlyjs=True, div_id="ablation-deltas")
        + "<section>"
        + table
        + "</section></html>"
    )
    atomic_write(destination, page.encode())
