"""Offline review of raw-history experiments with enforced evidence lineage."""

from __future__ import annotations

import base64
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from home_credit.features.distributions import HistoryFeature
from home_credit.modeling.acceptance import compare_number, require
from home_credit.modeling.checkpoints import sha256_file
from home_credit.modeling.history_research import fold_sensitivity, validate_plan
from home_credit.modeling.release import read_object, verify_file

LABELS = {
    "control": "Original 700",
    "history_shape": "Original + history shape",
    "without_history_skew": "Original + history, without skew",
}


def load_evidence(root: Path) -> dict[str, Any]:
    """Reject stale hashes and coherent but scientifically invalid rewritten results."""
    policy = read_object(root / "configs/history_research_review.json")
    plan = read_object(root / "configs/history_research.json")
    protocol = read_object(root / "configs/validation_protocol.json")
    validate_plan(plan, protocol)
    require(policy["schema_version"] == 1, "unsupported history publication")
    require(
        set(policy["files"])
        == {
            "comparison.json",
            "study.json",
            "screen.json",
            "history_manifest.json",
            "verification.json",
        },
        "incomplete history publication",
    )
    data = {}
    for name, member in policy["files"].items():
        path = root / "reports/history_research" / name
        verify_file(path, member["sha256"])
        require(path.stat().st_size == member["bytes"], "history publication size changed")
        data[name] = read_object(path)
    result, study = data["comparison.json"], data["study.json"]
    screen, manifest = data["screen.json"], data["history_manifest.json"]
    verification = data["verification.json"]
    identity = result["identity"]
    require(study["complete"] is True and study["identity"] == identity, "incomplete history study")
    require(identity["source_commit"] == policy["source_commit"], "history source changed")
    for name, path in (
        ("plan_sha256", "configs/history_research.json"),
        ("benchmark_config_sha256", "configs/model_benchmark.json"),
        ("lock_sha256", "uv.lock"),
    ):
        require(identity[name] == sha256_file(root / path), "history execution inputs changed")
    require(
        identity["protocol_sha256"] == plan["protocol_sha256"]
        and identity["raw_manifest_sha256"] == plan["raw_manifest_sha256"]
        and identity["feature_manifest_sha256"] == plan["feature_manifest_sha256"],
        "history snapshot lineage changed",
    )
    require(
        result["holdout_previously_opened"] is True
        and result["outer_holdout_touched"] is False
        and result["release_changed"] is False,
        "research scope changed",
    )
    for name, member in (
        ("comparison.json", study["stages"]["report"]),
        ("screen.json", result["screen"]),
        ("history_manifest.json", result["history_manifest"]),
    ):
        require(
            {k: member[k] for k in ("bytes", "sha256")} == policy["files"][name],
            "publication differs from durable study",
        )
    require(
        study["stages"]["screen"] == result["screen"]
        and study["stages"]["history_manifest"] == result["history_manifest"],
        "study stages changed",
    )
    require(
        manifest["source_commit"] == identity["source_commit"]
        and manifest["plan_sha256"] == identity["plan_sha256"]
        and manifest["base_feature_manifest_sha256"] == plan["feature_manifest_sha256"]
        and manifest["raw_manifest_sha256"] == plan["raw_manifest_sha256"],
        "history view lineage changed",
    )
    require(
        manifest["development_weeks"] == [0, 72] and manifest["development_rows"] == 1323314,
        "history population changed",
    )
    audits = manifest["source_audits"]
    require(
        bool(audits)
        and all(
            a["chronology_admitted"] is False and a["source_order_is_time"] is False for a in audits
        ),
        "unsupported chronology admitted",
    )
    for audit in audits:
        require(
            all(
                d["event_order_verified"] is False and d["availability_time_verified"] is False
                for d in audit["date_fields"]
            ),
            "date evidence was not verified",
        )
    catalog = screen["catalog"]
    selected = tuple(HistoryFeature(**s) for s in screen["selected"])
    selected_names = {s.name for s in selected}
    rejected = Counter(r["rejection_reason"] for r in catalog if r["rejection_reason"])
    require(
        len(catalog)
        == len({r["name"] for r in catalog})
        == screen["generated"]
        == result["additional_candidates"],
        "candidate accounting changed",
    )
    require(
        len(selected)
        == len(selected_names)
        == screen["retained"]
        == result["additional_retained"]
        <= 96,
        "selection count changed",
    )
    require(
        selected_names == {r["name"] for r in catalog if r["rejection_reason"] is None},
        "selected catalog differs",
    )
    require(
        sum(rejected.values())
        == screen["rejected"]
        == result["additional_rejected"]
        == len(catalog) - len(selected),
        "rejection counts changed",
    )
    require(
        dict(rejected) == screen["rejection_counts"] == result["rejection_counts"],
        "rejection reasons changed",
    )
    require(
        screen["screen_weeks"] == plan["screen_weeks"]
        and screen["train_rows"] == 160000
        and screen["validation_rows"] == 80000,
        "early selection population changed",
    )
    require(
        Counter(s.statistic for s in selected) == result["retained_statistics"]
        and Counter(s.ref.block for s in selected) == result["retained_sources"],
        "selected family accounting changed",
    )
    fits = result["fit_records"]
    grid = {(e, f) for e in plan["experiments"] for f in range(1, 6)}
    require(
        fits == study["trials"]
        and len(fits) == 10
        and {(t["experiment"], t["fold"]) for t in fits} == grid,
        "incomplete fit ledger",
    )
    require(
        result["new_model_fits"] == sum(t["model_fit_performed"] for t in fits) <= 10
        and result["screen_model_fits"] == 2
        and result["aligned_oof_cases"] == 727187,
        "fit budget or population changed",
    )
    require(
        len(result["rows"]) == 3 and {r["experiment"] for r in result["rows"]} == set(LABELS),
        "incomplete comparison",
    )
    folds = {(r["experiment"], r["fold"]): r for r in result["folds"]}
    require(
        len(result["folds"]) == 15 and set(folds) == {(e, f) for e in LABELS for f in range(1, 6)},
        "incomplete fold metrics",
    )
    control = next(r for r in result["rows"] if r["experiment"] == "control")
    for row in result["rows"]:
        name = row["experiment"]
        scores = [folds[name, f]["stability_score"] for f in range(1, 6)]
        compare_number(float(np.mean(scores)), row["mean_fold_stability"], "history mean")
        compare_number(min(scores), row["worst_fold_stability"], "history worst fold")
        compare_number(
            row["mean_fold_stability"] - control["mean_fold_stability"],
            row["delta_vs_control"],
            "history delta",
        )
        if name == "control":
            continue
        trials = [t for t in fits if t["experiment"] == name]
        require(
            sum(t["validation_rows"] for t in trials) == 727187,
            "history validation coverage changed",
        )
        require(
            len({t["artifacts"]["features"]["sha256"] for t in trials}) == 1,
            "history features vary by fold",
        )
        width = 700 + sum(name == "history_shape" or s.statistic != "skew" for s in selected)
        for trial in trials:
            require(
                trial["features"] == width
                and trial["original_features"] == 700
                and trial["history_features"] == width - 700
                and trial["history_view_sha256"]
                == policy["files"]["history_manifest.json"]["sha256"],
                "history model recipe changed",
            )
            for metric, value in trial["metrics"].items():
                compare_number(value, folds[name, trial["fold"]][metric], metric)
    require(result["fold_sensitivity"] == fold_sensitivity(result), "history sensitivity changed")
    require(
        verification["status"] == "passed"
        and verification["identity"] == identity
        and verification["study_key"] == result["study_key"]
        and verification["comparison_sha256"] == policy["files"]["comparison.json"]["sha256"],
        "history verification lineage changed",
    )
    require(
        verification["native_models_replayed"] == 10
        and verification["predictions_replayed"] == 1454374
        and verification["metric_identities_checked"] == 141
        and verification["model_fits"] == 0
        and verification["holdout_accessed"] is False,
        "incomplete independent verification",
    )
    require(
        all(
            0 <= verification[k] <= 1e-12
            for k in ("maximum_prediction_absolute_error", "maximum_metric_absolute_error")
        ),
        "history replay failed",
    )
    verify_file(root / "scripts/verify_history_research.py", policy["verifier_sha256"])
    return {
        "result": result,
        "screen": screen,
        "manifest": manifest,
        "verification": verification,
        "study": study,
    }


def chart_pairs(evidence: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Static and Plotly views use the same verified numerical arrays."""
    result = evidence["result"]
    rows = {r["experiment"]: r for r in result["rows"]}
    names = list(LABELS)
    labels = list(LABELS.values())
    pairs = {}
    static, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    chart = go.Figure()
    for key, label, color in (
        ("mean_fold_stability", "Mean fold", "#176b87"),
        ("worst_fold_stability", "Worst fold", "#d0752e"),
    ):
        values = [rows[n][key] for n in names]
        ax.scatter(values, labels, label=label, color=color, s=55)
        chart.add_scatter(
            x=values, y=labels, mode="markers", name=label, marker={"color": color, "size": 11}
        )
    ax.invert_yaxis()
    ax.set(xlabel="Official weekly Gini stability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    pairs["Do raw histories improve temporal stability?"] = static, chart
    folds = {(r["experiment"], r["fold"]): r["stability_score"] for r in result["folds"]}
    contrasts = names[1:]
    changes = [[folds[n, f] - folds["control", f] for f in range(1, 6)] for n in contrasts]
    bound = max(float(np.max(np.abs(changes))), 1e-8)
    static, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    heat = ax.imshow(changes, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set(
        xticks=range(5),
        xticklabels=range(1, 6),
        xlabel="Development fold",
        yticks=range(2),
        yticklabels=[LABELS[n] for n in contrasts],
    )
    for i, values in enumerate(changes):
        for j, value in enumerate(values):
            ax.text(
                j,
                i,
                f"{value:+.4f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.65 * bound else "#172334",
            )
    static.colorbar(heat, ax=ax, label="Change vs original 700")
    chart = go.Figure(
        go.Heatmap(
            z=changes,
            x=list(range(1, 6)),
            y=[LABELS[n] for n in contrasts],
            zmin=-bound,
            zmax=bound,
            colorscale="RdBu",
            texttemplate="%{z:+.4f}",
        )
    )
    pairs["History features: blue improves, red worsens"] = static, chart
    for title, (static, chart) in pairs.items():
        static.axes[0].set_title(title, loc="left", fontsize=12, pad=12)
        chart.update_layout(
            title=title,
            template="plotly_white",
            height=420,
            margin={"l": 230, "r": 40, "t": 70, "b": 55},
            font={"family": "Arial", "size": 13},
            yaxis_autorange="reversed",
        )
    return pairs


def display_results(evidence: dict[str, Any]) -> None:
    from IPython.display import display

    result = evidence["result"]
    widths = {t["experiment"]: t["features"] for t in result["fit_records"]}
    rows = pd.DataFrame(result["rows"])
    rows.insert(1, "features", rows["experiment"].map({"control": 700, **widths}))
    rows = rows.set_index("experiment").rename(index=LABELS)
    display(rows.round(6))  # type: ignore[no-untyped-call]
    complexity = pd.DataFrame(
        [
            {
                "condition": LABELS[t["experiment"]],
                "rounds": t["best_iteration"],
                "native_model_MB": t["artifacts"]["model"]["bytes"] / 1_000_000,
            }
            for t in result["fit_records"]
        ]
    )
    display(complexity.groupby("condition").agg(["mean", "min", "max"]).round(3))  # type: ignore[no-untyped-call]
    for static, chart in chart_pairs(evidence).values():
        try:
            image = BytesIO()
            static.savefig(image, format="png", dpi=140)
            display(
                {
                    "image/png": base64.b64encode(image.getvalue()).decode(),
                    "application/vnd.plotly.v1+json": json.loads(chart.to_json()),
                },
                raw=True,
            )  # type: ignore[no-untyped-call]
        finally:
            plt.close(static)
