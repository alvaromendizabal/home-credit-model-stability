"""Offline, verified publication of the completed development feature study."""

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

from home_credit.modeling.acceptance import compare_number, require
from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.portfolio_report import write_notebook
from home_credit.modeling.release import read_object, verify_file
from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook

LABELS = {
    "control": "Original 700",
    "wider_original": "Wider original 1,400",
    "engineered": "Original + additions (956)",
    "without_amount_ratios": "Without added ratios (861)",
    "without_peer_statistics": "Without peer statistics (906)",
}
FAMILIES = {
    "amount_ratios": "Amount ratios",
    "category_interactions": "Category interactions",
    "dispersion": "Dispersion",
    "household": "Household comparisons",
    "missingness": "Missingness",
    "peer_statistics": "Peer statistics",
    "recency": "Recency shares",
    "source_order": "Source-order differences",
}
WIDTHS = dict(zip(LABELS, [700, 1400, 956, 861, 906], strict=True))


def _validate_screen(result: dict[str, Any], screen: dict[str, Any]) -> None:
    require(
        screen["screen_weeks"] == {"train": [0, 24], "validation": [25, 32]},
        "screen overlaps later development folds",
    )
    require(
        screen["train_rows"] == 160000 and screen["validation_rows"] == 80000,
        "early screen population changed",
    )
    catalog = screen["catalog"]
    require(
        len(catalog) == len({r["name"] for r in catalog}) == 4617,
        "candidate catalog is incomplete or duplicated",
    )
    retained = [r for r in catalog if r["rejection_reason"] is None]
    reasons = Counter(r["rejection_reason"] for r in catalog if r["rejection_reason"])
    require(
        len(retained) == screen["retained"] == result["additional_retained"] == 256,
        "retained feature budget changed",
    )
    require(
        sum(reasons.values()) == screen["rejected"] == result["additional_rejected"] == 4361,
        "feature rejection accounting changed",
    )
    require(
        dict(reasons) == screen["rejection_counts"] == result["rejection_counts"],
        "feature rejection reasons do not reconcile",
    )
    require(
        screen["generated"] == result["additional_candidates"] == len(catalog)
        and result["original_candidates"] == 2508,
        "candidate accounting changed",
    )
    require(
        Counter(r["family"] for r in catalog) == result["candidate_families"]
        and Counter(r["family"] for r in retained) == result["retained_families"],
        "feature family accounting changed",
    )
    selected = screen["selected"]
    require(
        len(selected) == len({r["name"] for r in selected}) == len(retained)
        and {r["name"] for r in selected} == {r["name"] for r in retained},
        "selected list differs from the catalog",
    )
    by_name = {r["name"]: r for r in catalog}
    for spec in selected:
        require(
            all(
                by_name[spec["name"]][k] == spec[k]
                for k in ("family", "sources", "operation", "rationale")
            ),
            "selected feature provenance changed",
        )


def _validate_comparison(result: dict[str, Any], study: dict[str, Any]) -> None:
    require(
        study["complete"] is True and study["identity"] == result["identity"],
        "training study is incomplete or has a different identity",
    )
    require(
        result["holdout_previously_opened"] is True
        and result["outer_holdout_touched"] is False
        and result["release_changed"] is False,
        "post-release research cannot reuse the holdout or change the release",
    )
    require(result["new_model_fits"] == 20, "feature comparison fit budget changed")
    require(
        len(result["rows"]) == 5 and {r["experiment"] for r in result["rows"]} == set(LABELS),
        "missing or duplicate comparison condition",
    )
    folds = {(r["experiment"], r["fold"]): r for r in result["folds"]}
    require(
        len(result["folds"]) == 25 and set(folds) == {(e, f) for e in LABELS for f in range(1, 6)},
        "missing or duplicate comparison fold",
    )
    fits = result["fit_records"]
    require(
        fits == study["trials"]
        and len(fits) == 20
        and {(r["experiment"], r["fold"]) for r in fits}
        == {(e, f) for e in LABELS if e != "control" for f in range(1, 6)},
        "native fit ledger does not cover the planned grid",
    )
    reference = next(r for r in result["rows"] if r["experiment"] == "control")
    for row in result["rows"]:
        experiment = row["experiment"]
        scores = [folds[experiment, f]["stability_score"] for f in range(1, 6)]
        compare_number(float(np.mean(scores)), row["mean_fold_stability"], "mean stability")
        compare_number(min(scores), row["worst_fold_stability"], "worst fold")
        compare_number(
            row["mean_fold_stability"] - reference["mean_fold_stability"],
            row["delta_vs_control"],
            "control delta",
        )
        records = [r for r in fits if r["experiment"] == experiment]
        if not records:
            continue
        require(
            sum(r["validation_rows"] for r in records) == 727187, "validation case coverage changed"
        )
        require(
            len({r["artifacts"]["features"]["sha256"] for r in records}) == 1,
            "feature recipe differs between folds",
        )
        for record in records:
            require(record["features"] == WIDTHS[experiment], "model feature width changed")
            for metric, value in record["metrics"].items():
                compare_number(value, folds[experiment, record["fold"]][metric], metric)


def _validate_interpretation(root: Path, data: dict[str, Any], policy: dict[str, Any]) -> None:
    interpretation = data["interpretation.json"]
    identity = interpretation["identity"]
    require(
        identity["source_commit"] == policy["interpretation_source_commit"]
        and identity["training_ledger_sha256"] == policy["files"]["study.json"]["sha256"]
        and identity["policy_sha256"] == sha256_file(root / "configs/feature_interpretation.json")
        and identity["lock_sha256"] == sha256_file(root / "uv.lock"),
        "interpretation lineage changed",
    )
    require(
        interpretation["new_model_fits"] == 0 and interpretation["holdout_used"] is False,
        "interpretation must only replay development models",
    )
    require(
        len(interpretation["folds"]) == 5
        and {r["fold"] for r in interpretation["folds"]} == set(range(1, 6)),
        "incomplete interpretation folds",
    )
    fits = {
        r["fold"]: r
        for r in data["comparison.json"]["fit_records"]
        if r["experiment"] == "engineered"
    }
    for row in interpretation["folds"]:
        fold = row["fold"]
        member = row["diagnostics"]
        require(
            member["path"] == f"diagnostics_fold_{fold}.json"
            and {k: member[k] for k in ("sha256", "bytes")} == policy["files"][member["path"]],
            "corrected diagnostics identity changed",
        )
        require(
            row["verified_inputs"] == fits[fold]["artifacts"],
            "interpretation inputs differ from the training ledger",
        )
        diag = data[member["path"]]
        require(
            diag["fold"] == fold
            and diag["native_model_sha256"] == fits[fold]["artifacts"]["model"]["sha256"]
            and diag["original_diagnostics_sha256"]
            == fits[fold]["artifacts"]["diagnostics"]["sha256"],
            "diagnostic model lineage changed",
        )
        require(
            diag["new_model_fits"] == 0 and diag["peer_references_refitted"] is False,
            "interpretation refitted a model or peer map",
        )
        require(
            diag["replayed_predictions"] == fits[fold]["validation_rows"]
            and 0 <= diag["prediction_maximum_absolute_error"] <= 1e-12,
            "native prediction replay failed",
        )
        require(
            diag["shap_sampling"] == "uniform_without_replacement_from_full_validation_fold"
            and diag["shap_rows"] == 512
            and diag["sample_rows"] == 12000,
            "SHAP or permutation sampling changed",
        )
        counts = diag["shap_week_counts"]
        require(
            len(counts) == 8
            and {r["WEEK_NUM"] for r in counts} == set(range(25 + 8 * fold, 33 + 8 * fold))
            and all(isinstance(r["len"], int) and r["len"] > 0 for r in counts)
            and sum(r["len"] for r in counts) == 512,
            "SHAP week coverage changed",
        )
        require(0 <= diag["maximum_additivity_absolute_error"] <= 1e-6, "SHAP additivity failed")
        importance = diag["importance"]
        require(
            len(importance) == len({r["name"] for r in importance}) == 956
            and all(
                np.isfinite(r[k]) and r[k] >= 0
                for r in importance
                for k in ("mean_abs_shap", "gain")
            ),
            "incomplete or invalid feature importance",
        )
        permutation = diag["permutation"]
        require(
            len(permutation) == 24
            and {(r["family"], r["repeat"]) for r in permutation}
            == {(f, n) for f in FAMILIES for n in range(3)}
            and all(
                np.isfinite(r[k])
                for r in permutation
                for k in ("stability_decrease", "auc_decrease")
            ),
            "incomplete grouped permutation",
        )
        names = {r["name"] for r in importance}
        require(
            all(
                r["a"] in names
                and r["b"] in names
                and r["a"] != r["b"]
                and r["paired_rows"] >= 100
                and 0.98 <= abs(r["training_correlation"]) <= 1.000000000001
                for r in diag["redundancy"]
            ),
            "invalid training redundancy record",
        )


def load_evidence(root: Path) -> dict[str, Any]:
    """Verify identities and reconcile published claims before exposing any result."""
    policy = read_object(root / "configs/feature_research_review.json")
    expected = {
        "screen.json",
        "comparison.json",
        "study.json",
        "interpretation.json",
        "verification.json",
        *(f"diagnostics_fold_{f}.json" for f in range(1, 6)),
    }
    require(
        policy["schema_version"] == 1 and set(policy["files"]) == expected,
        "unsupported or incomplete feature review policy",
    )
    data = {}
    for name, member in policy["files"].items():
        path = root / "reports/feature_research" / name
        verify_file(path, member["sha256"])
        require(path.stat().st_size == member["bytes"], "feature evidence length changed")
        data[name] = read_object(path)
    result = data["comparison.json"]
    require(
        result["schema_version"] == 1
        and result["identity"]["source_commit"] == policy["source_commit"]
        and result["identity"]["plan_sha256"] == sha256_file(root / "configs/feature_research.json")
        and result["identity"]["lock_sha256"] == sha256_file(root / "uv.lock"),
        "feature training lineage changed",
    )
    for name, member in (
        ("screen.json", result["screen"]["screen"]),
        ("comparison.json", data["study.json"]["stages"]["report"]),
    ):
        require(
            {k: member[k] for k in ("sha256", "bytes")} == policy["files"][name],
            "training checkpoint differs from publication",
        )
    _validate_screen(result, data["screen.json"])
    _validate_comparison(result, data["study.json"])
    verification = data["verification.json"]
    require(
        verification["report_sha256"] == policy["files"]["comparison.json"]["sha256"]
        and verification["source_sha256"]
        == sha256_file(root / "scripts/verify_feature_research.py")
        and verification["prediction_files_verified"] == 21
        and verification["screen_checkpoints_verified"] == 2
        and verification["model_fold_comparisons"] == 25
        and verification["evaluation_cases_per_condition"] == 727187
        and verification["predictions_verified"] == 3635935
        and verification["metric_comparisons"] == 235
        and 0 <= verification["maximum_metric_absolute_error"] <= 1e-12
        and verification["new_model_fits"] == 0
        and verification["holdout_used"] is False,
        "independent feature verification is incomplete",
    )
    _validate_interpretation(root, data, policy)
    diagnostics = [data[f"diagnostics_fold_{f}.json"] for f in range(1, 6)]
    return {
        "result": result,
        "screen": data["screen.json"],
        "verification": verification,
        "diagnostics": diagnostics,
        **{
            key: [{"fold": d["fold"], **r} for d in diagnostics for r in d[key]]
            for key in ("importance", "permutation", "redundancy")
        },
    }


def importance_summary(evidence: dict[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(evidence["importance"])
    rows["top20"] = (
        rows.groupby("fold")["mean_abs_shap"].rank(method="first", ascending=False) <= 20
    )
    return (
        rows.groupby(["name", "family"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            minimum=("mean_abs_shap", "min"),
            maximum=("mean_abs_shap", "max"),
            top20_folds=("top20", "sum"),
        )
        .sort_values(["mean_abs_shap", "name"], ascending=[False, True])
    )


def permutation_summary(evidence: dict[str, Any]) -> pd.DataFrame:
    return (
        pd.DataFrame(evidence["permutation"])
        .groupby("family")
        .agg(
            mean=("stability_decrease", "mean"),
            minimum=("stability_decrease", "min"),
            maximum=("stability_decrease", "max"),
        )
        .reindex(list(FAMILIES))
    )


def feature_label(name: str) -> str:
    parts = name.split("__")
    if parts[0] == "research":
        return f"{FAMILIES[parts[1]]} / {parts[-1][:8]}"
    return f"{parts[0]} / {' / '.join(parts[2:])}"


def chart_pairs(evidence: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Build interactive figures and GitHub PNGs from the same numerical arrays."""
    result = evidence["result"]
    pairs: dict[str, tuple[Any, Any]] = {}
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    interactive = go.Figure()
    y = np.arange(len(FAMILIES))
    labels = list(FAMILIES.values())
    for offset, key, label, color in (
        (-0.18, "candidate_families", "Candidates", "#94a3b8"),
        (0.18, "retained_families", "Retained", "#176b87"),
    ):
        values = [result[key][f] for f in FAMILIES]
        bars = ax.barh(y + offset, values, height=0.35, color=color, label=label)
        ax.bar_label(bars, padding=3, fontsize=9)
        interactive.add_bar(
            x=values,
            y=labels,
            orientation="h",
            name=label,
            marker_color=color,
            text=values,
            textposition="outside",
        )
    ax.set(yticks=y, yticklabels=labels, xlabel="Feature representations", xlim=(0, 2500))
    ax.invert_yaxis()
    ax.legend()
    interactive.update_layout(barmode="group", xaxis_title="Feature representations")
    pairs["Broad search, explicit selection budget"] = (fig, interactive)

    rows = {r["experiment"]: r for r in result["rows"]}
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    interactive = go.Figure()
    for metric, label, color in (
        ("mean_fold_stability", "Mean across five folds", "#176b87"),
        ("worst_fold_stability", "Worst fold", "#b45343"),
    ):
        values = [rows[e][metric] for e in LABELS]
        ax.plot(values, list(LABELS.values()), "o", color=color, label=label)
        interactive.add_scatter(
            x=values,
            y=list(LABELS.values()),
            mode="markers",
            name=label,
            marker={"color": color, "size": 11},
        )
    ax.set(xlabel="Official weekly Gini stability (higher is better)")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    interactive.update_layout(xaxis_title="Official weekly Gini stability")
    pairs["Extra features do not guarantee stable discrimination"] = (fig, interactive)

    folds = {(r["experiment"], r["fold"]): r for r in result["folds"]}
    conditions = [e for e in LABELS if e != "control"]
    changes = [
        [
            folds[e, f]["stability_score"] - folds["control", f]["stability_score"]
            for f in range(1, 6)
        ]
        for e in conditions
    ]
    limit = float(np.max(np.abs(changes)))
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    heat = ax.imshow(changes, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(
        xticks=range(5),
        xticklabels=range(1, 6),
        xlabel="Development fold",
        yticks=range(4),
        yticklabels=[LABELS[e] for e in conditions],
    )
    for i, row in enumerate(changes):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:+.4f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.65 * limit else "#172334",
            )
    fig.colorbar(heat, ax=ax, label="Change from original 700")
    interactive = go.Figure(
        go.Heatmap(
            z=changes,
            x=list(range(1, 6)),
            y=[LABELS[e] for e in conditions],
            zmin=-limit,
            zmax=limit,
            colorscale="RdBu",
            texttemplate="%{z:+.4f}",
        )
    )
    interactive.update_layout(xaxis_title="Development fold")
    pairs["Fold changes: blue improves, red worsens"] = (fig, interactive)

    summary = permutation_summary(evidence)
    means, low, high = (summary[c].tolist() for c in ("mean", "minimum", "maximum"))
    errors = [np.subtract(means, low).tolist(), np.subtract(high, means).tolist()]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.errorbar(means, labels, xerr=errors, fmt="o", color="#176b87", capsize=4)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set(xlabel="Stability decrease after within-week family permutation")
    ax.invert_yaxis()
    interactive = go.Figure(
        go.Scatter(
            x=means,
            y=labels,
            mode="markers",
            marker_color="#176b87",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": errors[1],
                "arrayminus": errors[0],
            },
        )
    )
    interactive.update_layout(xaxis_title="Stability decrease after within-week family permutation")
    pairs["Permutation: mean and observed range, not confidence intervals"] = (fig, interactive)

    top = importance_summary(evidence).head(12)
    names = [feature_label(n) for n in top["name"]]
    values = top["mean_abs_shap"].tolist()
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.barh(names, values, color="#176b87")
    ax.set(xlabel="Mean absolute TreeSHAP contribution (raw log-odds)")
    ax.invert_yaxis()
    interactive = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#176b87",
            customdata=top["name"].tolist(),
            hovertemplate="%{customdata}<br>%{x:.5f}<extra></extra>",
        )
    )
    interactive.update_layout(xaxis_title="Mean absolute TreeSHAP contribution (raw log-odds)")
    pairs["Influential predictors in the engineered condition"] = (fig, interactive)
    for title, (static, chart) in pairs.items():
        static.axes[0].set_title(title, loc="left", fontsize=12, pad=12)
        chart.update_layout(
            title=title,
            template="plotly_white",
            height=520,
            margin={"l": 220, "r": 35, "t": 75, "b": 60},
            font={"family": "Arial", "size": 13},
            yaxis_autorange="reversed",
        )
    return pairs


def display_charts(evidence: dict[str, Any]) -> None:
    from IPython.display import display

    for static, interactive in chart_pairs(evidence).values():
        try:
            buffer = BytesIO()
            static.savefig(buffer, format="png", dpi=140)
            display(
                {
                    "image/png": base64.b64encode(buffer.getvalue()).decode(),
                    "application/vnd.plotly.v1+json": json.loads(interactive.to_json()),
                },
                raw=True,
            )  # type: ignore[no-untyped-call]
        finally:
            plt.close(static)


CELLS = [
    (
        "markdown",
        """
    # Feature research: when more predictors hurt temporal stability

    **Question:** does a wider representation improve the original 700-feature
    LightGBM control on later development periods? We tested 4,617 additional
    hypotheses, retained 256 using earlier data, and completed four conditions
    across all five expanding folds. The archived control contributes five reused
    fits. Every condition covers the same 727,187 validation applications.

    **Result:** the full engineered condition improves pooled AUC, average precision,
    Brier and log loss, but reduces mean official stability from 0.585188 to 0.559904.
    Doubling the original feature budget gives a small mean gain and a weaker worst
    fold. These findings support retaining the frozen release; no result was promoted.

    This is post-release exploratory development research. The original final holdout
    was already opened and is not used here. The original tuning and blend decisions
    also used these development periods; this is not a fresh independent evaluation.
    Read [the frozen release](09_model_release.ipynb) for its separate final result.
    """,
    ),
    (
        "code",
        """
from pathlib import Path

import pandas as pd
from IPython.display import display

from home_credit.modeling.feature_research_report import FAMILIES, LABELS, WIDTHS, load_evidence

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
evidence = load_evidence(root)
result, screen = evidence["result"], evidence["screen"]
print("Verified complete study:", result["study_key"])
print("20 new comparison fits + 5 reused controls; 2 separate early-screen model fits.")
print("This review performs zero fits and requires no private data or cloud credentials.")
display(pd.DataFrame([{"Original candidates": result["original_candidates"],
    "Additional hypotheses": screen["generated"], "Additional retained": screen["retained"],
    "Additional rejected": screen["rejected"], "Cases per condition": 727187}]))
display(pd.Series(screen["rejection_counts"], name="Rejected candidates").to_frame())
""",
    ),
    (
        "markdown",
        """
    ## Formulas, selection and availability

    The original 2,508 features summarize 17 relational groups. The extension adds
    ratios, dispersion, recency shares, household comparisons, category interactions,
    missingness, source-order differences and training-population peer statistics.
    These total 7,125 representations, not independent signals. Every new candidate
    has source columns, an operation, a rationale and a recorded rejection decision.

    Structural screening uses only training rows. Target and drift screeners fit
    weeks 0-24 (160,000 sampled cases) and evaluate weeks 25-32 (80,000 cases), before
    model-validation weeks 33-72. After constants, duplicates and near-constants,
    2,159 additions remain; missingness/cardinality filters leave 2,036 eligible.
    A fixed ranking budget retains 256. Early-window drift AUC is 0.998473, making
    subsequent temporal testing essential. Screen metrics are not final CV results.

    Positive ratios return null for missing or nonpositive denominators; relative
    standard deviation is `std / abs(mean)` when `abs(mean) > 1e-8`, otherwise null.
    Category combinations receive
    training-fold frequency encoding. Each model fits its own peer references:
    44 selected empirical ranks, three conditional median ratios and three conditional
    interquartile positions. Ranks mostly re-express existing information. Unsupported
    peer groups (fewer than 50 training cases) and unseen groups return null.

    Source group order is **not verified chronology**. Historical quantiles/skew were
    not recomputed from raw histories; peer quantiles describe training populations.
    All features rely on the competition's application-time snapshots, without a
    production guarantee about event timestamps or label maturity. CatBoost's native
    target statistics were tested in the original benchmark; online default-rate
    histories are not introduced without outcome-availability timestamps.
    """,
    ),
    (
        "code",
        """
examples = [next(r for r in screen["selected"] if r["family"] == f) for f in FAMILIES]
with pd.option_context("display.max_colwidth", 110):
    display(pd.DataFrame(examples)[["family", "operation", "sources", "rationale"]])
display(pd.DataFrame({"Candidates": result["candidate_families"],
                      "Retained": result["retained_families"]}).rename(index=FAMILIES))
""",
    ),
    (
        "markdown",
        """
    ## Controlled comparisons and model complexity

    The four new conditions use the original LightGBM parameters, seeds, six-thread
    configuration and five full folds. Only the representation changes. All 20 fits
    retain native models, encoders, peer references, feature recipes and predictions.
    The 700-feature control reuses its archived predictions. No learner was retuned
    to rescue a weak feature condition.

    Report mean and worst-fold stability alongside pooled ROC AUC, average precision,
    raw Brier and log loss. Native-model size and selected boosting rounds expose
    complexity; they are not inference-latency measurements. Different worker instance
    families prevent treating job duration as a controlled speed comparison.

    The official metric penalizes declining weekly Gini and residual variation.
    Pooled discrimination can therefore improve while the primary temporal objective
    worsens. Fold changes make that tradeoff visible. Permutation error bars show the
    observed minimum/maximum across five folds and three repeats, **not confidence
    intervals**. Permuting a whole added family within each week preserves its internal
    relationships but breaks dependencies with unpermuted features. On 12,000 sampled
    cases per fold, the slope penalty can produce wide ranges. Full-fold ablations
    carry more weight than one noisy importance number.
    """,
    ),
    (
        "code",
        """
from home_credit.modeling.feature_research_report import display_charts

rows = pd.DataFrame(result["rows"])
rows.insert(1, "features", rows["experiment"].map(WIDTHS))
display(rows.round(6))
fits = pd.DataFrame([{"condition": LABELS[r["experiment"]], "rounds": r["best_iteration"],
    "native_model_MB": r["artifacts"]["model"]["bytes"] / 1_000_000}
    for r in result["fit_records"]])
display(fits.groupby("condition").agg(["mean", "min", "max"]).round(3))
display_charts(evidence)
""",
    ),
    (
        "markdown",
        """
    ## Interpretation, redundancy and an audited correction

    The first SHAP implementation selected a sorted sample prefix, which could
    overrepresent earlier source-ordered cases. This publication uses the corrected
    independent uniform sample of 512 cases from each complete validation fold.
    All eight weeks are represented. A separate job replayed all 727,187 validation
    predictions using saved native models, encoders and peer maps, with zero new fits.
    It checked SHAP additivity and retained links to the original diagnostic hashes.

    TreeSHAP values are raw log-odds contributions, averaged equally across the five
    fold samples. The table shows variation and membership in each fold's top twenty.
    Correlation pairs use up to 4,000 systematically spaced training cases, pairwise
    complete values, and absolute correlation at least 0.98 among the 80 leading
    gain-ranked encoded predictors. This is a targeted redundancy check, not a
    complete dependence analysis.

    Sex is the leading mean-absolute-SHAP predictor in this engineered condition;
    birth-related features also rank highly. That is a disclosed modeling limitation,
    not a fairness finding or justification for a credit decision. Interpretations
    are predictive associations, not causal effects. The model card states that
    this pipeline has not been validated for real lending decisions.
    """,
    ),
    (
        "code",
        """
from home_credit.modeling.feature_research_report import importance_summary

importance = importance_summary(evidence)
with pd.option_context("display.max_colwidth", 100):
    display(importance.head(20).round(6))
    added = importance[importance["name"].str.startswith("research__")].head(5)
    rationale = pd.DataFrame(screen["selected"])
    display(added.merge(rationale, on=["name", "family"])[
        ["name", "mean_abs_shap", "operation", "sources", "rationale"]].round(6))
    redundant = pd.DataFrame(evidence["redundancy"])
    redundant = redundant.sort_values(["training_correlation", "fold"], ascending=[False, True])
    display(redundant.head(12))
checks = [ {k: d[k] for k in ["fold", "replayed_predictions", "prediction_maximum_absolute_error",
    "shap_rows", "maximum_additivity_absolute_error", "new_model_fits", "peer_references_refitted"]}
    for d in evidence["diagnostics"]]
display(pd.DataFrame(checks))
weeks = pd.DataFrame([{"fold": d["fold"], **w} for d in evidence["diagnostics"]
                     for w in d["shap_week_counts"]])
display(weeks.pivot(index="fold", columns="WEEK_NUM", values="len").fillna(0).astype(int))
print("Independent metric recomputation:", evidence["verification"]["metric_comparisons"], "checks")
print("Maximum metric absolute error:", evidence["verification"]["maximum_metric_absolute_error"])
""",
    ),
    (
        "markdown",
        """
    ## Decision and reproducible closure

    - **All 256 additions:** mean stability falls by 0.025284 despite better pooled
      discrimination and probability scores. Feature influence alone does not justify
      inclusion in a model optimized for temporal stability.
    - **Remove the 95 added ratios:** mean stability recovers by 0.024731 relative to
      the full extension, finishing 0.000553 below the control with a stronger worst
      fold. This is not a claim of statistical equivalence.
    - **Remove 50 peer features:** mean stability falls another 0.004293 relative to
      the full extension, while its worst fold improves. Their utility is contextual.
    - **Double the original budget:** mean rises by only 0.001290 and worst-fold
      stability falls by 0.031427, with twice the inputs. No significance claim follows.

    **Decision: preserve the frozen tuned blend.** This extension has answered its
    bounded research questions; it does not retroactively optimize the observed
    holdout. Raw-history quantiles, neural challengers and fully nested promotion
    studies remain possible future work, not unfinished requirements for this release.

    Independent recomputation verified 21 prediction files, 3,635,935 predictions
    across the five conditions and 235 metric identities. The original feature job
    stopped at its time limit after 11 durable fits. Three workers supplied eight
    disjoint late-fold fits; the collector resumed the one unfinished early-fold fit.
    No completed model fit was repeated. The complete driver then restored all 20
    checkpoints with **zero new fits**, and interpretation also demonstrated unchanged
    zero-fit reuse. Recovery is at complete-fold boundaries, not mid-tree checkpoints.

    This notebook validates committed aggregate evidence and can run without private
    data. The catalog, comparison, ledgers, corrected diagnostics, verification script
    and exact hashes are linked in [the research record](../reports/feature_research/README.md).
    [Notebook 12](12_calibration.ipynb) separately tests temporal probability calibration.
    """,
    ),
]


def review(root: Path, *, force: bool = False, write_only: bool = False) -> bool:
    evidence = load_evidence(root)
    logger = ReleaseLogger("feature-research-review", root / "logs")
    work = root / "artifacts/feature_research_review"
    work.mkdir(parents=True, exist_ok=True)
    notebook = work / "11_feature_research.ipynb"
    write_notebook(notebook, CELLS)
    if write_only:
        atomic_write(root / "notebooks" / notebook.name, notebook.read_bytes())
        return False
    policy = read_object(root / "configs/feature_research_review.json")
    dependencies = [
        Path(__file__),
        *(
            root / p
            for p in (
                "configs/feature_research_review.json",
                "configs/feature_research.json",
                "configs/feature_interpretation.json",
                "uv.lock",
                "scripts/verify_feature_research.py",
                "src/home_credit/modeling/portfolio_report.py",
                "src/home_credit/runtime/notebooks.py",
            )
        ),
        *(root / "reports/feature_research" / name for name in sorted(policy["files"])),
    ]
    with StageTimer(logger, "execute_feature_research_review", heartbeat_seconds=15):
        reused = execute_notebook(
            root,
            notebook,
            logger,
            force=force,
            dependencies=dependencies,
            receipt_path=work / "receipt.json",
            execution_root=root,
        )
    atomic_write(root / "notebooks" / notebook.name, notebook.read_bytes())
    atomic_write(
        root / "reports/feature_research/publication.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "study_key": evidence["result"]["study_key"],
                "notebook_sha256": sha256_file(notebook),
                "renderer_sha256": sha256_file(Path(__file__)),
                "execution_receipt": read_object(work / "receipt.json"),
            }
        ),
    )
    logger.event("feature_research_review_completed", reused=reused, new_model_fits=0)
    return reused
