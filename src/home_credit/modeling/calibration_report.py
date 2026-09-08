"""Executed, offline probability-calibration research with explicit temporal scope."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from home_credit.modeling.acceptance import compare_number, require
from home_credit.modeling.calibration_research import apply_calibrator
from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.portfolio_report import write_notebook
from home_credit.modeling.release import read_object, verify_file
from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook

LABELS = {
    "uncalibrated": "Frozen blend",
    "sigmoid": "Past-fold sigmoid",
    "isotonic": "Past-fold isotonic",
}


def load_evidence(root: Path) -> dict[str, Any]:
    policy = read_object(root / "configs/calibration_review.json")
    path = root / "reports/calibration/comparison.json"
    verify_file(path, policy["sha256"])
    require(path.stat().st_size == policy["bytes"], "calibration report length changed")
    result = read_object(path)
    require(result["schema_version"] == 1, "unsupported calibration evidence")
    require(
        result["identity"]["source_commit"] == policy["source_commit"], "calibration source changed"
    )
    require(
        result["identity"]["plan_sha256"]
        == sha256_file(root / "configs/calibration_research.json"),
        "calibration plan changed",
    )
    require(
        result["holdout_used"] is False and result["release_changed"] is False,
        "calibration cannot use the observed holdout or change the release",
    )
    require(
        result["new_base_model_fits"] == 0 and result["new_calibrator_fits"] == 8,
        "calibration fit budget changed",
    )
    require(result["evaluation_weeks"] == [41, 72], "calibration evaluation weeks changed")
    require(len(result["folds"]) == 12 and len(result["rows"]) == 3, "incomplete calibration study")
    require({row["method"] for row in result["rows"]} == set(LABELS), "method coverage changed")
    require(
        {(r["method"], r["fold"]) for r in result["folds"]}
        == {(m, f) for m in LABELS for f in (2, 3, 4, 5)},
        "missing or duplicate method-fold",
    )
    for record in result["folds"]:
        require(
            record["fit_week_max"]
            < record["evaluation_week_min"]
            <= record["evaluation_week_max"]
            <= 72,
            "calibrator fitting overlaps its evaluation",
        )
        apply_calibrator(np.array([0.0, 0.05, 0.5, 1.0]), record["model"])
    for row in result["rows"]:
        records = [r for r in result["folds"] if r["method"] == row["method"]]
        count = sum(r["evaluation_rows"] for r in records)
        require(count == row["evaluation_rows"], "calibration row coverage changed")
        compare_number(
            float(np.mean([r["metrics"]["stability_score"] for r in records])),
            row["mean_fold_stability"],
            "calibration mean stability",
        )
        compare_number(
            min(r["metrics"]["stability_score"] for r in records),
            row["worst_fold_stability"],
            "calibration worst fold",
        )
        for metric in ("brier_score", "log_loss"):
            compare_number(
                sum(r["metrics"][metric] * r["evaluation_rows"] for r in records) / count,
                row[f"pooled_{metric}"],
                metric,
            )
        require(
            sum(b["rows"] for b in result["reliability"][row["method"]]) == count,
            "reliability support mismatch",
        )
    verification_path = root / "reports/calibration/verification.json"
    verify_file(verification_path, policy["verification_sha256"])
    verification = read_object(verification_path)
    require(
        verification["report_sha256"] == policy["sha256"]
        and verification["prediction_checkpoints_verified"] == 12
        and verification["input_oof_files_verified"] == 2
        and verification["replayed_predictions"] == 3 * 544611
        and verification["maximum_prediction_absolute_error"] <= 1e-14
        and verification["maximum_metric_absolute_error"] <= 1e-12
        and verification["new_base_model_fits"] == verification["new_calibrator_fits"] == 0
        and verification["holdout_used"] is False,
        "independent calibration verification is incomplete",
    )
    return result


def chart_pairs(result: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    pairs = {}
    colors = {"uncalibrated": "#64748b", "sigmoid": "#176b87", "isotonic": "#b45343"}
    for metric, title in (("brier_score", "Brier score"), ("log_loss", "Log loss")):
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        interactive = go.Figure()
        reference = {
            r["fold"]: r["metrics"][metric]
            for r in result["folds"]
            if r["method"] == "uncalibrated"
        }
        for method in LABELS:
            rows = sorted(
                (r for r in result["folds"] if r["method"] == method), key=lambda r: r["fold"]
            )
            x = [r["fold"] for r in rows]
            y = [r["metrics"][metric] - reference[r["fold"]] for r in rows]
            ax.plot(x, y, "o-", color=colors[method], label=LABELS[method])
            interactive.add_scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=LABELS[method],
                line={"color": colors[method]},
            )
        ax.set(
            xlabel="Later development fold",
            ylabel=f"Change in {title.lower()} vs frozen blend",
            title=f"{title}: below zero improves on the frozen blend",
        )
        ax.set_xticks([2, 3, 4, 5])
        ax.legend()
        interactive.update_layout(
            xaxis_title="Later development fold",
            yaxis_title=f"Change in {title.lower()} vs frozen blend",
        )
        pairs[title] = (fig, interactive)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    interactive = go.Figure()
    maximum = (
        max(
            max(b["mean_prediction"], b["observed_default_rate"])
            for bins in result["reliability"].values()
            for b in bins
        )
        * 1.05
    )
    ax.plot([0, maximum], [0, maximum], "--", color="#94a3b8", label="Perfect reliability")
    interactive.add_scatter(
        x=[0, maximum], y=[0, maximum], mode="lines", name="Perfect reliability"
    )
    for method in LABELS:
        bins = result["reliability"][method]
        x, y = [b["mean_prediction"] for b in bins], [b["observed_default_rate"] for b in bins]
        ax.plot(x, y, "o-", color=colors[method], label=LABELS[method])
        interactive.add_scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=LABELS[method],
            customdata=[[b["rows"]] for b in bins],
            hovertemplate=(
                "Predicted %{x:.3f}<br>Observed %{y:.3f}<br>Cases %{customdata[0]:,}<extra></extra>"
            ),
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title="Reliability on later development periods",
    )
    ax.legend()
    interactive.update_layout(
        xaxis_title="Mean predicted probability", yaxis_title="Observed default rate"
    )
    pairs["Reliability"] = (fig, interactive)
    for name, (_, chart) in pairs.items():
        chart.update_layout(
            title=name, template="plotly_white", height=450, font={"family": "Arial", "size": 14}
        )
    return pairs


def display_charts(result: dict[str, Any]) -> None:
    from IPython.display import display

    for static, interactive in chart_pairs(result).values():
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
    # Probability calibration under temporal shift

    Good ranking does not guarantee useful probability estimates. This study
    compares the frozen 90/10 LightGBM blend with sigmoid and isotonic calibration.
    For each evaluation fold, calibrators see **only earlier development-fold
    predictions and labels**. Four later folds cover weeks 41-72.

    This is post-release exploratory research. Base hyperparameters and blend
    weights were selected using all development folds, so the comparison is **not
    fully nested unbiased validation**. It neither recalibrates the released bundle
    nor uses the observed final holdout. No base model is retrained.
    """,
    ),
    (
        "code",
        """
from pathlib import Path

import pandas as pd
from IPython.display import display

from home_credit.modeling.calibration_report import load_evidence

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
evidence = load_evidence(root)
print("Verified calibration study:", evidence["study_key"])
print("Eight calibrator fits; zero new base-model fits; no fitting in this review.")
rows = pd.DataFrame(evidence["rows"])
columns = ["method", "mean_fold_stability", "worst_fold_stability", "pooled_auc",
           "pooled_pr_auc", "pooled_brier_score", "pooled_log_loss", "evaluation_rows"]
display(rows[columns].round(6))
""",
    ),
    (
        "markdown",
        """
    ## The temporal contract

    Evaluation starts at fold 2 because fold 1 supplies the first out-of-fold
    calibration population. Later fits expand across all earlier folds. Both
    calibration methods use the same past cases. Sigmoid calibration fits a
    positive-slope logistic transformation of clipped log-odds. Isotonic calibration
    fits a nondecreasing mapping and can introduce ties that affect ranking metrics.
    Portable JSON parameters reproduce each calibrator's predictions exactly.
    An independent replay of all 1,633,833 saved predictions also checked input
    identities, exact case coverage and all eight fold and pooled metrics.
    """,
    ),
    (
        "code",
        """
folds = pd.DataFrame([{k: r[k] for k in ["method", "fold", "fit_rows", "fit_week_max",
    "evaluation_week_min", "evaluation_week_max", "evaluation_rows"]} for r in evidence["folds"]])
display(folds)
""",
    ),
    (
        "markdown",
        """
    ## Probability quality and discrimination are different questions

    Brier score and log loss assess probabilities; lower values are better. The
    first two plots show changes from the frozen blend within each fold, so below
    zero means improvement. This exposes differences otherwise hidden by changing
    prevalence across periods. The
    official weekly Gini stability remains visible alongside ROC AUC and average
    precision. A rank-preserving transformation can improve probability metrics
    without improving within-fold discrimination. Pooled AUC can change because
    different fold-specific transforms alter cross-period ordering.

    The reliability plot uses fixed probability bins; hover shows support. High-risk
    bins can contain relatively few applications, and these descriptive curves are
    not confidence intervals or evidence of production calibration.
    """,
    ),
    (
        "code",
        """
from home_credit.modeling.calibration_report import display_charts

display_charts(evidence)
baseline = rows.set_index("method").loc["uncalibrated"]
comparison = rows[["method", "pooled_brier_score", "pooled_log_loss", "mean_fold_stability"]].copy()
for metric in ["pooled_brier_score", "pooled_log_loss", "mean_fold_stability"]:
    comparison[f"change_{metric}"] = comparison[metric] - baseline[metric]
display(comparison.round(6))
""",
    ),
    (
        "markdown",
        """
    ## Observed conclusion and decision boundary

    Neither tested calibrator improved pooled Brier score or log loss on these
    544,611 later-period applications. Sigmoid retained within-fold ranking and
    stability; isotonic introduced ties and reduced mean stability. Earlier-period
    calibration therefore did not reliably transfer in this comparison. This is
    evidence against adopting either tested map for this reference pipeline.

    These results answer whether earlier-period calibration transfers to later
    development periods for this fixed reference blend. They do not retrospectively
    alter the model evaluated in notebook 09. A future deployment needs a separate
    calibration population, outcome-availability controls, population monitoring and
    a new independent evaluation after any promotion decision.

    Every method/fold has verified input hashes, a persisted prediction checkpoint,
    portable parameters and exact parameter replay. The complete study was rerun
    using those checkpoints with zero new calibrator fits. The review uses only
    committed aggregate evidence and requires no private data or cloud credentials.
    """,
    ),
]


def review(root: Path, *, force: bool = False, write_only: bool = False) -> bool:
    evidence = load_evidence(root)
    logger = ReleaseLogger("calibration-review", root / "logs")
    work = root / "artifacts/calibration_review"
    work.mkdir(parents=True, exist_ok=True)
    notebook = work / "12_calibration.ipynb"
    write_notebook(notebook, CELLS)
    if write_only:
        atomic_write(root / "notebooks" / notebook.name, notebook.read_bytes())
        return False
    with StageTimer(logger, "execute_calibration_review", heartbeat_seconds=15):
        reused = execute_notebook(
            root,
            notebook,
            logger,
            force=force,
            dependencies=[
                Path(__file__),
                root / "configs/calibration_review.json",
                root / "configs/calibration_research.json",
                root / "uv.lock",
                root / "reports/calibration/comparison.json",
                root / "reports/calibration/verification.json",
                root / "src/home_credit/modeling/portfolio_report.py",
                root / "src/home_credit/modeling/calibration_research.py",
                root / "src/home_credit/runtime/notebooks.py",
            ],
            receipt_path=work / "receipt.json",
            execution_root=root,
        )
    atomic_write(root / "notebooks" / notebook.name, notebook.read_bytes())
    atomic_write(
        root / "reports/calibration/publication.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "study_key": evidence["study_key"],
                "notebook_sha256": sha256_file(notebook),
                "renderer_sha256": sha256_file(Path(__file__)),
                "execution_receipt": read_object(work / "receipt.json"),
            }
        ),
    )
    logger.event("calibration_review_completed", reused=reused, new_model_fits=0)
    return reused
