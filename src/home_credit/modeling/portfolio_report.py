"""Canonical feature/release notebooks with interactive and static representations."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import plotly.graph_objects as go

from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.portfolio import load_portfolio
from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook

LABELS = {
    "applprev": "Previous applications",
    "credit_bureau_a": "Credit bureau A",
    "credit_bureau_b": "Credit bureau B",
    "person": "Applicant and related persons",
    "static": "Application snapshot",
    "static_cb": "Bureau snapshot",
    "tax_registry_a": "Tax registry A",
    "tax_registry_b": "Tax registry B",
    "tax_registry_c": "Tax registry C",
    "other": "Other relationships",
    "debitcard": "Debit cards",
    "deposit": "Deposits",
}


def chart_pairs(evidence: dict[str, Any], section: str) -> dict[str, tuple[Any, Any]]:
    """Plotly and PNG fallbacks use the same source values and axis labels."""
    pairs = {}
    if section == "features":
        families = evidence["features"]["families"]
        labels = [LABELS[r["family"]] for r in families]
        generated = [r["candidates"] for r in families]
        retained = [r.get("retained", 0) for r in families]
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        y = np_arange(len(labels))
        ax.barh(y - 0.18, generated, 0.36, label="Candidates", color="#c4d5e4")
        ax.barh(y + 0.18, retained, 0.36, label="Retained", color="#176b87")
        ax.set_yticks(y, labels)
        ax.set(xlabel="Feature count", title="Broad generation, selective retention")
        ax.legend()
        interactive = go.Figure()
        interactive.add_bar(y=labels, x=generated, orientation="h", name="Candidates")
        interactive.add_bar(y=labels, x=retained, orientation="h", name="Retained")
        interactive.update_layout(barmode="group", xaxis_title="Feature count")
        pairs["Feature families"] = (fig, interactive)
        rows = [r for r in evidence["ablation"]["rows"] if r["experiment"] != "control"]
        names = {
            "without_depth2": "Remove depth-two history",
            "without_previous_applications": "Remove previous applications",
            "without_credit_bureau_a": "Remove credit bureau A",
        }
        labels = [names[r["experiment"]] for r in rows]
        deltas = [r["delta_vs_control"] for r in rows]
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.barh(labels, deltas, color="#b45343")
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set(xlabel="Change in mean development stability", title="Every tested removal hurt")
        interactive = go.Figure(go.Bar(y=labels, x=deltas, orientation="h"))
        interactive.update_layout(xaxis_title="Change in mean development stability")
        pairs["Controlled ablations"] = (fig, interactive)
    elif section == "release":
        evaluation = evidence["evaluation"]
        weekly = evaluation["weekly"]
        weeks = [r["week"] for r in weekly]
        gini = [r["gini"] for r in weekly]
        fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
        ax.plot(weeks, gini, marker="o", color="#176b87")
        ax.set(
            xlabel="Reserved week", ylabel="Gini", title="Frozen model: future-week discrimination"
        )
        ax.set_xticks(weeks[::2])
        interactive = go.Figure(
            go.Scatter(
                x=weeks,
                y=gini,
                mode="lines+markers",
                customdata=[[r["rows"], r["positives"]] for r in weekly],
                hovertemplate=(
                    "Week %{x}<br>Gini %{y:.4f}<br>Cases %{customdata[0]:,}"
                    "<br>Defaults %{customdata[1]:,}<extra></extra>"
                ),
            )
        )
        interactive.update_layout(xaxis_title="Reserved week", yaxis_title="Gini")
        pairs["Holdout discrimination"] = (fig, interactive)
        bins = evaluation["reliability"]
        predicted = [r["mean_prediction"] for r in bins]
        observed = [r["observed_default_rate"] for r in bins]
        maximum = max([*predicted, *observed]) * 1.1
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.plot([0, maximum], [0, maximum], "--", color="#94a3b8", label="Perfect reliability")
        ax.plot(predicted, observed, "o-", color="#176b87", label="Frozen uncalibrated model")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate")
        ax.legend()
        interactive = go.Figure()
        interactive.add_scatter(
            x=[0, maximum], y=[0, maximum], mode="lines", name="Perfect reliability"
        )
        interactive.add_scatter(x=predicted, y=observed, mode="lines+markers", name="Frozen model")
        interactive.update_layout(
            xaxis_title="Mean predicted probability", yaxis_title="Observed default rate"
        )
        pairs["Holdout reliability"] = (fig, interactive)
    else:
        raise ValueError("unknown portfolio chart section")
    for title, (_, interactive) in pairs.items():
        interactive.update_layout(
            title=title,
            template="plotly_white",
            font={"family": "Arial, sans-serif", "size": 14},
            height=500,
            margin={"l": 230 if section == "features" else 70, "r": 35, "t": 70, "b": 65},
        )
    return pairs


def np_arange(length: int) -> Any:
    """Float positions keep grouped bars aligned without converting labels to data."""
    import numpy as np

    return np.arange(length, dtype=float)


def display_charts(evidence: dict[str, Any], section: str) -> None:
    """Each notebook output contains Plotly plus a GitHub-compatible PNG fallback."""
    from IPython.display import display

    for static, interactive in chart_pairs(evidence, section).values():
        try:
            buffer = BytesIO()
            static.savefig(buffer, format="png", dpi=140)
            display(  # type: ignore[no-untyped-call]
                {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "application/vnd.plotly.v1+json": json.loads(interactive.to_json()),
                },
                raw=True,
            )
        finally:
            plt.close(static)


LOAD = """
from pathlib import Path

import pandas as pd
from IPython.display import display

from home_credit.modeling.portfolio import load_portfolio

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
evidence = load_portfolio(root)
print("Verified immutable experiment evidence; no model fitting.")
"""

FEATURE_CELLS = [
    (
        "markdown",
        """
    # Feature engineering: from 2,508 candidates to 700 predictors

    **Research question:** which information about an applicant's existing credit,
    payment and application history survives a future-period validation test?

    The frozen feature engine builds 34 train/test blocks from 17 relational table
    groups. Numeric distributions, application-relative dates, five recency windows,
    categorical diversity, missingness and applicant/related-person subgroups produce
    a broad candidate space. These are case-local features; frequency encoders learn
    their mappings only on each model's training population.

    This notebook audits the original experiment. It does not rerun screening or use
    the now-observed final holdout to change the released model.
    """,
    ),
    (
        "code",
        LOAD
        + """
features = evidence["features"]
display(pd.DataFrame([{
    "Candidates": features["candidate_count"],
    "Structurally eligible": features["eligible_count"],
    "Retained": features["retained_count"],
    "Rejected": features["rejected_count"],
}]))
display(pd.Series(features["decisions"], name="Count").to_frame())
""",
    ),
    (
        "markdown",
        """
    ## Where the information comes from

    Credit-bureau records describe overdue payments, balances and historical loan
    behavior. Previous applications describe earlier affordability and credit demand.
    Person tables distinguish the applicant from related people; tax, deposit and
    debit-card tables contribute additional historical context. The application and
    bureau snapshots supply current-at-application summaries.

    Dates become offsets from `date_decision`; counts use the previous 30, 180, 365,
    730 and 1,825 days. Group-index first/last features describe source ordering,
    **not a verified chronological trend**. Availability relies on the competition's
    application-time snapshots, not a production event-time lineage guarantee.
    """,
    ),
    (
        "code",
        """
from home_credit.modeling.portfolio_report import display_charts

display(pd.DataFrame(features["families"]).fillna(0).set_index("family"))
display_charts(evidence, "features")
""",
    ),
    (
        "markdown",
        """
    ## Training-only selection and its limits

    Screening fits on weeks 0-24, uses weeks 25-32 for early stopping and temporal
    drift diagnostics, and finishes before the five model-validation folds begin
    at week 33. It rejects extreme missingness, constants and high-cardinality
    categoricals, then ranks training-window predictive gain with a drift penalty.

    The 1,334 eligible features below the 700-feature limit are **not proven useless**.
    The limit is a computation budget; no 700-versus-larger-set ablation was run.
    The near-perfect early-versus-later drift classifier makes temporal testing essential.
    Feature gain is associative and is not a causal explanation or a substitute for ablation.
    """,
    ),
    (
        "code",
        """
catalog = pd.DataFrame(features["catalog"])
columns = ["name", "family", "target_gain", "drift_gain", "selection_score", "decision"]
ranked = catalog[catalog["selected"]].sort_values("selection_score", ascending=False)
display(ranked[columns].head(15))
print("Temporal drift-classifier AUC:", round(features["drift_validation_auc"], 6))
""",
    ),
    (
        "markdown",
        """
    ## What the ablations establish

    Under the same LightGBM control and five expanding folds, removing credit bureau A,
    previous applications or depth-two history reduced mean official stability.
    This supports retaining those families in this representation. It does not prove
    that every individual retained column helps, or that all possible feature families
    have been exhausted. The detailed executed ablation is notebook 06.
    """,
    ),
    (
        "code",
        """
display(pd.DataFrame(evidence["ablation"]["rows"])[[
    "experiment", "mean_fold_stability", "delta_vs_control", "worst_fold_stability", "oof_auc"
]].round(6))
""",
    ),
    (
        "markdown",
        """
    ## Explicit research boundary

    The frozen release does not include custom affordability ratios, cross-table
    interactions, quantile/skew features, peer ranks, target encoding, or exhaustive
    redundancy and feature-limit ablations. There is no completed SHAP or permutation
    stability study. These remain extensions, not completed experiments or claims of
    exhaustive discovery. External data and target encoding are unnecessary for the
    released pipeline and would require their own availability and leakage controls.

    The full aggregate feature catalog is `reports/feature_ablation/feature_screen.json`;
    each entry records provenance, dtype, missingness, cardinality and both model gains.
    Rejection reasons here are reconstructed from that immutable record and its original
    rules. The feature/model lineage is checked before any results are displayed.
    """,
    ),
]

RELEASE_CELLS = [
    (
        "markdown",
        """
    # Frozen release: evaluation on future weeks

    **Result:** official stability **0.729674**, ROC AUC **0.875759**, Brier **0.019282**
    on **203,345 applications from weeks 73-91**.

    Before reading these labels, the release froze the 700-feature plan, 90% tuned /
    10% original LightGBM weights, no calibration, and development-derived training
    budgets (1,852 and 1,355 rounds). The evaluated models were trained on weeks 0-72.
    The separate all-label refit is an inference artifact; this score is never attributed
    to a model trained on the evaluation labels.
    """,
    ),
    (
        "code",
        LOAD
        + """
evaluation = evidence["evaluation"]
display(pd.Series(evaluation["metrics"], name="Reserved-period result").to_frame().round(6))
print("Evaluation cases:", f"{evaluation['rows']:,}")
print("Observed default rate:", f"{evaluation['positive_rate']:.3%}")
print("Frozen evaluation completed (UTC):", evaluation["evaluated_utc"])
""",
    ),
    (
        "markdown",
        """
    ## Stability and probability quality

    The official formula is mean weekly Gini + 88 x min(weekly slope, 0) - 0.5 x
    residual standard deviation. Here the slope is positive, so its penalty is zero.
    The weekly plot shows the variation that the overall AUC alone would hide.

    Raw probability calibration remains imperfect. The reliability plot is descriptive;
    no calibrator was fitted after seeing this holdout. Its base rate differs from the
    development periods, so the lower Brier score alone is not a model-improvement claim.
    A Kaggle leaderboard result has not been obtained.
    """,
    ),
    (
        "code",
        """
from home_credit.modeling.portfolio_report import display_charts

display_charts(evidence, "release")
display(pd.DataFrame(evaluation["weekly"]).round(6))
""",
    ),
    (
        "markdown",
        """
    ## Distinct evaluation and inference models

    Four native fits completed: two development-only models for the one frozen evaluation,
    then two all-label models for inference. Each fit has a verified encoder, exact feature
    order, fixed iterations, a reload parity check and a content-addressed S3 checkpoint.
    The release was rerun successfully with all completed stages reused.
    """,
    ),
    (
        "code",
        """
stages = evidence["state"]["stages"]
records = []
for name, stage in stages.items():
    if "/" not in name:
        continue
    records.append({
        "Artifact": name,
        "Fit rows": stage["fit_rows"],
        "Fit weeks": str(stage["fit_weeks"]),
        "Iterations": stage["actual_rounds"],
        "Reload error": stage["reload_max_absolute_error"],
    })
display(pd.DataFrame(records))
print("Portable bundle files:", len(stages["bundle"]["files"]))
print("Public example parity cases:", stages["raw_parity"]["rows"])
""",
    ),
    (
        "markdown",
        """
    ## What is deployable and what has been tested

    The portable bundle contains native LightGBM models, frozen frequency maps,
    feature specifications, raw schema and matching inference source. It rebuilds
    features from the test files supplied to the run, fingerprints inputs, validates
    case coverage and reuses verified prediction batches.

    Full raw-to-feature parity was checked on the competition's ten public example
    cases. That is an integration fixture, not a hidden-test evaluation or proof of
    performance at hidden-test scale. Synthetic tests cover additional missingness,
    shard and corruption cases. Notebook 10 gives the owner explicit controls to
    generate, validate, save and download a submission; no automatic upload occurs.

    This is an evaluated research portfolio release. Production lending use, calibrated
    deployment probabilities, fairness assessment and operational monitoring are outside
    the validated scope. The feature-research limits are documented in notebook 02.
    """,
    ),
]


def write_notebook(path: Path, cells: list[tuple[str, str]]) -> None:
    """Stable cell IDs and unchanged sources retain valid executed notebook state."""
    sources = []
    for number, (kind, text) in enumerate(cells):
        source = dedent(text).strip()
        if kind == "code":
            source = subprocess.run(
                [sys.executable, "-m", "ruff", "format", "--stdin-filename", "cell.py", "-"],
                input=source,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        cell = (
            nbformat.v4.new_code_cell(source)  # type: ignore[no-untyped-call]
            if kind == "code"
            else nbformat.v4.new_markdown_cell(source)  # type: ignore[no-untyped-call]
        )
        cell.id = f"{path.stem}-{number:02d}"
        sources.append(cell)
    if path.exists():
        previous = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]
        if [(c.cell_type, c.source, c.id) for c in previous.cells] == [
            (c.cell_type, c.source, c.id) for c in sources
        ]:
            return
    notebook = nbformat.v4.new_notebook(cells=sources)  # type: ignore[no-untyped-call]
    notebook.metadata.kernelspec = {
        "display_name": "Python (Home Credit)",
        "language": "python",
        "name": "home-credit",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12.14"}
    atomic_write(path, nbformat.writes(notebook).encode())  # type: ignore[no-untyped-call]


def review_portfolio(
    root: Path, *, force: bool = False, write_only: bool = False
) -> dict[str, bool]:
    """Execute inexpensive reviews in a staging directory before publishing canonical copies."""
    evidence = load_portfolio(root)
    policy = evidence["policy"]
    logger = ReleaseLogger("portfolio-review", root / "logs")
    dependencies = [
        root / "uv.lock",
        root / "configs/portfolio_review.json",
        Path(__file__),
        root / "src/home_credit/modeling/portfolio.py",
        root / "src/home_credit/runtime/notebooks.py",
        *[root / entry["path"] for entry in policy["artifacts"]],
    ]
    work = root / "artifacts/portfolio_review"
    work.mkdir(parents=True, exist_ok=True)
    records = {}
    receipts = {}
    for filename, cells in (
        ("02_feature_engineering.ipynb", FEATURE_CELLS),
        ("09_model_release.ipynb", RELEASE_CELLS),
    ):
        path = work / filename
        write_notebook(path, cells)
        if write_only:
            atomic_write(root / "notebooks" / filename, path.read_bytes())
            continue
        receipt = work / (path.stem + ".json")
        with StageTimer(logger, path.stem, heartbeat_seconds=15):
            records[filename] = execute_notebook(
                root,
                path,
                logger,
                force=force,
                dependencies=dependencies,
                receipt_path=receipt,
                execution_root=root,
            )
        atomic_write(root / "notebooks" / filename, path.read_bytes())
        receipts[filename] = {
            "sha256": sha256_file(path),
            "execution": json.loads(receipt.read_text()),
        }
    if not write_only:
        atomic_write(
            root / "reports/model_release/publication.json", canonical_json_bytes(receipts)
        )
    return records
