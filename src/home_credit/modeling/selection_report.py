"""Reproducible Plotly reviews with static fallbacks for GitHub notebook readers."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import plotly.graph_objects as go

from home_credit.modeling.checkpoints import atomic_write, sha256_file
from home_credit.modeling.selection import fixed_candidates, rank_records, require, validate_record
from home_credit.observability.runtime import StageTimer

if TYPE_CHECKING:
    from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
    from home_credit.observability.logging import RunLogger


METRIC_COLUMNS = (
    "mean_fold_stability",
    "worst_fold_stability",
    "oof_auc",
    "oof_pr_auc",
    "oof_brier_score",
    "oof_log_loss",
)


def validate_selection_evidence(result: dict[str, Any]) -> None:
    """Reject incomplete or internally inconsistent aggregate publication evidence."""
    require(result["schema_version"] == 1, "unsupported selection evidence")
    require(result["outer_holdout_touched"] is False, "holdout result cannot enter this report")
    require(result["new_model_fits"] == 0, "unexpected training in selection evidence")
    require(result["kaggle_submitted"] is False, "unexpected Kaggle action")
    require(result["calibration_fitted"] is False, "unexpected calibration experiment")
    names = ["tuned_lightgbm", "lightgbm", "xgboost", "catboost"]
    candidates = fixed_candidates(names, names[0])
    rows = result["rows"]
    require(result["candidate_count"] == len(rows) == 15, "incomplete candidate report")
    require({row["candidate"] for row in rows} == set(candidates), "candidate coverage changed")
    require(len(result["folds"]) == 75, "incomplete fold evidence")
    records = []
    for row in rows:
        name = row["candidate"]
        record = {
            "name": name,
            "weights": candidates[name],
            "state": "complete",
            "metrics": {
                key: value
                for key, value in row.items()
                if key not in {"candidate", "delta_vs_tuned_lightgbm"}
            },
            "folds": [
                {key: value for key, value in fold.items() if key != "candidate"}
                for fold in result["folds"]
                if fold["candidate"] == name
            ],
        }
        validate_record(record, name, candidates[name], 5)
        records.append(record)
    ranked = rank_records(records, names[0])
    selected = ranked[0]["name"]
    require([r["candidate"] for r in rows] == [r["name"] for r in ranked], "ranking changed")
    require(result["selected_candidate"] == selected, "selected candidate mismatch")
    require(result["selected_weights"] == candidates[selected], "selected weights mismatch")
    incumbent = next(r for r in rows if r["candidate"] == names[0])
    for row in rows:
        expected = row["mean_fold_stability"] - incumbent["mean_fold_stability"]
        require(
            math.isclose(row["delta_vs_tuned_lightgbm"], expected, rel_tol=0, abs_tol=1e-12),
            "candidate delta mismatch",
        )
    weekly = result["diagnostics"]["weekly"]
    reliability = result["diagnostics"]["reliability"]
    require(sum(r["rows"] for r in weekly) == result["rows_evaluated"], "weekly support mismatch")
    require(
        sum(r["rows"] for r in reliability) == result["rows_evaluated"],
        "reliability support mismatch",
    )
    require(len({r["week"] for r in weekly}) == len(weekly), "duplicate diagnostic week")
    require(all(0 <= r["week"] < 73 for r in weekly), "holdout diagnostic week")
    if result["smoke"] is False:
        require(result["rows_evaluated"] == 727187, "full-data population changed")
        require({r["week"] for r in weekly} == set(range(33, 73)), "development weeks changed")


def candidate_label(name: str) -> str:
    """Keep the fifteen plotted labels readable while retaining exact names in tables."""
    labels = {
        "tuned_lightgbm": "Tuned LightGBM",
        "lightgbm": "Original LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "equal_three_families": "Equal three-family blend",
        "equal_all_four": "Equal four-predictor blend",
    }
    if name in labels:
        return labels[name]
    for other in ("lightgbm", "xgboost", "catboost"):
        for weight in (50, 75, 90):
            if name == f"tuned_lightgbm_{weight}_{other}":
                return f"Tuned LGBM {weight}% + {labels[other]}"
    return name


def selection_figures(result: dict[str, Any]) -> dict[str, Any]:
    """Static equivalents of the three interactive figures for GitHub and print."""
    figures = {}
    records = list(reversed(result["rows"]))
    figure, axis = plt.subplots(figsize=(11, 7), layout="constrained")
    axis.barh(
        [candidate_label(row["candidate"]) for row in records],
        [row["mean_fold_stability"] for row in records],
    )
    axis.set(
        xlabel="Mean official fold stability (higher is better)",
        title="Development comparison | 15 fixed candidates",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figures["candidates"] = figure
    weekly = result["diagnostics"]["weekly"]
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    axis.plot([row["week"] for row in weekly], [row["gini"] for row in weekly], "o-")
    axis.set(
        xlabel="Development week",
        ylabel="Gini = 2 x ROC AUC - 1",
        title="Selected candidate | discrimination over time",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figures["weekly_gini"] = figure
    reliability = result["diagnostics"]["reliability"]
    figure, axis = plt.subplots(figsize=(6.5, 5), layout="constrained")
    x = [row["mean_prediction"] for row in reliability]
    y = [row["observed_default_rate"] for row in reliability]
    maximum = max([*x, *y, 0.01]) * 1.08
    axis.plot([0, maximum], [0, maximum], "--", label="Perfect reliability")
    axis.plot(x, y, "o-", label="Quantile bins; ties kept together")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title="Development reliability | no calibration fitted",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figures["reliability"] = figure
    return figures


def selection_plotly_figures(result: dict[str, Any]) -> dict[str, Any]:
    """Interactive charts with useful hover support and no external data requests."""
    records = list(reversed(result["rows"]))
    candidates = go.Figure(
        go.Bar(
            x=[r["mean_fold_stability"] for r in records],
            y=[candidate_label(r["candidate"]) for r in records],
            orientation="h",
            customdata=[
                [r["candidate"], r["worst_fold_stability"], r["delta_vs_tuned_lightgbm"]]
                for r in records
            ],
            hovertemplate=(
                "%{customdata[0]}<br>Mean stability: %{x:.6f}"
                "<br>Worst fold: %{customdata[1]:.6f}"
                "<br>Delta vs tuned: %{customdata[2]:+.6f}<extra></extra>"
            ),
        )
    )
    candidates.update_layout(
        title="Development comparison | 15 fixed candidates",
        xaxis_title="Mean official fold stability (higher is better)",
        height=650,
        margin={"l": 270, "r": 30, "t": 70, "b": 60},
    )
    weekly = result["diagnostics"]["weekly"]
    temporal = go.Figure(
        go.Scatter(
            x=[r["week"] for r in weekly],
            y=[r["gini"] for r in weekly],
            mode="lines+markers",
            customdata=[[r["fold"], r["rows"], r["positive_rate"]] for r in weekly],
            hovertemplate=(
                "Week %{x}<br>Gini: %{y:.4f}<br>Fold: %{customdata[0]}"
                "<br>Cases: %{customdata[1]:,}<br>Default rate: %{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    temporal.update_layout(
        title="Selected candidate | weekly discrimination and population support",
        xaxis_title="Development week",
        yaxis_title="Gini = 2 x ROC AUC - 1",
        height=450,
    )
    bins = result["diagnostics"]["reliability"]
    x, y = [r["mean_prediction"] for r in bins], [r["observed_default_rate"] for r in bins]
    maximum = max([*x, *y, 0.01]) * 1.08
    calibration = go.Figure()
    calibration.add_scatter(
        x=[0, maximum], y=[0, maximum], mode="lines", name="Perfect reliability", line_dash="dash"
    )
    calibration.add_scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name="Quantile bins; ties kept together",
        customdata=[[r["bin"], r["rows"]] for r in bins],
        hovertemplate=(
            "Bin %{customdata[0]}<br>Cases: %{customdata[1]:,}"
            "<br>Predicted: %{x:.3%}<br>Observed: %{y:.3%}<extra></extra>"
        ),
    )
    calibration.update_layout(
        title="Development reliability | no calibrator fitted",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed default rate",
        height=500,
        legend={"orientation": "h", "y": -0.22},
    )
    figures = {"candidates": candidates, "weekly_gini": temporal, "reliability": calibration}
    for figure in figures.values():
        figure.update_layout(font={"family": "Arial, sans-serif", "size": 14})
    return figures


def write_selection_notebook(path: Path) -> None:
    """Generate canonical source, preserving unchanged executed cells and metadata."""
    digest = sha256_file(path.parent / "selection.json")
    cells = [
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "# Home Credit: Development model selection\n\n"
            "**15 predeclared candidates · 0 new model fits · holdout untouched.**\n\n"
            "Can complementary models improve the tuned LightGBM without another fit? "
            "This notebook compares hash-verified, aligned development predictions. "
            "It is not a final-test or leaderboard result. Weeks 73-91 remain locked. "
            "Charts contain interactive Plotly data and static GitHub fallbacks."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            dedent(
                """
                import hashlib
                import json
                from pathlib import Path

                import pandas as pd
                from IPython.display import display

                from home_credit.modeling.selection_report import validate_selection_evidence

                parents = [Path.cwd(), *Path.cwd().parents]
                candidates = [Path.cwd() / "selection.json"]
                candidates += [p / "reports/model_selection/selection.json" for p in parents]
                expected_sha256 = "__EVIDENCE_SHA256__"
                source = None
                for candidate in candidates:
                    if not candidate.is_file():
                        continue
                    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                    if digest == expected_sha256:
                        source = candidate
                        break
                if source is None:
                    raise FileNotFoundError("Matching selection evidence is missing")
                result = json.loads(source.read_text())
                validate_selection_evidence(result)
                print("Scope:", result["scope"])
                print("Cases evaluated:", f"{result['rows_evaluated']:,}")
                columns = [
                    "candidate",
                    "mean_fold_stability",
                    "worst_fold_stability",
                    "oof_auc",
                    "oof_pr_auc",
                    "oof_brier_score",
                    "oof_log_loss",
                ]
                display(pd.DataFrame(result["rows"])[columns].round(6))
                """
            )
            .strip()
            .replace("__EVIDENCE_SHA256__", digest)
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Stability, time and reliability\n\n"
            "Selection maximizes the mean of five official fold stability scores. "
            "ROC AUC and average precision describe ranking; raw Brier and clipped "
            "log loss describe probabilities. Weekly hover information includes fold, "
            "case count and default rate: a noisy, low-support week must not be "
            "mistaken for a reliable trend. The offline HTML preserves interactivity."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            dedent(
                """
                import base64
                from io import BytesIO

                import matplotlib.pyplot as plt

                from home_credit.modeling.selection_report import (
                    selection_figures,
                    selection_plotly_figures,
                )

                interactive = selection_plotly_figures(result)
                for name, figure in selection_figures(result).items():
                    buffer = BytesIO()
                    figure.savefig(buffer, format="png", dpi=125)
                    payload = json.loads(interactive[name].to_json())
                    display(
                        {
                            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                            "application/vnd.plotly.v1+json": payload,
                        },
                        raw=True,
                    )
                    plt.close(figure)
                """
            ).strip()
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Does the weight choice transfer to the next fold?\n\n"
            "This diagnostic chooses weights using earlier folds and observes the "
            "next fold. **It is not fully nested validation:** base-model tuning "
            "already used all development folds. Correlation indicates how similar "
            "the predictors are; only measured blend performance justifies complexity."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            dedent(
                """
                display(pd.DataFrame(result["prequential_weight_choices"]).round(6))
                display(pd.DataFrame(result["diagnostics"]["prediction_correlations"]).round(6))
                selected = result["selected_candidate"]
                winner = next(r for r in result["rows"] if r["candidate"] == selected)
                folds = pd.DataFrame(result["folds"])
                scores = folds.pivot(index="fold", columns="candidate", values="stability_score")
                comparison = scores[[selected, "tuned_lightgbm"]].copy()
                if selected == "tuned_lightgbm":
                    print("No blend improved the declared objective; retain the simpler incumbent.")
                else:
                    delta = comparison.iloc[:, 0] - comparison.iloc[:, 1]
                    display(delta.rename("Selected minus tuned LightGBM").to_frame().round(6))
                    print("Folds improved:", int((delta > 0).sum()), "of", len(delta))
                print("Selected development candidate:", selected)
                print("Mean stability gain:", f"{winner['delta_vs_tuned_lightgbm']:+.6f}")
                print(json.dumps(result["selected_weights"], indent=2, sort_keys=True))
                """
            ).strip()
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Decision boundary and owner-controlled release\n\n"
            "Development improvement is not a claim of state-of-the-art or future "
            "performance. Freeze model, ensemble and calibration choices before "
            "evaluating the locked holdout once. Final refit, train/test feature "
            "parity and competition-compatible inference still need their own tested stage.\n\n"
            "The submission notebook will require an explicit owner action to "
            "generate predictions, validate `case_id,score`, preserve sample order, "
            "save durably and show a download link. **This review creates no "
            "submission CSV and never uploads to Kaggle.**"
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            dedent(
                """
                import os

                from IPython.display import FileLink

                relative_evidence = os.path.relpath(source)
                display(FileLink(relative_evidence, result_html_prefix="Aggregate evidence: "))
                display(
                    FileLink(
                        os.path.relpath(source.parent / "report.html"),
                        result_html_prefix="Interactive offline review: ",
                    )
                )
                """
            ).strip()
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"selection-{index}"
    notebook = nbformat.v4.new_notebook(cells=cells)  # type: ignore[no-untyped-call]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.validate(notebook)
    if path.is_file():
        previous = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]
        if [c.source for c in previous.cells] == [c.source for c in cells]:
            return
    atomic_write(path, nbformat.writes(notebook).encode())  # type: ignore[no-untyped-call]


def write_selection_report(result: dict[str, Any], directory: Path) -> list[Path]:
    """Write deterministic offline Plotly HTML and static, account-free SVGs."""
    validate_selection_evidence(result)
    directory.mkdir(parents=True, exist_ok=True)
    figures = selection_figures(result)
    outputs = []
    try:
        with matplotlib.rc_context({"svg.hashsalt": "home-credit-model-selection"}):
            for name, figure in figures.items():
                path = directory / f"{name}.svg"
                figure.savefig(path, metadata={"Date": None})
                outputs.append(path)
    finally:
        for figure in figures.values():
            plt.close(figure)
    sections = [
        figure.to_html(
            full_html=False,
            include_plotlyjs=index == 0,
            div_id=f"selection-{name}",
            config={"displaylogo": False, "responsive": True},
        )
        for index, (name, figure) in enumerate(selection_plotly_figures(result).items())
    ]
    header = "<th>Candidate</th>" + "".join(
        f"<th>{html.escape(name.replace('_', ' '))}</th>" for name in METRIC_COLUMNS
    )
    table = "".join(
        "<tr><td>"
        + html.escape(row["candidate"])
        + "</td>"
        + "".join(f"<td>{row[name]:.6f}</td>" for name in METRIC_COLUMNS)
        + "</tr>"
        for row in result["rows"]
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Home Credit | Development selection</title><style>"
        "body{font:16px system-ui;max-width:1180px;margin:40px auto;padding:0 24px;"
        "line-height:1.6}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;"
        "font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #ddd}"
        "th:first-child,td:first-child{text-align:left}.table{overflow:auto}"
        "</style><h1>Development model selection</h1><p>"
        "15 fixed candidates; no new training (zero model fits); weeks 73-91 remain locked. "
        "No calibrator is fitted. No submission is generated or uploaded.</p><h2>"
        + html.escape(candidate_label(result["selected_candidate"]))
        + "</h2><p>Development selection, not an untouched test or leaderboard score.</p>"
        + "".join(sections)
        + "<noscript>"
        + "".join(path.read_text() for path in outputs)
        + "</noscript>"
        + '<h2>Complete comparison</h2><div class="table"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + table
        + "</tbody></table></div>"
        + "<details><summary>Aggregate evidence and provenance</summary><pre>"
        + html.escape(json.dumps(result, indent=2, sort_keys=True))
        + "</pre></details></html>"
    )
    path = directory / "report.html"
    atomic_write(path, page.encode())
    return [*outputs, path]


def publish_selection_report(
    root: Path, store: ExperimentStore, lease: WriterLease, logger: RunLogger
) -> None:
    """Execute and verify every S3 artifact before changing canonical public copies."""
    from home_credit.runtime.notebooks import execute_notebook

    directory = store.root / "report"
    evidence = directory / "selection.json"
    result = json.loads(evidence.read_text())
    with StageTimer(logger, "selection_report", heartbeat_seconds=15):
        outputs = write_selection_report(result, directory)
        notebook = directory / "08_model_selection.ipynb"
        write_selection_notebook(notebook)
        execute_notebook(
            root,
            notebook,
            logger,
            dependencies=[evidence, root / "uv.lock", Path(__file__)],
            receipt_path=store.root / "notebook_receipt.json",
            execution_root=directory,
        )
        for path in [evidence, *outputs, notebook]:
            lease.check()
            store.publish(path)
    for path in [evidence, *outputs]:
        atomic_write(root / "reports/model_selection" / path.name, path.read_bytes())
    canonical = root / "notebooks/08_model_selection.ipynb"
    atomic_write(canonical, notebook.read_bytes())
    logger.event("selection_report_ready", notebook=canonical, outer_holdout_touched=False)
