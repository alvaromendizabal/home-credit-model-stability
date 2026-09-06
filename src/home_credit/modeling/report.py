"""Portable, aggregate-only presentation of accepted benchmark evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from home_credit.modeling.checkpoints import atomic_write

LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "linear_logistic": "Logistic SGD",
}
COLORS = {
    "lightgbm": "#007d79",
    "xgboost": "#3566b5",
    "catboost": "#aa6925",
    "linear_logistic": "#a4506d",
}


def _chart(fig: Any, title: str, yaxis: str, *, first: bool = False) -> str:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 20}},
        font={"family": "Arial, sans-serif", "color": "#203342"},
        margin={"l": 65, "r": 28, "t": 64, "b": 65},
        height=440,
        yaxis_title=yaxis,
        legend={"orientation": "h", "y": -0.22},
        paper_bgcolor="#ffffff",
        hovermode="closest",
    )
    return str(
        fig.to_html(
            full_html=False,
            include_plotlyjs=first,
            config={"displaylogo": False, "responsive": True},
        )
    )


def build_report(evidence: dict[str, Any], destination: Path) -> None:
    """Write a self-contained HTML report, SVG preview, JSON evidence, and Markdown."""
    if evidence["status"] != "accepted":
        raise ValueError("only accepted evidence can be reported")
    destination.mkdir(parents=True, exist_ok=True)
    models = evidence["models"]
    names = [r["model"] for r in models]
    leader = models[0]
    selection = go.Figure()
    weekly = go.Figure()
    folds = go.Figure()
    calibration = go.Figure()
    for name in names:
        row = next(r for r in models if r["model"] == name)
        selection.add_trace(
            go.Bar(
                name=LABELS[name],
                x=[LABELS[name]],
                y=[row["mean_inner_stability_score"]],
                marker_color=COLORS[name],
                text=[f"{row['mean_inner_stability_score']:.4f}"],
                textposition="outside",
                showlegend=False,
            )
        )
        wr = [r for r in evidence["weekly_metrics"] if r["model"] == name]
        weekly.add_trace(
            go.Scatter(
                name=LABELS[name],
                x=[r["week_num"] for r in wr],
                y=[r["gini"] for r in wr],
                mode="lines+markers",
                marker={"size": 5},
                line={"color": COLORS[name]},
                customdata=[[r["rows"], r["positive_rate"]] for r in wr],
                hovertemplate="Week %{x}<br>Gini %{y:.4f}<br>Cases %{customdata[0]:,}"
                "<br>Positive rate %{customdata[1]:.2%}<extra>%{fullData.name}</extra>",
            )
        )
        fr = [r for r in evidence["fold_metrics"] if r["model"] == name]
        folds.add_trace(
            go.Scatter(
                name=LABELS[name],
                x=[r["fold"] for r in fr],
                y=[r["stability_score"] for r in fr],
                mode="lines+markers",
                line={"color": COLORS[name]},
                marker={"size": 8},
            )
        )
        cr = [r for r in evidence["calibration"] if r["model"] == name]
        calibration.add_trace(
            go.Scatter(
                name=LABELS[name],
                x=[r["prediction_mean"] for r in cr],
                y=[r["observed_rate"] for r in cr],
                customdata=[r["rows"] for r in cr],
                mode="lines+markers",
                line={"color": COLORS[name]},
                hovertemplate="Predicted %{x:.3f}<br>Observed %{y:.3f}"
                "<br>Cases %{customdata:,}<extra>%{fullData.name}</extra>",
            )
        )
    calibration.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#708090", "dash": "dash"},
        )
    )
    calibration.update_xaxes(title="Mean predicted probability", range=[0, 1])
    calibration.update_yaxes(range=[0, 1])
    weekly.update_xaxes(title="Development week", dtick=4)
    folds.update_xaxes(title="Temporal fold", dtick=1)
    charts = [
        (
            "Selection objective",
            "Each model receives equal weight across five temporal folds. "
            "Higher is better. The slope penalty makes deterioration costly.",
            _chart(selection, "Mean fold stability", "Stability score", first=True),
        ),
        (
            "Performance through time",
            "Hover for sample counts and positive rates. Click a legend "
            "item to hide or restore a model; double-click to isolate it.",
            _chart(weekly, "Weekly discrimination", "Normalized Gini"),
        ),
        (
            "Weak periods remain visible",
            "These are the five actual fold scores. Variation across "
            "folds describes different time windows, not a confidence interval.",
            _chart(folds, "Fold-by-fold stability", "Stability score"),
        ),
        (
            "Probability reliability",
            "Fixed probability bins; hover for bin counts. Sparse bins "
            "are uncertain. These predictions have not been recalibrated.",
            _chart(calibration, "Observed versus predicted risk", "Observed positive rate"),
        ),
    ]
    table_rows = "".join(
        f"<tr><td>{r['rank']}</td><th>{LABELS[r['model']]}</th>"
        f"<td>{r['mean_inner_stability_score']:.4f}</td>"
        f"<td>{r['worst_fold_stability_score']:.4f}</td><td>{r['oof_auc']:.4f}</td>"
        f"<td>{r['oof_pr_auc']:.4f}</td><td>{r['oof_brier_score']:.5f}</td></tr>"
        for r in models
    )
    limitations = "".join(f"<li>{html.escape(s)}</li>" for s in evidence["limitations"])
    sections = "".join(
        f"<section><h2>{title}</h2><p>{description}</p>{chart}</section>"
        for title, description, chart in charts
    )
    runtime_rows = "".join(
        f"<tr><th>{LABELS[name]}</th><td>{seconds / 60:.1f} minutes</td></tr>"
        for name, seconds in evidence.get("fit_seconds_by_model", {}).items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home Credit | Temporal model benchmark</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;
background:#f0f4f5;
color:#203342;
font:16px/1.6 Arial,sans-serif}}
main{{max-width:1180px;
margin:auto;
padding:48px 24px}}header{{padding:48px;
background:#143543;
color:white;
border-radius:18px}}
.eyebrow{{letter-spacing:.16em;
text-transform:uppercase;
font-size:12px;
font-weight:bold;
color:#99ded3}}
h1{{font-size:clamp(30px,5vw,52px);
line-height:1.1;
margin:18px 0}}header p{{max-width:760px;
color:#d5e6e9}}
.tag{{display:inline-block;
border:1px solid #7bb5b6;
padding:4px 12px;
border-radius:25px;
font-size:13px}}
.cards{{display:grid;
grid-template-columns:repeat(4,1fr);
gap:16px;
margin:24px 0}}
.card,section{{background:white;
border:1px solid #dbe4e7;
border-radius:14px;
padding:26px;
margin-bottom:22px}}
.card{{margin:0}}.value{{font-size:30px;
font-weight:bold;
color:#007d79}}.label{{font-size:13px;
color:#526a79}}
h2{{font-size:23px;
line-height:1.25;
margin:0 0 12px}}p{{margin:10px 0 20px}}.note{{border-left:4px solid #bf8840;
padding-left:18px}}
table{{width:100%;
border-collapse:collapse;
font-size:14px}}th,td{{text-align:right;
padding:12px;
border-bottom:1px solid #e5ebee}}
th{{text-align:left}}thead th{{background:#f0f5f6;
text-align:right}}thead th:nth-child(2){{text-align:left}}
.table-wrap{{overflow-x:auto}}li{{margin-bottom:10px}}code{{overflow-wrap:anywhere;
font-size:12px}}
footer{{font-size:13px;
color:#526a79;
padding:18px 4px}}@media(max-width:700px){{main{{padding:16px 10px}}header{{padding:28px}}
.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:18px}}.value{{font-size:25px}}}}
</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">Home Credit / Model stability</div>
<h1>Credit risk across time.</h1>
<p>A reproducible comparison of four model families on five expanding temporal folds.
Every published artifact is hash verified and every reported score is checked against saved
predictions.</p>
<span class="tag">Development benchmark accepted</span> <span class="tag">Final holdout
pending</span>
</header>
<div class="cards">
<div class="card">
<div class="value">{LABELS[leader["model"]]}</div>
<div class="label">Development leader</div>
</div>
<div class="card">
<div class="value">{leader["mean_inner_stability_score"]:.4f}</div>
<div class="label">Mean fold stability</div>
</div>
<div class="card">
<div class="value">{evidence["oof_rows_per_model"]:,}</div>
<div class="label">OOF cases per model</div>
</div>
<div class="card">
<div class="value">{evidence["model_folds"]}/{evidence["model_folds"]}</div>
<div class="label">Verified model folds</div>
</div>
</div>
<section>
<h2>The current result</h2>
<p>{LABELS[leader["model"]]} leads under the frozen mean-fold stability objective.
The pooled OOF AUC is {leader["oof_auc"]:.4f}. These are development results;
 the final evaluation on weeks
{evidence["holdout_week_min"]}-{evidence["holdout_week_max"]} remains pending.</p>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>Rank</th>
<th>Model</th>
<th>Mean stability ↑</th>
<th>Worst fold ↑</th>
<th>OOF AUC ↑</th>
<th>OOF AP ↑</th>
<th>OOF Brier ↓</th>
</tr>
</thead>
<tbody>{table_rows}</tbody>
</table>
</div>
<p>AP = average precision (reported as PR AUC by the benchmark).
The boosting models use {evidence["selected_features"]} screened features;
logistic SGD is a lightweight baseline using the first
{evidence["model_feature_counts"]["linear_logistic"]},
with training-only imputation and standardization.
OOF positive rate: {evidence["oof_positive_rate"]:.2%}. Descriptive constant-probability Brier
reference:
{evidence["descriptive_constant_brier"]:.5f}. A lower Brier score is better.</p>
</section>
{sections}<section>
<h2>What the evidence says to do next</h2>
<p class="note">The adversarial screening AUC is {evidence["screening_drift_auc"]:.4f}:
the screening sample's earlier and later periods are readily distinguishable. This is a
temporal-shift diagnostic,
not a credit-risk accuracy score and not proof of leakage.</p>
<ol>
<li>Investigate the weakest folds and the features driving the adversarial screen.</li>
<li>Run controlled LightGBM and XGBoost tuning and feature-block ablations under the existing
development protocol.</li>
<li>Evaluate time-respecting ensembles and calibration;
 keep the final holdout locked until decisions are frozen.</li>
</ol>
</section>
<section>
<h2>Compute and provenance</h2>
<table>
<tbody>{runtime_rows}</tbody>
</table>
<p>Times sum completed model-fit stages recorded in the verified logs. They exclude screening,
loading,
restarts, idle time, and reporting;
 they are not billable AWS time or dollar costs.</p>
<p>{evidence["verified_files"]} verified artifacts · {evidence["selected_features"]} screened
features ·
{evidence["categorical_features"]} categorical features · OOF weeks
{evidence["oof_week_min"]}-{evidence["oof_week_max"]}</p>
<p>Training commit: <code>{evidence["training_commit"]}</code>
<br>
Summary SHA-256: <code>{evidence["summary_sha256"]}</code>
<br>
Protocol SHA-256: <code>{evidence["validation_protocol_sha256"]}</code>
</p>
</section>
<section>
<h2>Scope and limitations</h2>
<ul>{limitations}</ul>
</section>
<footer>Alvaro Mendizabal · Home Credit model stability · Reproduce with
scripts/accept_model_benchmark.py</footer>
</main>
</body>
</html>"""
    atomic_write(destination / "report.html", document.encode("utf-8"))
    atomic_write(
        destination / "acceptance.json",
        (json.dumps(evidence, indent=2, allow_nan=False) + "\n").encode(),
    )
    _overview(evidence, destination / "overview.svg")
    markdown = [
        "# Home Credit temporal model benchmark",
        "",
        "Development artifact acceptance passed. Final holdout evaluation is pending.",
        "",
        "![Benchmark overview](overview.svg)",
        "",
        "Open `report.html` in a browser for interactive charts. All chart assets are embedded.",
        "",
        "| Model | Mean fold stability | Worst fold | OOF AUC | OOF AP | OOF Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        markdown.append(
            f"| {LABELS[row['model']]} | {row['mean_inner_stability_score']:.6f} | "
            f"{row['worst_fold_stability_score']:.6f} | {row['oof_auc']:.6f} | "
            f"{row['oof_pr_auc']:.6f} | {row['oof_brier_score']:.6f} |"
        )
    markdown.extend(
        [
            "",
            f"Verified {evidence['verified_files']} files and "
            f"{evidence['model_folds']} model folds; "
            f"{evidence['oof_rows_per_model']:,} OOF cases per model.",
            "",
            "## Interpretation",
            "",
            *[f"- {s}" for s in evidence["limitations"]],
            "",
            "## Next experiment",
            "",
            "Inspect weak-fold behavior and temporal shift, then run "
            "LightGBM/XGBoost tuning and feature-block ablations. Keep weeks 73-91 locked.",
            "",
            f"Training commit: `{evidence['training_commit']}`.",
            "",
            f"Summary SHA-256: `{evidence['summary_sha256']}`.",
            "",
        ]
    )
    atomic_write(destination / "README.md", "\n".join(markdown).encode())


def _overview(evidence: dict[str, Any], path: Path) -> None:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1, 1.4]})
        fig.patch.set_facecolor("#f7fafb")
        models = evidence["models"]
        names = [r["model"] for r in models][::-1]
        scores = [r["mean_inner_stability_score"] for r in models][::-1]
        axes[0].barh(
            [LABELS[n] for n in names], scores, color=[COLORS[n] for n in names], height=0.55
        )
        for i, score in enumerate(scores):
            axes[0].text(max(score, 0) + 0.025, i, f"{score:.4f}", va="center", fontweight="bold")
        axes[0].set_xlim(-0.2, 0.78)
        axes[0].axvline(0, color="#d3dfe4", linewidth=1)
        axes[0].set_xlabel("Mean fold stability · higher is better")
        axes[0].set_title("Predeclared selection objective", loc="left", pad=18, fontweight="bold")
        for name in [r["model"] for r in models]:
            rows = [r for r in evidence["weekly_metrics"] if r["model"] == name]
            axes[1].plot(
                [r["week_num"] for r in rows],
                [r["gini"] for r in rows],
                label=LABELS[name],
                color=COLORS[name],
                linewidth=1.9,
            )
        axes[1].set_title("Performance through time", loc="left", pad=18, fontweight="bold")
        axes[1].set_xlabel("Development week")
        axes[1].set_ylabel("Weekly normalized Gini")
        axes[1].set_ylim(0, 0.82)
        axes[1].grid(axis="y", alpha=0.18)
        axes[1].legend(loc="lower right", frameon=False, fontsize=9)
        fig.suptitle(
            "HOME CREDIT   /   TEMPORAL MODEL BENCHMARK",
            x=0.06,
            y=0.97,
            ha="left",
            fontsize=17,
            fontweight="bold",
            color="#143543",
        )
        fig.text(
            0.06,
            0.885,
            f"{evidence['model_folds']} verified model folds  ·  "
            f"{evidence['oof_rows_per_model']:,} OOF cases per model  ·  Final holdout pending",
            fontsize=10,
            color="#526a79",
        )
        fig.subplots_adjust(left=0.14, right=0.97, top=0.75, bottom=0.19, wspace=0.36)
        fig.text(
            0.06,
            0.055,
            "Development scores reflect early stopping and model selection. "
            "Pooled OOF scores do not replace final holdout evaluation.",
            fontsize=9,
            color="#526a79",
        )
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
