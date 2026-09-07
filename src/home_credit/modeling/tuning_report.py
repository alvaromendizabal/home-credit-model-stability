"""Readable, offline study reports with static notebook figures and explicit scope."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import plotly.graph_objects as go

from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes
from home_credit.modeling.experiment_store import ExperimentStore
from home_credit.modeling.tuning import rank_records
from home_credit.observability.logging import RunLogger
from home_credit.runtime.notebooks import execute_notebook


def study_figure(state: dict[str, Any]) -> Any:
    """Plot observed stability and fold behavior, including the reused control."""
    records = [r for r in state["trials"] if r["state"] == "complete"]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "home-credit-model-tuning"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    fig.patch.set_facecolor("#f5f8fc")
    baseline = records[0]["value"]
    axes[0].plot(
        [r["slot"] for r in records],
        [r["value"] for r in records],
        "o-",
        color="#087f8c",
        linewidth=2,
    )
    axes[0].axhline(baseline, color="#5e6c84", linestyle="--", label="Reused control")
    axes[0].set(
        xlabel="Completed trial (0 = control)",
        ylabel="Mean fold stability",
        title="Did tuning improve temporal stability?",
    )
    axes[0].legend(frameon=False)
    best = rank_records(records)[0]
    for record, color in [(records[0], "#5e6c84"), (best, "#087f8c")]:
        rows = sorted(record["folds"], key=lambda r: r["fold"])
        axes[1].plot(
            [r["fold"] for r in rows],
            [r["stability_score"] for r in rows],
            "o-",
            label=record["name"],
            color=color,
            linewidth=2,
        )
        if best["slot"] == 0:
            break
    axes[1].set(
        xlabel="Expanding temporal fold",
        ylabel="Official stability score",
        title="Where does the leading candidate improve?",
    )
    axes[1].set_xticks(range(1, 6))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Home Credit | Development model tuning", fontsize=17, fontweight="bold")
    return fig


def write_report(state: dict[str, Any], path: Path) -> None:
    """Write an embedded-JavaScript report with full metrics and parameter values."""
    records = rank_records(state["trials"])
    figure = go.Figure()
    for record in records:
        figure.add_trace(
            go.Scatter(
                x=[r["fold"] for r in record["folds"]],
                y=[r["stability_score"] for r in record["folds"]],
                name=record["name"],
                mode="lines+markers",
            )
        )
    figure.update_layout(
        template="plotly_white",
        title="Every candidate across time",
        xaxis_title="Temporal fold",
        yaxis_title="Official stability score",
    )
    columns = [
        ("experiment", "Candidate"),
        ("mean_fold_stability", "Stability"),
        ("delta_vs_control", "Change"),
        ("worst_fold_stability", "Worst fold"),
        ("oof_auc", "ROC AUC"),
        ("oof_pr_auc", "Average precision"),
        ("oof_brier_score", "Brier"),
        ("oof_log_loss", "Log loss"),
    ]
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{label}</th>" for _, label in columns)
        + "</tr></thead><tbody>"
    )
    for record in records:
        cells = []
        for key, _ in columns:
            value = record["metrics"][key]
            cells.append(
                f"<td>{html.escape(value) if isinstance(value, str) else f'{value:.6f}'}</td>"
            )
        table += "<tr>" + "".join(cells) + "</tr>"
    table += "</tbody></table>"
    mode = "SMOKE CHECK: capped data" if state["identity"]["smoke"] else "FULL DEVELOPMENT STUDY"
    status = "Complete" if state["complete"] else "In progress"
    completed = len(records) - 1
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home Credit | Model tuning</title><style>
body{font:16px system-ui;background:#f5f8fc;color:#19334a;
max-width:1250px;margin:40px auto;padding:0 24px}
h1{font-size:36px}p{line-height:1.65;max-width:1000px}
.card{background:white;border-radius:12px;padding:24px;margin:24px 0}
.eyebrow{color:#087f8c;font-weight:700}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:12px;text-align:left;border-bottom:1px solid #dbe3ea}section{overflow:auto}
pre{white-space:pre-wrap;background:white;padding:20px}summary{cursor:pointer;font-weight:600}
</style><p class="eyebrow">HOME CREDIT / TEMPORAL VALIDATION</p><h1>Model tuning</h1>"""
    page += (
        f"<p><strong>{mode}</strong> · {status} · "
        f"{completed}/{state['new_trial_budget']} new candidates complete.</p>"
    )
    page += "<p>Optuna searches LightGBM capacity, sampling and regularization. "
    page += "All 700 features are retained. "
    page += "Selection maximizes the mean official stability score over five expanding time folds. "
    page += "The control reuses the accepted run. ROC AUC and average precision assess ranking; "
    page += "Brier and log loss assess probabilities. "
    page += "Early stopping and selection use development folds; weeks 73-91 remain locked. "
    page += "These results guide model selection. Final holdout evaluation remains pending.</p>"
    page += "<section class='card'>" + table + "</section>"
    page += figure.to_html(full_html=False, include_plotlyjs=True, div_id="tuning-folds")
    page += "<details class='card'><summary>Sampled parameters and provenance</summary><pre>"
    page += (
        html.escape(
            json.dumps(
                {
                    "identity": state["identity"],
                    "parameters": {r["name"]: r["params"] for r in records},
                },
                indent=2,
            )
        )
        + "</pre></details></html>"
    )
    atomic_write(path, page.encode())


def write_notebook(state: dict[str, Any], path: Path) -> None:
    """Build a notebook whose displayed figures remain readable directly on GitHub."""
    cells = [
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "# Home Credit: Model tuning\n\n"
            "A bounded Optuna study on five expanding temporal folds. "
            "All 700 features are retained. The accepted control is reused. "
            "These are development results; the final holdout remains locked."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "import json\nfrom pathlib import Path\nimport pandas as pd\n"
            "from IPython.display import display\n"
            "from home_credit.modeling.tuning import rank_records\n"
            "from home_credit.modeling.tuning_report import study_figure\n"
            "state = json.loads(Path('study.json').read_text())\n"
            "assert state['outer_holdout_touched'] is False\n"
            "print('Smoke:', state['identity']['smoke'], '| Complete:', state['complete'])\n"
            "display(pd.DataFrame([r['metrics'] for r in rank_records(state['trials'])]))"
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "from io import BytesIO\nfrom IPython.display import Image\n"
            "import matplotlib.pyplot as plt\nfig = study_figure(state)\n"
            "buffer = BytesIO()\nfig.savefig(buffer, format='png', dpi=140)\n"
            "display(Image(data=buffer.getvalue()))\nplt.close(fig)"
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "leader = rank_records(state['trials'])[0]\n"
            "print('Development leader:', leader['name'])\n"
            "print(json.dumps(leader['params'], indent=2, sort_keys=True))"
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Interpretation\n\nThe official weekly Gini stability formula penalizes declining "
            "performance over time and residual volatility. "
            "This study ranks the mean of five fold scores; "
            "the pooled OOF AUC is a supporting metric. Average precision is reported as PR AUC. "
            "Inspect the worst fold and probability metrics before expanding tuning. "
            "Hyperparameter importance is intentionally omitted for this small study.\n\n"
            "Next: review the candidate, evaluate complementary models and ensembles, "
            "freeze selection, and evaluate the holdout once. "
            "Then build and validate Kaggle inference and submission artifacts."
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"tuning-{index}"
    notebook = nbformat.v4.new_notebook(cells=cells)  # type: ignore[no-untyped-call]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.validate(notebook)
    if path.exists():
        old = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]
        if [c.source for c in old.cells] == [c.source for c in cells]:
            return
    atomic_write(path, nbformat.writes(notebook).encode())  # type: ignore[no-untyped-call]


def publish_report(
    root: Path, state: dict[str, Any], store: ExperimentStore, logger: RunLogger, *, execute: bool
) -> None:
    """Refresh the view after every completed trial and execute the final notebook."""
    directory = store.root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    evidence = directory / "study.json"
    atomic_write(evidence, canonical_json_bytes(state))
    write_report(state, directory / "report.html")
    figure = study_figure(state)
    figure.savefig(directory / "overview.svg", metadata={"Date": None})
    plt.close(figure)
    outputs = [evidence, directory / "report.html", directory / "overview.svg"]
    if execute:
        notebook = directory / "07_model_tuning.ipynb"
        write_notebook(state, notebook)
        execute_notebook(
            root,
            notebook,
            logger,
            dependencies=[evidence, root / "uv.lock", Path(__file__)],
            receipt_path=store.root / "notebook_receipt.json",
            execution_root=directory,
        )
        outputs.append(notebook)
    for output in outputs:
        store.publish(output)
