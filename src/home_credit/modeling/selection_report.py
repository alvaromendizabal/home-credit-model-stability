"""Static notebook figures and an offline review of the bounded OOF comparison."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

from home_credit.modeling.checkpoints import atomic_write, sha256_file
from home_credit.modeling.selection import require
from home_credit.observability.runtime import StageTimer

if TYPE_CHECKING:
    from home_credit.modeling.experiment_store import ExperimentStore, WriterLease
    from home_credit.observability.logging import RunLogger


def selection_figures(result: dict[str, Any]) -> dict[str, Any]:
    """Return separate, labeled figures; none imply an untouched test result."""
    figures = {}
    records = list(reversed(result["rows"]))
    figure, axis = plt.subplots(figsize=(11, 7), layout="constrained")
    values = [row["mean_fold_stability"] for row in records]
    axis.barh([row["candidate"] for row in records], values)
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


def write_selection_notebook(path: Path) -> None:
    """Create the canonical report source without erasing unchanged executions."""
    digest = sha256_file(path.parent / "selection.json")
    cells = [
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "# Home Credit: Development model selection\n\n"
            "**15 predeclared candidates · 0 new model fits · holdout untouched.**\n\n"
            "This notebook reviews hash-verified, aligned development predictions. "
            "It is a model-selection report, not a final test or leaderboard result. "
            "Weeks 73-91 remain locked. Nothing is submitted to Kaggle."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "import hashlib\nimport json\nfrom pathlib import Path\nimport pandas as pd\n"
            "from IPython.display import display\n"
            "candidates = [Path.cwd() / 'selection.json'] + [\n"
            "    p / 'reports/model_selection/selection.json'\n"
            "    for p in [Path.cwd(), *Path.cwd().parents]\n"
            "]\n"
            f"expected_sha256 = {digest!r}\n"
            "source = next((p for p in candidates if p.is_file() and\n"
            "    hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha256), None)\n"
            "if source is None:\n"
            "    raise FileNotFoundError('Matching selection evidence is missing')\n"
            "result = json.loads(source.read_text())\n"
            "assert result['outer_holdout_touched'] is False\n"
            "assert result['new_model_fits'] == 0\n"
            "assert result['candidate_count'] == 15\n"
            "assert result['kaggle_submitted'] is False\n"
            "print('Scope:', result['scope'])\n"
            "print('Cases evaluated:', f\"{result['rows_evaluated']:,}\")\n"
            "columns = ['candidate', 'mean_fold_stability', 'worst_fold_stability', "
            "'oof_auc', 'oof_pr_auc', 'oof_brier_score', 'oof_log_loss']\n"
            "display(pd.DataFrame(result['rows'])[columns].round(6))"
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Stability, time and reliability\n\n"
            "Selection maximizes the mean of five official fold stability scores. "
            "ROC AUC and average precision describe ranking; raw Brier score and "
            "clipped log loss describe probabilities. Pooling OOF predictions is a "
            "supporting diagnostic, not the primary selection objective."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "from io import BytesIO\nfrom IPython.display import Image\n"
            "import matplotlib.pyplot as plt\n"
            "from home_credit.modeling.selection_report import selection_figures\n"
            "for name, figure in selection_figures(result).items():\n"
            "    buffer = BytesIO()\n"
            "    figure.savefig(buffer, format='png', dpi=125)\n"
            "    display(Image(data=buffer.getvalue()))\n"
            "    plt.close(figure)"
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Does the weight choice transfer to the next fold?\n\n"
            "The following diagnostic chooses blend weights using earlier folds "
            "and then observes the next fold. **This is not fully nested validation:** "
            "the tuned base model was selected using all development folds. "
            "It therefore cannot certify unbiased future performance."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "display(pd.DataFrame(result['prequential_weight_choices']).round(6))\n"
            "display(pd.DataFrame(result['diagnostics']['prediction_correlations']))\n"
            "print('Selected development candidate:', result['selected_candidate'])\n"
            "print(json.dumps(result['selected_weights'], indent=2, sort_keys=True))"
        ),
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            "## Decision boundary and next work\n\n"
            "A development improvement is evidence to investigate, not a claim "
            "of state-of-the-art performance. Review complementary challengers, "
            "freeze model/ensemble/calibration choices, and only then evaluate the "
            "locked holdout once. Train/test feature parity, the final refit and "
            "competition-compatible inference still require their own tested stage.\n\n"
            "Submission generation will be an explicit notebook action: generate, "
            "validate `case_id,score`, save durably and display a download link. "
            "Kaggle upload remains a separate action performed by the notebook owner. "
            "This report does not create a submission CSV."
        ),
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            "import os\nfrom IPython.display import FileLink\n"
            "display(FileLink(os.path.relpath(source), "
            "result_html_prefix='Aggregate evidence: '))\n"
            "display(FileLink(os.path.relpath(source.parent / 'report.html'), "
            "result_html_prefix='Offline review: '))"
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
    """Write static, self-contained HTML and figures without customer-level data."""
    require(result["outer_holdout_touched"] is False, "holdout result cannot enter this report")
    require(result["candidate_count"] == len(result["rows"]) == 15, "incomplete candidate report")
    directory.mkdir(parents=True, exist_ok=True)
    figures = selection_figures(result)
    outputs = []
    sections = []
    try:
        for name, figure in figures.items():
            path = directory / f"{name}.svg"
            figure.savefig(path, metadata={"Date": None})
            outputs.append(path)
            sections.append(path.read_text())
    finally:
        for figure in figures.values():
            plt.close(figure)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Home Credit | Development selection</title><style>"
        "body{font:16px system-ui;max-width:1180px;margin:40px auto;padding:0 24px;"
        "line-height:1.6}svg{width:100%;height:auto}pre{white-space:pre-wrap}"
        "</style><h1>Development model selection</h1><p>"
        "15 fixed candidates; no new training; weeks 73-91 remain locked. "
        "These are selection diagnostics, not final test results. "
        "No calibrator is fitted and nothing is submitted to Kaggle.</p><h2>"
        + html.escape(result["selected_candidate"])
        + "</h2>"
        + "".join(sections)
        + "<h2>Aggregate evidence and provenance</h2><pre>"
        + html.escape(json.dumps(result, indent=2, sort_keys=True))
        + "</pre></html>"
    )
    path = directory / "report.html"
    atomic_write(path, page.encode())
    return [*outputs, path]


def publish_selection_report(
    root: Path, store: ExperimentStore, lease: WriterLease, logger: RunLogger
) -> None:
    """Execute and upload every report artifact before the study can finish."""
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
    public_report = root / "reports/model_selection"
    for path in (evidence, directory / "report.html"):
        atomic_write(public_report / path.name, path.read_bytes())
    canonical = root / "notebooks/08_model_selection.ipynb"
    atomic_write(canonical, notebook.read_bytes())
    logger.event("selection_report_ready", notebook=canonical, outer_holdout_touched=False)
