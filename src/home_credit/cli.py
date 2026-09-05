"""Command-line interface for project diagnostics and reproducible utilities."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from home_credit.data.manifest import build_manifest, write_manifest
from home_credit.metrics.stability import stability_score
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.environment import collect_environment_report, require_free_disk
from home_credit.runtime.smoke import dataframe_roundtrip, model_smoke

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("doctor")
def doctor() -> None:
    """Import the core stack and print versions plus machine capacity."""
    logger = RunLogger("doctor", Path("logs"))
    with StageTimer(logger, "environment_imports", heartbeat_seconds=15):
        report = collect_environment_report()
    logger.event(
        "environment",
        python=report.python,
        cpu_count=report.cpu_count,
        memory_gib=report.memory_gib,
        free_disk_gib=report.free_disk_gib,
    )
    for package, package_version in sorted(report.packages.items()):
        logger.event("package_version", package=package, version=package_version)


@app.command("data-preflight")
def data_preflight(
    min_free_gib: Annotated[float, typer.Option(min=1.0)] = 30.0,
    path: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
) -> None:
    """Require enough local disk capacity before downloading competition data."""
    logger = RunLogger("data-preflight", Path("logs"))
    probe_path = path if path is not None else Path.cwd()
    with StageTimer(logger, "storage_preflight"):
        available_gib = require_free_disk(min_free_gib, probe_path)
    logger.event(
        "data_preflight_passed",
        available_gib=available_gib,
        required_gib=min_free_gib,
        path=probe_path.as_posix(),
    )


@app.command("dataframe-smoke")
def dataframe_smoke() -> None:
    """Verify dataframe and Arrow interoperability."""
    logger = RunLogger("dataframe-smoke", Path("logs"))
    with StageTimer(logger, "dataframe_roundtrip"):
        rows, columns = dataframe_roundtrip()
    logger.event("dataframe_smoke_passed", rows=rows, columns=columns)


@app.command("model-smoke")
def model_smoke_command() -> None:
    """Fit tiny LightGBM, CatBoost, and XGBoost models."""
    logger = RunLogger("model-smoke", Path("logs"))
    with StageTimer(logger, "gbdt_compatibility", heartbeat_seconds=10):
        results = model_smoke()
    for result in results:
        logger.event("model_smoke_passed", model=result.model, auc=round(result.auc, 6))


@app.command("metric-smoke")
def metric_smoke() -> None:
    """Exercise the temporal stability metric on deterministic synthetic data."""
    logger = RunLogger("metric-smoke", Path("logs"))
    rng = np.random.default_rng(20260904)
    weeks = np.repeat(np.arange(8), 100)
    target = rng.binomial(1, 0.25, size=weeks.size)
    prediction = np.clip(0.1 + 0.65 * target + rng.normal(0.0, 0.12, weeks.size), 0.0, 1.0)
    with StageTimer(logger, "stability_metric"):
        result = stability_score(target, prediction, weeks)
    logger.event(
        "metric_smoke_passed",
        score=round(result.score, 6),
        mean_gini=round(result.mean_gini, 6),
        slope=round(result.slope, 8),
        residual_std=round(result.residual_std, 6),
    )


@app.command("heartbeat-smoke")
def heartbeat_smoke(
    seconds: Annotated[int, typer.Option(min=1)] = 4,
    interval: Annotated[float, typer.Option(min=0.1)] = 1.0,
) -> None:
    """Demonstrate timestamped heartbeat behavior."""
    logger = RunLogger("heartbeat-smoke", Path("logs"))
    with StageTimer(logger, "heartbeat_demo", heartbeat_seconds=interval):
        time.sleep(seconds)
    logger.event("heartbeat_smoke_passed")


@app.command("data-manifest")
def data_manifest(
    root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option()],
    hashes: Annotated[bool, typer.Option("--hashes")] = False,
) -> None:
    """Generate a deterministic file manifest."""
    logger = RunLogger("data-manifest", Path("logs"))
    with StageTimer(logger, "manifest_generation", heartbeat_seconds=30):
        manifest = build_manifest(root, include_hashes=hashes)
        write_manifest(manifest, output)
    logger.event(
        "manifest_written",
        output=output.as_posix(),
        file_count=manifest["file_count"],
        total_bytes=manifest["total_bytes"],
        hashes=hashes,
    )


@app.command("version-json")
def version_json() -> None:
    """Print machine-readable environment metadata."""
    report = collect_environment_report()
    typer.echo(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
