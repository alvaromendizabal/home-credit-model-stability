from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from home_credit.cli import app

runner = CliRunner()


def test_metric_smoke_cli() -> None:
    result = runner.invoke(app, ["metric-smoke"])
    assert result.exit_code == 0, result.output


def test_dataframe_smoke_cli() -> None:
    result = runner.invoke(app, ["dataframe-smoke"])
    assert result.exit_code == 0, result.output


def test_data_manifest_cli(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("a", encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        ["data-manifest", "--root", str(raw), "--output", str(output), "--hashes"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["file_count"] == 1
    assert len(payload["files"][0]["sha256"]) == 64
