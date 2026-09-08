#!/usr/bin/env python3
"""Execute notebook 08 from pinned aggregate evidence; require no AWS or loan data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from home_credit.modeling.checkpoints import atomic_write, canonical_json_bytes, sha256_file
from home_credit.modeling.selection import require
from home_credit.modeling.selection_report import (
    validate_selection_evidence,
    write_selection_notebook,
    write_selection_report,
)
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook


class ReviewLogger(RunLogger):
    """Add invocation elapsed time to every stage, cell and heartbeat event."""

    def __init__(self, name: str, directory: Path) -> None:
        self.started = time.monotonic()
        super().__init__(name, directory)

    def event(self, event: str, **fields: Any) -> None:
        fields.setdefault("total_elapsed_seconds", round(time.monotonic() - self.started, 3))
        super().event(event, **fields)


def load_evidence(root: Path) -> tuple[Path, dict[str, Any]]:
    """Verify the publication pin and the complete scoring-code/lock/input lineage."""
    policy = json.loads((root / "configs/model_selection_review.json").read_text())
    require(policy["schema_version"] == 1, "unsupported publication policy")
    source = root / "reports/model_selection/selection.json"
    require(sha256_file(source) == policy["selection_sha256"], "selection evidence digest changed")
    result = json.loads(source.read_text())
    validate_selection_evidence(result)
    require(result["smoke"] is False, "synthetic evidence cannot be published as real results")
    plan = json.loads((root / "configs/model_selection.json").read_text())
    require(result["source_study_sha256"] == plan["study"]["sha256"], "tuning source changed")
    inputs = result["identity"]["inputs"]
    require(bool(inputs), "missing scoring lineage")
    for name, digest in inputs.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe lineage path")
        require(sha256_file(root / relative) == digest, f"scoring lineage changed: {name}")
    return source, result


def review(root: Path, *, force: bool = False) -> bool:
    """Publish only after successful execution; keep expensive selection untouched."""
    source, result = load_evidence(root)
    renderer = root / "src/home_credit/modeling/selection_report.py"
    work = root / "artifacts/model_selection_review" / sha256_file(source) / sha256_file(renderer)
    work.mkdir(parents=True, exist_ok=True)
    logger = ReviewLogger("selection-review", root / "logs")
    with StageTimer(logger, "selection_review", heartbeat_seconds=15):
        evidence = work / "selection.json"
        atomic_write(evidence, source.read_bytes())
        outputs = write_selection_report(result, work)
        notebook = work / "08_model_selection.ipynb"
        write_selection_notebook(notebook)
        receipt = work / "notebook_receipt.json"
        reused = execute_notebook(
            root,
            notebook,
            logger,
            force=force,
            dependencies=[
                evidence,
                renderer,
                root / "uv.lock",
                root / "configs/model_selection_review.json",
                root / "src/home_credit/runtime/notebooks.py",
                Path(__file__),
            ],
            receipt_path=receipt,
            execution_root=work,
        )
        publication: dict[str, Any] = {
            "schema_version": 1,
            "selection_sha256": sha256_file(source),
            "renderer_sha256": sha256_file(renderer),
            "execution_receipt": json.loads(receipt.read_text()),
            "artifacts": {},
        }
        for path in [*outputs, notebook]:
            relative = (
                Path("notebooks") / path.name
                if path.suffix == ".ipynb"
                else Path("reports/model_selection") / path.name
            )
            atomic_write(root / relative, path.read_bytes())
            publication["artifacts"][relative.as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        atomic_write(
            root / "reports/model_selection/publication.json", canonical_json_bytes(publication)
        )
    logger.event(
        "selection_review_completed",
        reused=reused,
        selected_candidate=result["selected_candidate"],
        new_model_fits=0,
        outer_holdout_touched=False,
    )
    return reused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    review(Path(__file__).resolve().parents[1], force=args.force)
    print("MODEL_SELECTION_REVIEW_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
