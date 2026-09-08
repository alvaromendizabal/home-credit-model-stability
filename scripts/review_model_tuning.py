#!/usr/bin/env python3
"""Execute the canonical tuning review without regenerating or erasing its source."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer
from home_credit.runtime.notebooks import execute_notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/07_model_tuning.ipynb"
    started = time.monotonic()
    logger = RunLogger("tuning-review", root / "logs")
    with StageTimer(logger, "tuning_review", heartbeat_seconds=15):
        execute_notebook(
            root,
            path,
            logger,
            force=args.force,
            dependencies=[
                root / "reports/model_tuning/metrics.json",
                root / "uv.lock",
                Path(__file__),
                root / "src/home_credit/runtime/notebooks.py",
            ],
            receipt_path=root / "artifacts/model_tuning_review/notebook.json",
        )
    logger.event(
        "tuning_review_completed",
        notebook=path,
        total_elapsed_seconds=round(time.monotonic() - started, 3),
    )


if __name__ == "__main__":
    main()
