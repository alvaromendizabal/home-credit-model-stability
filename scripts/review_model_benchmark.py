#!/usr/bin/env python3
"""Review accepted development metrics and execute the portfolio notebook."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
from pathlib import Path

from home_credit.modeling.acceptance import read_json
from home_credit.modeling.checkpoints import atomic_write
from home_credit.modeling.review import load_review, rescore_predictions, verify_rescored_metrics
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import Heartbeat, StageTimer
from home_credit.runtime.notebooks import execute_notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir", type=Path, help="Optionally recheck saved OOF predictions"
    )
    parser.add_argument(
        "--force", action="store_true", help="Reexecute the notebook instead of reusing it"
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive and finite")
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    logger = RunLogger("benchmark-review", root / "logs")
    logger.event("review_started", log=logger.jsonl_path)
    try:
        with (root / "logs/benchmark-review.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with Heartbeat(
                logger, interval_seconds=args.heartbeat_seconds, label="benchmark_review"
            ):
                with StageTimer(logger, "verify_review_evidence"):
                    evidence, review = load_review(root)
                if args.benchmark_dir is not None:
                    with StageTimer(logger, "recompute_saved_prediction_metrics"):
                        metrics = rescore_predictions(
                            evidence,
                            read_json(root / "configs/validation_protocol.json"),
                            args.benchmark_dir,
                        )
                        verify_rescored_metrics(
                            metrics, read_json(root / "reports/benchmark/metrics.json")
                        )
                with StageTimer(logger, "write_review_diagnostics"):
                    atomic_write(
                        root / "reports/benchmark/review.json",
                        (json.dumps(review, indent=2, allow_nan=False) + "\n").encode(),
                    )
                with StageTimer(logger, "execute_review_notebook"):
                    execute_notebook(
                        root, root / "notebooks/05_benchmark_review.ipynb", logger, force=args.force
                    )
            logger.event(
                "review_completed",
                leader=review["leader"],
                total_elapsed_seconds=round(time.monotonic() - started, 3),
            )
            print("PHASE_5B_REVIEW_COMPLETED", flush=True)
            return 0
    except (Exception, KeyboardInterrupt) as exc:
        logger.event(
            "review_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
