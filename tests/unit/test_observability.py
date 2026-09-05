from __future__ import annotations

import json
import time
from pathlib import Path

from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def test_logger_writes_timestamped_jsonl(tmp_path: Path) -> None:
    logger = RunLogger("test", tmp_path)
    logger.event("hello", answer=42)
    lines = logger.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "hello"
    assert payload["answer"] == 42
    assert payload["timestamp"].endswith("Z")


def test_stage_timer_logs_start_and_completion(tmp_path: Path) -> None:
    logger = RunLogger("timer", tmp_path)
    with StageTimer(logger, "tiny"):
        time.sleep(0.01)
    events = [json.loads(line)["event"] for line in logger.jsonl_path.read_text().splitlines()]
    assert events == ["stage_started", "stage_completed"]
