"""Structured UTC logging for reproducible experiment runs."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def iso_utc() -> str:
    """Return an ISO-8601 UTC timestamp with a Z suffix."""
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


class UtcFormatter(logging.Formatter):
    """Logging formatter that always renders timestamps in UTC."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format a log record timestamp in UTC without mutating global formatter state."""
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        if datefmt is not None:
            return timestamp.strftime(datefmt)
        return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class RunLogger:
    """Human-readable console logging plus machine-readable JSONL logging."""

    name: str
    logs_dir: Path
    level: int = logging.INFO
    jsonl_path: Path = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        self.jsonl_path = self.logs_dir / f"{self.name}-{stamp}.jsonl"

        logger_name = f"home_credit.{self.name}.{stamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        self.logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            UtcFormatter(
                "[%(asctime)sZ] %(levelname)s %(message)s",
                "%Y-%m-%dT%H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

    def event(self, event: str, **fields: Any) -> None:
        """Emit one structured event to console and JSONL."""
        payload = {"timestamp": iso_utc(), "event": event, **fields}
        self.logger.info("%s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
