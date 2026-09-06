"""Run timing and heartbeat primitives."""

from __future__ import annotations

import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import psutil

from home_credit.observability.logging import RunLogger


@dataclass(slots=True)
class Heartbeat(AbstractContextManager["Heartbeat"]):
    """Emit regular health signals while a long-running stage executes."""

    logger: RunLogger
    interval_seconds: float = 30.0
    label: str = "run"
    _started: float = field(init=False, default=0.0)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)

    def __enter__(self) -> Self:
        self._started = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"heartbeat-{self.label}",
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1.0))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            # Some containers hide their own PID from /proc. Progress must remain
            # visible even when optional resource telemetry is unavailable.
            try:
                process = psutil.Process(os.getpid())
                memory: float | None = round(process.memory_info().rss / (1024 * 1024), 1)
                cpu: float | None = psutil.cpu_percent(interval=None)
                telemetry_status = "available"
            except (psutil.Error, OSError) as exc:
                memory = None
                cpu = None
                telemetry_status = type(exc).__name__
            self.logger.event(
                "heartbeat",
                label=self.label,
                elapsed_seconds=round(time.monotonic() - self._started, 1),
                rss_mb=memory,
                cpu_percent=cpu,
                telemetry_status=telemetry_status,
            )


@dataclass(slots=True)
class StageTimer(AbstractContextManager["StageTimer"]):
    """Log stage start, completion, failure, and elapsed time."""

    logger: RunLogger
    stage: str
    heartbeat_seconds: float | None = None
    _started: float = field(init=False, default=0.0)
    _heartbeat: Heartbeat | None = field(init=False, default=None)

    def __enter__(self) -> Self:
        self._started = time.monotonic()
        self.logger.event("stage_started", stage=self.stage)
        if self.heartbeat_seconds is not None:
            self._heartbeat = Heartbeat(
                self.logger,
                interval_seconds=self.heartbeat_seconds,
                label=self.stage,
            )
            self._heartbeat.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._heartbeat is not None:
            self._heartbeat.__exit__(exc_type, exc_value, traceback)
        elapsed = round(time.monotonic() - self._started, 3)
        if exc_type is None:
            self.logger.event("stage_completed", stage=self.stage, elapsed_seconds=elapsed)
        else:
            self.logger.event(
                "stage_failed",
                stage=self.stage,
                elapsed_seconds=elapsed,
                error_type=exc_type.__name__,
                error=str(exc_value),
            )
