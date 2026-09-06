from __future__ import annotations

import threading

import psutil
import pytest

from home_credit.observability.runtime import Heartbeat


@pytest.mark.parametrize(
    "error", [psutil.NoSuchProcess(123), psutil.AccessDenied(123), OSError("proc")]
)
def test_heartbeat_continues_when_resource_telemetry_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    received = threading.Event()
    events = []

    class Logger:
        def event(self, name: str, **fields: object) -> None:
            events.append((name, fields))
            received.set()

    def unavailable(pid: int) -> None:
        raise error

    monkeypatch.setattr(psutil, "Process", unavailable)
    heartbeat = Heartbeat(Logger(), interval_seconds=0.01)
    with heartbeat:
        assert received.wait(2), "Heartbeat must survive unavailable /proc telemetry"
    assert events[0][0] == "heartbeat"
    assert events[0][1]["rss_mb"] is None
    assert events[0][1]["elapsed_seconds"] >= 0
    assert events[0][1]["telemetry_status"] == type(error).__name__
    assert not heartbeat._thread.is_alive()
