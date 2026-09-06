#!/usr/bin/env python3
"""Prepare persistent storage and resume acceptance without invoking model training.

This dependency-free entry point must work before .venv exists. Legacy symlink
targets are preserved. All child processes inherit persistent cache locations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

UV_VERSION = "0.12.10"
PYTHON_VERSION = "3.12.14"
GIB = 1024**3


class Journal:
    """UTC events and child output persisted independently of the terminal."""

    def __init__(self, root: Path) -> None:
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        name = f"bootstrap-{datetime.now(UTC):%Y%m%dT%H%M%S}-{os.getpid()}"
        self.text_path = logs / f"{name}.log"
        self.json_path = logs / f"{name}.jsonl"
        self.lock = threading.Lock()

    def line(self, value: str) -> None:
        with self.lock, self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(value + "\n")
            print(value, flush=True)

    def event(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        encoded = json.dumps(row, sort_keys=True, default=str)
        with self.lock, self.json_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        self.line(encoded)


def run_stage(
    name: str,
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    journal: Journal,
    heartbeat_seconds: float,
) -> None:
    """Stream output and emit heartbeats even when a child is silent."""
    started = time.monotonic()
    journal.event("stage_started", stage=name)
    with subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        assert process.stdout is not None

        def consume(stream: TextIO) -> None:
            for line in stream:
                journal.line(line.rstrip("\n"))

        reader = threading.Thread(target=consume, args=(process.stdout,), daemon=True)
        reader.start()
        try:
            while True:
                try:
                    code = process.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    journal.event(
                        "heartbeat",
                        stage=name,
                        pid=process.pid,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            reader.join(timeout=5)
    journal.event(
        "stage_completed" if code == 0 else "stage_failed",
        stage=name,
        exit_code=code,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    if code:
        raise subprocess.CalledProcessError(code, command)


def storage_probe(root: Path) -> dict[str, Any]:
    """Read actual mount capacity, not the Studio configuration value."""
    mount: dict[str, Any] = {"target": "unknown", "source": "unknown", "fstype": "unknown"}
    command = shutil.which("findmnt")
    if command:
        result = subprocess.run(
            [command, "--json", "--target", str(root), "--output", "TARGET,SOURCE,FSTYPE"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        mount = json.loads(result.stdout)["filesystems"][0]
    disk = shutil.disk_usage(root)
    return {**mount, "free_gib": disk.free / GIB, "total_gib": disk.total / GIB}


def validate_storage(probe: dict[str, Any], minimum: float, require_persistent: bool) -> None:
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("minimum free GiB must be positive and finite")
    if require_persistent and probe["fstype"] in {"overlay", "tmpfs", "ramfs", "unknown"}:
        raise ValueError(f"project is not on a verified persistent mount: {probe}")
    if probe["free_gib"] < minimum:
        raise ValueError(
            f"persistent storage capacity insufficient: free_gib={probe['free_gib']:.2f} "
            f"required_gib={minimum:.2f}; inspect the mounted space volume. "
            "No environment was deleted and no fallback to /tmp is allowed."
        )


def runtime_environment(root: Path) -> dict[str, str]:
    """Override historical temporary paths for every child process."""
    runtime = root / "artifacts" / "runtime"
    environment = dict(os.environ)
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
            "UV_CACHE_DIR": str(runtime / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(runtime / "python"),
            "PIP_CACHE_DIR": str(runtime / "pip-cache"),
            "PRE_COMMIT_HOME": str(runtime / "pre-commit"),
            "TMPDIR": str(runtime / "tmp"),
            "UV_LINK_MODE": "copy",
            "HOME_CREDIT_BOOTSTRAP_ACTIVE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for key in ("VIRTUAL_ENV", "UV_PYTHON", "UV_NO_MANAGED_PYTHON"):
        environment.pop(key, None)
    environment["PATH"] = (
        str(runtime / "tools" / "bin")
        + os.pathsep
        + str(Path.home() / ".local" / "bin")
        + os.pathsep
        + environment.get("PATH", "")
    )
    return environment


def interpreter_info(venv: Path) -> dict[str, str] | None:
    python = venv / "bin" / "python"
    if not python.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import json,sys; print(json.dumps({"
                "'version':'.'.join(map(str,sys.version_info[:3])),"
                "'base':sys.base_prefix}))",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return dict(json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def prepare_venv_path(root: Path) -> None:
    """Detach a legacy link only; refuse to overwrite unrelated directories."""
    venv = root / ".venv"
    marker = root / "artifacts" / "runtime" / "environment.json"
    if venv.is_symlink():
        venv.unlink()
    elif venv.exists():
        if not venv.is_dir():
            raise ValueError(".venv is an ordinary file; preserved for inspection")
        owned = marker.is_file() and json.loads(marker.read_text()).get("venv") == str(venv)
        if not (venv / "pyvenv.cfg").is_file() and not owned and any(venv.iterdir()):
            raise ValueError(".venv is a non-environment directory; preserved for inspection")
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"venv": str(venv), "python": PYTHON_VERSION}) + "\n")
    temporary.replace(marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-benchmark", action="store_true")
    parser.add_argument("--bucket", default=os.environ.get("HOME_CREDIT_ARTIFACT_BUCKET"))
    parser.add_argument("--require-persistent-storage", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.accept_benchmark and not args.bucket:
        parser.error("--accept-benchmark requires --bucket or HOME_CREDIT_ARTIFACT_BUCKET")
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("heartbeat seconds must be positive and finite")
    root = Path(__file__).resolve().parents[1]
    journal = Journal(root)
    started = time.monotonic()
    journal.event("start_here_started", project_root=root, log_file=journal.text_path)
    environment = runtime_environment(root)

    def run(name: str, command: list[str]) -> None:
        run_stage(
            name,
            command,
            root=root,
            environment=environment,
            journal=journal,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    try:
        with (root / "logs" / "bootstrap.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            probe = storage_probe(root)
            journal.event("storage_preflight", **probe, required_free_gib=args.min_free_gib)
            validate_storage(probe, args.min_free_gib, args.require_persistent_storage)
            runtime = root / "artifacts" / "runtime"
            for path in [
                runtime,
                *(
                    Path(environment[k])
                    for k in (
                        "UV_CACHE_DIR",
                        "UV_PYTHON_INSTALL_DIR",
                        "PIP_CACHE_DIR",
                        "TMPDIR",
                        "PRE_COMMIT_HOME",
                    )
                ),
                runtime / "tools",
            ]:
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"runtime storage escapes the project: {path}")
                path.mkdir(parents=True, exist_ok=True)
                if path.stat().st_dev != root.stat().st_dev:
                    raise ValueError(f"runtime storage is on a different mount: {path}")
            run("source_validation", [sys.executable, "scripts/source_check.py"])
            uv = shutil.which("uv", path=environment["PATH"])
            version = (
                subprocess.run(
                    [uv, "--version"], capture_output=True, text=True, check=True, timeout=15
                ).stdout.strip()
                if uv
                else ""
            )
            if version.split()[:2] != ["uv", UV_VERSION]:
                run(
                    "install_uv",
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--no-deps",
                        "--target",
                        str(runtime / "tools"),
                        f"uv=={UV_VERSION}",
                    ],
                )
                uv = str(runtime / "tools" / "bin" / "uv")
            assert uv is not None
            venv = root / ".venv"
            info = interpreter_info(venv)
            managed_root = Path(environment["UV_PYTHON_INSTALL_DIR"])
            reusable = (
                not venv.is_symlink()
                and info is not None
                and info["version"] == PYTHON_VERSION
                and Path(info["base"]).is_relative_to(managed_root)
            )
            if reusable:
                journal.event("environment_reused", path=venv, python=info)
            else:
                run("persistent_python", [uv, "python", "install", PYTHON_VERSION])
                prepare_venv_path(root)
                run(
                    "persistent_environment",
                    [
                        uv,
                        "venv",
                        "--allow-existing",
                        "--managed-python",
                        "--python",
                        PYTHON_VERSION,
                        str(venv),
                    ],
                )
                info = interpreter_info(venv)
                if (
                    info is None
                    or info["version"] != PYTHON_VERSION
                    or not Path(info["base"]).is_relative_to(managed_root)
                ):
                    raise ValueError("persistent Python interpreter verification failed")
                journal.event("environment_created", path=venv, python=info)
            assert info is not None
            environment["UV_PYTHON"] = str(Path(info["base"]) / "bin" / "python3.12")
            environment["UV_PYTHON_PREFERENCE"] = "only-managed"
            run("bootstrap", ["bash", "scripts/bootstrap.sh"])
            if args.accept_benchmark:
                run(
                    "benchmark_acceptance",
                    [
                        str(venv / "bin" / "python"),
                        "scripts/accept_model_benchmark.py",
                        "--download",
                        "--bucket",
                        args.bucket,
                        "--benchmark-dir",
                        str(root / "artifacts" / "benchmark_acceptance"),
                        "--output-dir",
                        str(root / "reports" / "benchmark"),
                        "--heartbeat-seconds",
                        str(args.heartbeat_seconds),
                    ],
                )
            journal.event(
                "start_here_completed",
                total_elapsed_seconds=round(time.monotonic() - started, 3),
                lock_sha256=hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
            )
            return 0
    except (Exception, KeyboardInterrupt) as exc:
        journal.event(
            "start_here_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            total_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
