from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bootstrap_environment_test", ROOT / "scripts/bootstrap_environment.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target_exists", [False, True])
def test_legacy_link_detaches_without_deleting_target_or_models(
    runtime: ModuleType, tmp_path: Path, target_exists: bool
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "legacy"
    if target_exists:
        target.mkdir()
        (target / "keep.txt").write_text("existing environment")
    (project / ".venv").symlink_to(target, target_is_directory=True)
    (project / "model.bin").write_bytes(b"completed model")
    runtime.prepare_venv_path(project)
    assert not (project / ".venv").is_symlink()
    assert (project / "model.bin").read_bytes() == b"completed model"
    if target_exists:
        assert (target / "keep.txt").read_text() == "existing environment"


def test_existing_environment_is_preserved(runtime: ModuleType, tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /old/python")
    (venv / "installed_package.py").write_text("value = 1")
    runtime.prepare_venv_path(tmp_path)
    assert (venv / "installed_package.py").read_text() == "value = 1"


@pytest.mark.parametrize("ordinary_file", [False, True])
def test_unrelated_venv_contents_are_preserved(
    runtime: ModuleType, tmp_path: Path, ordinary_file: bool
) -> None:
    venv = tmp_path / ".venv"
    if ordinary_file:
        venv.write_text("user data")
    else:
        venv.mkdir()
        (venv / "important.txt").write_text("user data")
    with pytest.raises(ValueError, match="preserved"):
        runtime.prepare_venv_path(tmp_path)
    marker = venv if ordinary_file else venv / "important.txt"
    assert marker.read_text() == "user data"


def test_interrupted_environment_creation_can_resume(runtime: ModuleType, tmp_path: Path) -> None:
    runtime.prepare_venv_path(tmp_path)
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "partial-download").write_text("retain")
    runtime.prepare_venv_path(tmp_path)
    assert (venv / "partial-download").read_text() == "retain"


@pytest.mark.parametrize("filesystem", ["tmpfs", "overlay", "ramfs", "unknown"])
def test_temporary_mount_is_rejected(runtime: ModuleType, filesystem: str) -> None:
    with pytest.raises(ValueError, match="persistent mount"):
        runtime.validate_storage({"fstype": filesystem, "free_gib": 100}, 12, True)


def test_insufficient_space_is_rejected(runtime: ModuleType) -> None:
    with pytest.raises(ValueError, match="capacity insufficient"):
        runtime.validate_storage({"fstype": "ext4", "free_gib": 2}, 12, True)


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf")])
def test_invalid_storage_threshold(runtime: ModuleType, minimum: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        runtime.validate_storage({"fstype": "ext4", "free_gib": 100}, minimum, True)


def test_runtime_paths_override_legacy_ephemeral_settings(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/tmp/old-environment")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/old-cache")
    environment = runtime.runtime_environment(tmp_path)
    for key in (
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "PIP_CACHE_DIR",
        "PRE_COMMIT_HOME",
        "TMPDIR",
    ):
        assert Path(environment[key]).is_relative_to(tmp_path)
    assert environment["UV_LINK_MODE"] == "copy"


def test_silent_child_has_heartbeat_and_elapsed_time(runtime: ModuleType, tmp_path: Path) -> None:
    journal = runtime.Journal(tmp_path)
    runtime.run_stage(
        "silent_child",
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        root=tmp_path,
        environment=dict(os.environ),
        journal=journal,
        heartbeat_seconds=0.03,
    )
    events = [json.loads(line) for line in journal.json_path.read_text().splitlines()]
    assert any(row["event"] == "heartbeat" for row in events)
    assert all("timestamp_utc" in row for row in events)
    assert events[-1]["event"] == "stage_completed"
    assert events[-1]["elapsed_seconds"] >= 0.2


def test_child_failure_propagates(runtime: ModuleType, tmp_path: Path) -> None:
    journal = runtime.Journal(tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        runtime.run_stage(
            "failed_child",
            [sys.executable, "-c", "raise SystemExit(7)"],
            root=tmp_path,
            environment=dict(os.environ),
            journal=journal,
            heartbeat_seconds=1,
        )
    assert failure.value.returncode == 7
    events = [json.loads(line) for line in journal.json_path.read_text().splitlines()]
    assert events[-1]["event"] == "stage_failed"
    assert events[-1]["exit_code"] == 7


def test_check_entry_point_recovers_link_before_calling_uv(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/check.sh", scripts / "check.sh")
    (scripts / "start_here.sh").write_text("echo startup_reached\n")
    (tmp_path / ".venv").symlink_to(tmp_path / "missing")
    result = subprocess.run(
        ["bash", str(scripts / "check.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME_CREDIT_BOOTSTRAP_ACTIVE": "0"},
        timeout=10,
    )
    assert result.returncode == 0
    assert "startup_reached" in result.stdout


def test_check_entry_point_prevents_recursive_bootstrap(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/check.sh", scripts / "check.sh")
    result = subprocess.run(
        ["bash", str(scripts / "check.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME_CREDIT_BOOTSTRAP_ACTIVE": "1"},
        timeout=10,
    )
    assert result.returncode != 0
    assert "Environment preparation failed" in result.stderr


def test_real_uv_reproduces_and_recovers_dangling_link(runtime: ModuleType, tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "environment-probe"\nversion = "0.1.0"\ndependencies = []\n'
    )
    (tmp_path / ".venv").symlink_to(tmp_path / "missing-environment")
    command = [uv, "run", "--offline", "--python", sys.executable, "python", "--version"]
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(tmp_path / ".venv")}
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0 and "File exists" in result.stderr
    runtime.prepare_venv_path(tmp_path)
    result = subprocess.run(
        [
            uv,
            "venv",
            "--offline",
            "--allow-existing",
            "--python",
            sys.executable,
            str(tmp_path / ".venv"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert runtime.interpreter_info(tmp_path / ".venv") is not None


def configure_workflow(
    runtime: ModuleType,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_stage: str | None = None,
) -> list[tuple[str, list[str]]]:
    """Inject installers while exercising the complete entry-point order."""
    monkeypatch.setattr(runtime, "__file__", str(root / "scripts/bootstrap_environment.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap_environment.py",
            "--accept-benchmark",
            "--bucket",
            "test-bucket",
            "--require-persistent-storage",
        ],
    )
    (root / "uv.lock").write_text("locked dependencies")
    (root / ".venv").mkdir()
    monkeypatch.setattr(runtime, "storage_probe", lambda _: {"fstype": "ext4", "free_gib": 100})
    monkeypatch.setattr(
        runtime,
        "interpreter_info",
        lambda _: {
            "version": runtime.PYTHON_VERSION,
            "base": str(root / "artifacts/runtime/python/cpython"),
        },
    )
    monkeypatch.setattr(runtime.shutil, "which", lambda *a, **k: "/configured/uv")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, f"uv {runtime.UV_VERSION}", ""),
    )
    stages: list[tuple[str, list[str]]] = []

    def stage(name: str, command: list[str], **kwargs: object) -> None:
        stages.append((name, command))
        if name == failing_stage:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(runtime, "run_stage", stage)
    return stages


def test_workflow_uses_persistent_acceptance_cache_and_reuses_environment(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = configure_workflow(runtime, tmp_path, monkeypatch)
    assert runtime.main() == 0
    assert [name for name, _ in stages] == [
        "source_validation",
        "bootstrap",
        "benchmark_acceptance",
    ]
    acceptance = stages[-1][1]
    assert acceptance[acceptance.index("--benchmark-dir") + 1] == str(
        tmp_path / "artifacts/benchmark_acceptance"
    )
    stages.clear()
    assert runtime.main() == 0
    assert all(name not in {"persistent_environment", "persistent_python"} for name, _ in stages)


def test_failed_quality_gate_cannot_start_acceptance(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = configure_workflow(runtime, tmp_path, monkeypatch, failing_stage="bootstrap")
    assert runtime.main() == 1
    assert [name for name, _ in stages] == ["source_validation", "bootstrap"]


def test_low_disk_stops_before_any_installation(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = configure_workflow(runtime, tmp_path, monkeypatch)
    marker = tmp_path / ".venv/keep.txt"
    marker.write_text("preserve installed work")
    monkeypatch.setattr(runtime, "storage_probe", lambda _: {"fstype": "ext4", "free_gib": 1})
    assert runtime.main() == 1
    assert stages == []
    assert marker.read_text() == "preserve installed work"


def test_concurrent_bootstrap_cannot_modify_environment(
    runtime: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = configure_workflow(runtime, tmp_path, monkeypatch)
    (tmp_path / "logs").mkdir()
    with (tmp_path / "logs/bootstrap.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runtime.main() == 1
    assert stages == []
