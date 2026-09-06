#!/usr/bin/env python3
"""Dependency-free source-tree validation run before environment creation."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FORBIDDEN = re.compile(
    r"(^|[_-])(fix|fixed|repair|repaired|final[0-9]*|v[0-9]+)([_\.-]|$)",
    re.I,
)
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "data",
    "logs",
}
REQUIRED_PATHS = (
    "pyproject.toml",
    "scripts/start_here.sh",
    "scripts/bootstrap.sh",
    "scripts/check.sh",
    "scripts/git_setup.sh",
    "src/home_credit/cli.py",
    "src/home_credit/data/__init__.py",
    "src/home_credit/data/manifest.py",
    "src/home_credit/metrics/stability.py",
    "src/home_credit/observability/logging.py",
    "src/home_credit/observability/runtime.py",
    "src/home_credit/runtime/environment.py",
    "src/home_credit/runtime/smoke.py",
)


def stamp() -> str:
    return datetime.now(UTC).strftime("[%Y-%m-%dT%H:%M:%SZ]")


def log(message: str) -> None:
    print(f"{stamp()} {message}", flush=True)


def source_files() -> list[Path]:
    """Prune installed environments and persistent caches before traversal."""
    files: list[Path] = []
    for directory, names, filenames in os.walk(ROOT):
        names[:] = [name for name in names if name not in IGNORED_PARTS]
        files.extend(Path(directory) / name for name in filenames)
    return files


def resolve_internal_module(module: str) -> bool:
    relative = Path(*module.split("."))
    module_file = SRC / relative.with_suffix(".py")
    package_init = SRC / relative / "__init__.py"
    return module_file.is_file() or package_init.is_file()


def internal_import_errors() -> list[str]:
    errors: list[str] = []
    for path in sorted((SRC / "home_credit").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
                modules = [name for name in names if name.startswith("home_credit")]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module] if node.module.startswith("home_credit") else []
            else:
                modules = []
            for module in modules:
                if not resolve_internal_module(module):
                    errors.append(f"{path.relative_to(ROOT)} imports missing module {module}")
    return errors


def shell_syntax_errors() -> list[str]:
    errors: list[str] = []
    for script in sorted((ROOT / "scripts").glob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()
            errors.append(f"shell syntax failed for {script.relative_to(ROOT)}: {detail}")
    return errors


def main() -> int:
    log("source_check_started")
    errors: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    paths = source_files()
    for path in paths:
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if FORBIDDEN.search(path.name):
            errors.append(f"forbidden source filename: {path.relative_to(ROOT)}")

    errors.extend(internal_import_errors())

    for path in sorted(path for path in paths if path.suffix == ".py"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Python syntax failed for {path.relative_to(ROOT)}: {exc}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])
    required_dependencies = {"kaggle==2.2.4", "xgboost-cpu==3.4.1"}
    missing_dependencies = sorted(required_dependencies - dependencies)
    for dependency in missing_dependencies:
        errors.append(f"missing required dependency: {dependency}")

    dev_dependencies = set(pyproject["dependency-groups"]["dev"])
    required_dev_dependencies = {
        "pandas-stubs==3.0.5.260730",
        "types-psutil==7.2.2.20260827",
    }
    for dependency in sorted(required_dev_dependencies - dev_dependencies):
        errors.append(f"missing required development dependency: {dependency}")

    mypy_overrides = pyproject.get("tool", {}).get("mypy", {}).get("overrides", [])
    ignored_mypy_modules = {
        module
        for override in mypy_overrides
        if override.get("ignore_missing_imports") is True
        for module in override.get("module", [])
    }
    if "sklearn.*" not in ignored_mypy_modules:
        errors.append("mypy must explicitly ignore missing third-party sklearn stubs")
    if "xgboost==3.4.1" in dependencies:
        message = "CPU environment must use xgboost-cpu==3.4.1, not xgboost==3.4.1"
        errors.append(message)

    errors.extend(shell_syntax_errors())

    if errors:
        for error in errors:
            log(f"source_check_error detail={error}")
        log(f"source_check_failed error_count={len(errors)}")
        return 1

    log("source_check_passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
