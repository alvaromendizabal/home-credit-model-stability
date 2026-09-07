"""Environment and binary compatibility checks."""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil

CORE_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "xgboost",
    "optuna",
)

IMPORT_NAMES = {
    "scikit-learn": "sklearn",
}

DISTRIBUTION_NAMES = {
    "xgboost": "xgboost-cpu",
}


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Core runtime information used by bootstrap and experiment metadata."""

    python: str
    platform: str
    cpu_count: int
    memory_gib: float
    free_disk_gib: float
    packages: dict[str, str]


def free_disk_gib(path: Path | None = None) -> float:
    """Return free filesystem capacity in GiB for ``path`` or the current directory."""
    probe_path = path if path is not None else Path.cwd()
    disk = shutil.disk_usage(probe_path)
    return round(disk.free / (1024**3), 2)


def require_free_disk(min_free_gib: float, path: Path | None = None) -> float:
    """Return free GiB or raise when capacity is below the requested safety floor."""
    if min_free_gib <= 0:
        raise ValueError("min_free_gib must be positive")
    available_gib = free_disk_gib(path)
    if available_gib < min_free_gib:
        raise RuntimeError(
            "insufficient free disk space: "
            f"available_gib={available_gib} required_gib={min_free_gib}"
        )
    return available_gib


def collect_environment_report(path: Path | None = None) -> EnvironmentReport:
    """Collect deterministic environment metadata and import every core package."""
    packages: dict[str, str] = {}
    for package in CORE_PACKAGES:
        importlib.import_module(IMPORT_NAMES.get(package, package.replace("-", "_")))
        try:
            packages[package] = version(DISTRIBUTION_NAMES.get(package, package))
        except PackageNotFoundError:
            packages[package] = "unknown"

    return EnvironmentReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(logical=True) or 1,
        memory_gib=round(psutil.virtual_memory().total / (1024**3), 2),
        free_disk_gib=free_disk_gib(path),
        packages=packages,
    )
