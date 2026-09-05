#!/usr/bin/env python3
"""Run the frozen Home Credit temporal model-family benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from home_credit.modeling.runner import BenchmarkRunner, summary_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Home Credit temporal benchmark with early-window "
            "feature screening and resumable model-fold checkpoints."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("/tmp/home-credit-features"),
    )
    parser.add_argument(
        "--expected-feature-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--validation-protocol",
        type=Path,
        default=Path("configs/validation_protocol.json"),
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_benchmark.json"),
    )
    parser.add_argument(
        "--expected-config-sha256",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/home-credit-model-benchmark"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one capped fold with reduced boosting rounds.",
    )
    return parser


def _required_string(namespace: argparse.Namespace, name: str) -> str:
    value = getattr(namespace, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_path(namespace: argparse.Namespace, name: str) -> Path:
    value = getattr(namespace, name)
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a Path")
    return value


def main() -> int:
    """Execute the benchmark and print a deterministic compact receipt."""
    args = _parser().parse_args()
    runner = BenchmarkRunner(
        feature_dir=_required_path(args, "feature_dir"),
        expected_feature_manifest_sha256=_required_string(
            args,
            "expected_feature_manifest_sha256",
        ),
        protocol_path=_required_path(args, "validation_protocol"),
        expected_protocol_sha256=_required_string(
            args,
            "expected_protocol_sha256",
        ),
        config_path=_required_path(args, "config"),
        expected_config_sha256=_required_string(
            args,
            "expected_config_sha256",
        ),
        output_dir=_required_path(args, "output_dir"),
        logs_dir=_required_path(args, "logs_dir"),
        smoke=bool(args.smoke),
    )
    summary = runner.run()
    output_dir = _required_path(args, "output_dir")
    summary_path = output_dir / "benchmark_summary.json"
    ranking_raw = summary.get("models")
    if not isinstance(ranking_raw, list):
        raise RuntimeError("benchmark summary models must be a list")
    ranking = cast(list[dict[str, Any]], ranking_raw)

    print(f"BENCHMARK_SMOKE={bool(summary['smoke'])}")
    print(f"BENCHMARK_FOLDS={int(summary['folds'])}")
    print(f"BENCHMARK_MODELS={len(ranking)}")
    print(f"SELECTED_FEATURES={int(summary['selected_features'])}")
    print(f"OUTER_HOLDOUT_TOUCHED={bool(summary['outer_holdout_touched'])}")
    for row in ranking:
        print(
            "MODEL_RANK="
            f"{int(row['rank'])} "
            f"model={row['model']} "
            f"mean_inner_stability={float(row['mean_inner_stability_score']):.8f} "
            f"worst_fold={float(row['worst_fold_stability_score']):.8f} "
            f"oof_auc={float(row['oof_auc']):.8f}"
        )
    print(f"BENCHMARK_SUMMARY_SHA256={summary_sha256(summary_path)}")
    print(f"BENCHMARK_SUMMARY={summary_path}")
    print(f"BENCHMARK_LOG={runner.logger.jsonl_path}")
    print("MODEL_BENCHMARK_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
