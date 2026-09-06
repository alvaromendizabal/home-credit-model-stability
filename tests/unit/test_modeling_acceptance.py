from __future__ import annotations

import copy
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from home_credit.modeling.acceptance import (
    accept_benchmark,
    compare_number,
    prediction_metrics,
    ranking_key,
    read_json,
    validate_predictions,
)
from home_credit.modeling.checkpoints import (
    build_checkpoint_manifest,
    derive_run_key,
    manifest_bytes,
    sha256_bytes,
    sha256_file,
)
from home_credit.modeling.config import BenchmarkConfig
from home_credit.observability.logging import RunLogger


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": range(8),
            "WEEK_NUM": [33] * 4 + [34] * 4,
            "target": [0, 0, 1, 1] * 2,
            "prediction": [0.1, 0.2, 0.8, 0.9] * 2,
        }
    )


def test_known_perfect_ranking_and_probability_metrics() -> None:
    result = prediction_metrics(frame())
    assert result["auc"] == 1
    assert result["pr_auc"] == 1
    assert result["stability_score"] == pytest.approx(1)
    assert result["brier_score"] == pytest.approx(0.025)
    assert result["log_loss"] == pytest.approx(-np.log(0.9 * 0.8) / 2)


@pytest.mark.parametrize(
    ("column", "values", "error"),
    [
        ("prediction", [np.nan] * 8, "nonfinite"),
        ("prediction", [1.1] * 8, "invalid probability"),
        ("prediction", [-0.1] * 8, "invalid probability"),
        ("case_id", [1] * 8, "duplicate case_id"),
        ("WEEK_NUM", [73] * 8, "week coverage"),
        ("WEEK_NUM", [33] * 8, "week coverage"),
        ("WEEK_NUM", [33.1] * 8, "noninteger"),
        ("target", [0, 0, 0, 0, 1, 1, 1, 1], "single-class week"),
        ("target", [2] * 8, "nonbinary"),
        ("prediction", [None] * 8, "null prediction"),
    ],
)
def test_invalid_predictions_are_rejected(column: str, values: list, error: str) -> None:
    bad = frame().with_columns(pl.Series(column, values))
    with pytest.raises(ValueError, match=error):
        validate_predictions(bad, {33, 34}, "fixture")


def test_metric_comparison_rejects_nan_and_material_differences() -> None:
    for actual, stored in ((np.nan, 1), (1, np.inf), (0.6, 0.61)):
        with pytest.raises(ValueError, match="metric mismatch"):
            compare_number(actual, stored, "test")


def test_selection_uses_mean_fold_score_before_pooled_score() -> None:
    common = {
        "worst_fold_stability_score": 0.3,
        "mean_weekly_gini": 0.7,
        "mean_temporal_slope": 0.001,
        "mean_residual_std": 0.02,
        "mean_brier_score": 0.03,
    }
    a = {**common, "mean_inner_stability_score": 0.6, "oof_stability_score": 0.65}
    b = {**common, "mean_inner_stability_score": 0.5, "oof_stability_score": 0.75}
    assert sorted([b, a], key=ranking_key)[0] == a


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def seal(root: Path, identity: dict, run_key: str) -> dict:
    summary_sha = sha256_file(root / "benchmark_summary.json")
    write_json(
        root / "run_metadata.json",
        {
            **identity,
            "outer_holdout_touched": False,
            "feature_screen_sha256": sha256_file(root / "feature_screen.json"),
            **{
                f"{name}_sha256": sha256_file(root / f"{name}{ext}")
                for name, ext in (
                    ("benchmark_summary", ".json"),
                    ("benchmark_state", ".json"),
                    ("fold_metrics", ".jsonl"),
                    ("weekly_metrics", ".jsonl"),
                )
            },
        },
    )
    manifest = build_checkpoint_manifest(
        root, run_key=run_key, prefix="benchmark", sequence=20, completed_model_folds=20, **identity
    )
    payload = manifest_bytes(manifest)
    (root / "checkpoint_manifest.json").write_bytes(payload)
    return {
        "run_key": run_key,
        "manifest_sha256": sha256_bytes(payload),
        "summary_sha256": summary_sha,
        "s3_prefix": "benchmark",
        "region": "us-west-2",
    }


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, dict, dict]:
    """Small but complete four-model/five-fold publication; no fitted models required."""
    root = tmp_path / "bundle"
    root.mkdir()
    config, digest = BenchmarkConfig.load(Path("configs/model_benchmark.json"))
    protocol = read_json(Path("configs/validation_protocol.json"))
    identity = {
        "git_commit": "a" * 40,
        "feature_manifest_sha256": "b" * 64,
        "validation_protocol_sha256": protocol["protocol_sha256"],
        "benchmark_config_sha256": digest,
        "smoke": False,
    }
    run_key = derive_run_key(**identity)
    write_json(
        root / "feature_screen.json",
        {
            "identity": identity,
            "result": {
                "selected_features": [{"name": "feature", "block": "test", "categorical": False}],
                "drift_validation_auc": 0.8,
            },
        },
    )
    receipts, models, fold_rows, week_rows, events = {}, [], [], [], []
    for name in config.enabled_model_names:
        frames = []
        for fold in range(1, 6):
            weeks = np.repeat(np.arange(33 + 8 * (fold - 1), 41 + 8 * (fold - 1)), 4)
            pred = pl.DataFrame(
                {
                    "case_id": np.arange((fold - 1) * 32, fold * 32),
                    "WEEK_NUM": weeks,
                    "target": [0, 0, 1, 1] * 8,
                    "prediction": [0.1, 0.2, 0.8, 0.9] * 8,
                }
            )
            directory = root / name
            directory.mkdir(exist_ok=True)
            p, m = directory / f"fold_{fold}.parquet", directory / f"model_{fold}.txt"
            pred.write_parquet(p)
            m.write_text("test artifact; never deserialized")
            metrics = {
                **prediction_metrics(pred),
                "validation_rows": 32,
                "validation_positive_rate": 0.5,
                "best_iteration": 1,
                "features": 1,
            }
            receipts[f"{name}:fold_{fold}"] = {
                "model": name,
                "fold": fold,
                "metrics": metrics,
                "prediction_path": p.relative_to(root).as_posix(),
                "prediction_sha256": sha256_file(p),
                "model_path": m.relative_to(root).as_posix(),
                "model_sha256": sha256_file(m),
            }
            fold_rows.append({"model": name, "fold": fold, **metrics})
            events.append(
                {
                    "event": "stage_completed",
                    "stage": f"fit_{name}_fold_{fold}",
                    "elapsed_seconds": 1,
                    "timestamp": "2026-09-06T00:00:00Z",
                }
            )
            for week in np.unique(weeks):
                week_rows.append(
                    {
                        "model": name,
                        "week_num": int(week),
                        "rows": 4,
                        "positives": 2,
                        "positive_rate": 0.5,
                        "prediction_mean": 0.5,
                        "gini": 1,
                    }
                )
            frames.append(pred)
        oof = pl.concat(frames)
        path = root / name / "oof.parquet"
        oof.write_parquet(path)
        model_folds = [r for r in fold_rows if r["model"] == name]
        scores = np.asarray([r["stability_score"] for r in model_folds])
        row = {
            "model": name,
            "folds": 5,
            "mean_inner_stability_score": float(scores.mean()),
            "std_inner_stability_score": float(scores.std()),
            "worst_fold_stability_score": float(scores.min()),
            "oof_path": path.relative_to(root).as_posix(),
            "oof_sha256": sha256_file(path),
            **{f"oof_{k}": v for k, v in prediction_metrics(oof).items()},
        }
        for dest, source in (
            ("mean_weekly_gini", "mean_gini"),
            ("mean_auc", "auc"),
            ("mean_temporal_slope", "temporal_slope"),
            ("mean_residual_std", "residual_std"),
            ("mean_brier_score", "brier_score"),
        ):
            row[dest] = float(np.mean([r[source] for r in model_folds]))
        models.append(row)
    models.sort(key=ranking_key)
    for rank, row in enumerate(models, 1):
        row["rank"] = rank
    write_json(
        root / "benchmark_state.json",
        {
            "schema_version": 1,
            "identity": {
                **identity,
                "feature_screen_sha256": sha256_file(root / "feature_screen.json"),
            },
            "folds": receipts,
        },
    )
    write_json(
        root / "benchmark_summary.json",
        {
            **identity,
            "models": models,
            "outer_holdout_touched": False,
            "folds": 5,
            "selection_policy": {"primary": "mean_inner_stability_score"},
            "selected_features": 1,
            "categorical_features": 0,
            "selected_features_by_block": {"test": 1},
            "feature_screen_sha256": sha256_file(root / "feature_screen.json"),
        },
    )
    (root / "logs").mkdir()
    for path, rows in (
        (root / "fold_metrics.jsonl", fold_rows),
        (root / "weekly_metrics.jsonl", week_rows),
        (root / "logs/run.jsonl", events),
    ):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root, seal(root, identity, run_key), identity


def accept(root: Path, policy: dict) -> dict:
    return accept_benchmark(
        root,
        policy=policy,
        config_path=Path("configs/model_benchmark.json"),
        protocol_path=Path("configs/validation_protocol.json"),
        logger=RunLogger("test-acceptance", root.parent / "logs"),
    )


def test_complete_publication_accepts_and_renders(bundle: tuple, tmp_path: Path) -> None:
    from home_credit.modeling.report import build_report

    root, policy, _ = bundle
    result = accept(root, policy)
    assert result["status"] == "accepted"
    assert result["model_folds"] == 20
    assert result["oof_rows_per_model"] == 160
    assert result["holdout_predictions_present"] is False
    build_report(result, tmp_path / "report")
    html = (tmp_path / "report/report.html").read_text()
    assert "Final holdout pending" in " ".join(html.split())
    assert '<script src="' not in html
    assert "case_id" not in json.dumps(result)
    assert (tmp_path / "report/overview.svg").is_file()


def test_report_bytes_are_reproducible_across_processes(bundle: tuple, tmp_path: Path) -> None:
    root, policy, _ = bundle
    evidence = tmp_path / "evidence.json"
    write_json(evidence, accept(root, policy))
    for index in (1, 2):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; "
                "from home_credit.modeling.report import build_report; "
                "build_report(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))",
                str(evidence),
                str(tmp_path / str(index)),
            ],
            env={**os.environ, "PYTHONHASHSEED": str(index), "SOURCE_DATE_EPOCH": str(index)},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    for name in ("report.html", "overview.svg", "acceptance.json", "README.md"):
        assert (tmp_path / "1" / name).read_bytes() == (tmp_path / "2" / name).read_bytes(), name


def test_saved_predictions_rescore_without_training_and_reject_changed_bytes(bundle: tuple) -> None:
    from home_credit.modeling.review import rescore_predictions

    root, policy, _ = bundle
    evidence = accept(root, policy)
    protocol = read_json(Path("configs/validation_protocol.json"))
    metrics = rescore_predictions(evidence, protocol, root)
    assert len(metrics["models"]) == 4
    assert len(metrics["folds"]) == 20
    assert all(r["auc"] == 1.0 for r in metrics["models"])
    assert all(r["stability_score"] == pytest.approx(1.0) for r in metrics["folds"])
    path = root / evidence["models"][0]["oof_path"]
    path.write_bytes(b"changed prediction artifact")
    with pytest.raises(ValueError, match="prediction hash"):
        rescore_predictions(evidence, protocol, root)


@pytest.mark.parametrize("mutation", ["checksum", "summary", "missing_fold", "holdout"])
def test_corruption_and_semantic_inconsistency_fail_closed(bundle: tuple, mutation: str) -> None:
    root, policy, identity = bundle
    if mutation == "checksum":
        with (root / "lightgbm/model_1.txt").open("a") as stream:
            stream.write("changed")
    elif mutation == "summary":
        summary = read_json(root / "benchmark_summary.json")
        summary["models"][0]["oof_auc"] = 0.1
        write_json(root / "benchmark_summary.json", summary)
        policy = seal(root, identity, policy["run_key"])
    elif mutation == "missing_fold":
        state = read_json(root / "benchmark_state.json")
        state["folds"].pop("lightgbm:fold_1")
        write_json(root / "benchmark_state.json", state)
        policy = seal(root, identity, policy["run_key"])
    else:
        state = read_json(root / "benchmark_state.json")
        receipt = state["folds"]["lightgbm:fold_1"]
        path = root / receipt["prediction_path"]
        data = pl.read_parquet(path).with_columns(pl.lit(73, dtype=pl.Int64).alias("WEEK_NUM"))
        data.write_parquet(path)
        receipt["prediction_sha256"] = sha256_file(path)
        write_json(root / "benchmark_state.json", state)
        policy = seal(root, identity, policy["run_key"])
    with pytest.raises(ValueError):
        accept(root, policy)


def test_untrusted_manifest_fails_before_report(bundle: tuple) -> None:
    root, policy, _ = bundle
    policy = copy.deepcopy(policy)
    policy["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        accept(root, policy)


@pytest.mark.parametrize("corrupt", [False, True])
def test_s3_restore_verifies_bytes_and_reuses_cache(
    bundle: tuple, tmp_path: Path, corrupt: bool
) -> None:
    root, policy, _ = bundle
    manifest = read_json(root / "checkpoint_manifest.json")
    objects = {r["object_key"]: root / r["path"] for r in manifest["files"]}
    downloads = []

    class Client:
        def get_object(self, **kwargs: object) -> dict:
            return {"Body": io.BytesIO((root / "checkpoint_manifest.json").read_bytes())}

        def download_file(self, bucket: str, key: str, path: str) -> None:
            downloads.append(key)
            Path(path).write_bytes(b"corrupt" if corrupt else objects[key].read_bytes())

    restore = runpy.run_path("scripts/accept_model_benchmark.py")["restore_bundle"]
    cache = tmp_path / "cache"
    logger = RunLogger("test-restore", tmp_path / "logs")
    if corrupt:
        with pytest.raises(ValueError, match="download size mismatch"):
            restore(cache, policy, logger, bucket="test", client=Client())
        assert not (cache / "checkpoint_manifest.json").exists()
        assert not list(cache.rglob("*.download"))
    else:
        restore(cache, policy, logger, bucket="test", client=Client())
        assert accept(cache, policy)["status"] == "accepted"
        assert len(downloads) == len(manifest["files"])
        restore(cache, policy, logger, bucket="test", client=Client())
        assert len(downloads) == len(manifest["files"]), (
            "A second run must reuse verified cached bytes"
        )
