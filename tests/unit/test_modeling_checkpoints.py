from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from home_credit.modeling.checkpoints import (
    build_checkpoint_manifest,
    derive_run_key,
    load_manifest_bytes,
    manifest_bytes,
    object_key_for_sha,
    parse_s3_uri,
    sha256_file,
    validate_benchmark_state,
    verify_checkpoint_manifest,
)


def _identity() -> dict[str, object]:
    return {
        "git_commit": "git",
        "feature_manifest_sha256": "feature",
        "validation_protocol_sha256": "protocol",
        "benchmark_config_sha256": "config",
        "feature_screen_sha256": "",
        "smoke": False,
    }


def _write_state(root: Path) -> None:
    screen = root / "feature_screen.json"
    screen.write_text('{"screen": true}\n', encoding="utf-8")
    identity = _identity()
    identity["feature_screen_sha256"] = sha256_file(screen)

    model = root / "models" / "lightgbm" / "fold_1.txt"
    prediction = root / "predictions" / "lightgbm" / "fold_1.parquet"
    model.parent.mkdir(parents=True)
    prediction.parent.mkdir(parents=True)
    model.write_text("model\n", encoding="utf-8")
    prediction.write_bytes(b"prediction")

    state = {
        "schema_version": 1,
        "identity": identity,
        "folds": {
            "lightgbm:fold_1": {
                "model": "lightgbm",
                "fold": 1,
                "model_path": "models/lightgbm/fold_1.txt",
                "model_sha256": sha256_file(model),
                "prediction_path": "predictions/lightgbm/fold_1.parquet",
                "prediction_sha256": sha256_file(prediction),
                "metrics": {"stability_score": 0.5},
            }
        },
    }
    (root / "benchmark_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_run_key_is_deterministic_and_provenance_bound() -> None:
    first = derive_run_key(
        git_commit="a",
        feature_manifest_sha256="b",
        validation_protocol_sha256="c",
        benchmark_config_sha256="d",
        smoke=False,
    )
    second = derive_run_key(
        git_commit="a",
        feature_manifest_sha256="b",
        validation_protocol_sha256="c",
        benchmark_config_sha256="d",
        smoke=False,
    )
    changed = derive_run_key(
        git_commit="a",
        feature_manifest_sha256="b",
        validation_protocol_sha256="c",
        benchmark_config_sha256="different",
        smoke=False,
    )

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_parse_s3_uri_and_content_addressed_key() -> None:
    bucket, prefix = parse_s3_uri("s3://example-bucket/a/b/")
    digest = hashlib.sha256(b"x").hexdigest()

    assert bucket == "example-bucket"
    assert prefix == "a/b"
    assert object_key_for_sha(prefix, digest) == (f"a/b/objects/{digest[:2]}/{digest}")

    with pytest.raises(ValueError, match="invalid S3 URI"):
        parse_s3_uri("/tmp/not-s3")


def test_validate_benchmark_state_verifies_referenced_hashes(tmp_path: Path) -> None:
    _write_state(tmp_path)

    count = validate_benchmark_state(
        tmp_path,
        git_commit="git",
        feature_manifest_sha256="feature",
        validation_protocol_sha256="protocol",
        benchmark_config_sha256="config",
        smoke=False,
    )

    assert count == 1

    model = tmp_path / "models" / "lightgbm" / "fold_1.txt"
    model.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_benchmark_state(
            tmp_path,
            git_commit="git",
            feature_manifest_sha256="feature",
            validation_protocol_sha256="protocol",
            benchmark_config_sha256="config",
            smoke=False,
        )


def test_checkpoint_manifest_roundtrip_and_corruption_detection(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)

    manifest = build_checkpoint_manifest(
        tmp_path,
        run_key="run",
        prefix="checkpoints/run",
        sequence=1,
        completed_model_folds=1,
        git_commit="git",
        feature_manifest_sha256="feature",
        validation_protocol_sha256="protocol",
        benchmark_config_sha256="config",
        smoke=False,
    )

    restored = load_manifest_bytes(manifest_bytes(manifest))

    assert restored.run_key == "run"
    assert restored.completed_model_folds == 1
    assert [item.path for item in restored.files] == sorted(item.path for item in restored.files)

    verify_checkpoint_manifest(tmp_path, restored)

    target = tmp_path / restored.files[0].path
    target.write_bytes(target.read_bytes() + b"x")

    with pytest.raises(ValueError, match="checkpoint file"):
        verify_checkpoint_manifest(tmp_path, restored)
