#!/usr/bin/env python3
"""Compare current features with frozen inference without altering the accepted bundle.

Requires the original native bundle, raw manifest and public test files locally.
The candidate process builds features only. A separate frozen-source process
verifies the original bundle, compares features and scores both matrices with the
saved models. It never fits, replaces a manifest, submits, or relaxes load_bundle.
Only the aggregate receipt is suitable for publication; work files remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root: Path, bundle: Path, raw: Path, manifest: Path) -> dict[str, str]:
    """Check the immutable release and public input identities before importing source."""
    state = json.loads((root / "reports/model_release/state.json").read_text())
    protocol = json.loads((root / "configs/validation_protocol.json").read_text())
    expected = {str(bundle / "bundle.json"): state["stages"]["bundle"]["manifest"]["sha256"]}
    expected[str(manifest)] = protocol["data_lock"]["manifest_sha256"]
    for filename, sha in expected.items():
        if digest(Path(filename)) != sha:
            raise ValueError("release or raw manifest identity changed")
    for member in json.loads((bundle / "bundle.json").read_text())["files"]:
        path = (bundle / member["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or path.stat().st_size != member["bytes"]:
            raise ValueError("invalid frozen bundle member")
        expected[str(path)] = member["sha256"]
    for line in manifest.read_text().splitlines():
        member = json.loads(line)
        name = Path(member["file"]).name
        if name.startswith("test_") or name == "sample_submission.csv":
            path = raw / name
            if path.stat().st_size != member["bytes"]:
                raise ValueError("raw input size changed")
            expected[str(path)] = member["sha256"]
    for filename, sha in expected.items():
        if digest(Path(filename)) != sha:
            raise ValueError("frozen bundle member or public input changed")
    return expected


def worker(args: argparse.Namespace) -> None:
    """Run in a fresh interpreter; baseline imports only its verified frozen source."""
    import numpy as np
    import pandas as pd
    import polars as pl
    from polars.testing import assert_frame_equal

    import home_credit.modeling.inference as inference
    from home_credit.modeling.data import FeatureRef
    from home_credit.modeling.release import predict_components, transform

    source = args.bundle / "source" if args.phase == "baseline" else args.root / "src"
    if not Path(inference.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("wrong inference implementation imported")
    manifest = json.loads((args.bundle / "bundle.json").read_text())
    if args.phase == "baseline":
        manifest, features = inference.load_bundle(
            args.bundle, expected_sha256=digest(args.bundle / "bundle.json")
        )
    else:
        features = tuple(
            FeatureRef(**item)
            for item in json.loads((args.bundle / "features.json").read_text())["features"]
        )
    frame = inference.raw_test_frame(
        args.raw,
        inference.local_test_records(args.raw),
        features,
        json.loads((args.bundle / "raw_schema.json").read_text()),
        args.bundle / "feature_recipe.json",
    )
    candidate_path = args.work / "candidate_features.parquet"
    if args.phase == "candidate":
        frame.write_parquet(candidate_path)
        return
    candidate = pl.read_parquet(candidate_path)
    assert_frame_equal(frame, candidate, check_exact=True)
    encoder = json.loads((args.bundle / "encoder.json").read_text())
    matrices = [transform(value, features, encoder) for value in (frame, candidate)]
    np.testing.assert_array_equal(matrices[0], matrices[1])
    paths = {name: args.bundle / item["path"] for name, item in manifest["models"].items()}
    predictions = [predict_components(m, features, paths, manifest["weights"]) for m in matrices]
    np.testing.assert_array_equal(predictions[0], predictions[1])
    sample = pd.read_csv(args.raw / "sample_submission.csv")
    payloads = [
        inference.submission_frame(
            sample, pd.DataFrame({"case_id": frame["case_id"].to_numpy(), "score": values})
        )
        .to_csv(index=False, float_format="%.17g")
        .encode("utf-8")
        for values in predictions
    ]
    accepted = json.loads((args.root / "reports/kaggle_submission/execution.json").read_text())
    csv_sha = hashlib.sha256(payloads[0]).hexdigest()
    if payloads[0] != payloads[1] or csv_sha != accepted["kaggle_saved_run"]["csv_sha256"]:
        raise ValueError("public prediction bytes differ from accepted saved run")
    (args.work / "comparison.json").write_text(
        json.dumps(
            {
                "rows": frame.height,
                "features": len(features),
                "raw_test_shards": len(inference.local_test_records(args.raw)),
                "features_exact": True,
                "encoded_matrices_exact": True,
                "predictions_exact": True,
                "prediction_max_absolute_error": float(
                    np.max(np.abs(predictions[0] - predictions[1]))
                ),
                "csv_sha256": csv_sha,
                "matches_accepted_public_example": True,
                "model_sha256": {name: digest(path) for name, path in paths.items()},
            },
            indent=2,
        )
        + "\n"
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    from home_credit.observability.logging import RunLogger
    from home_credit.observability.runtime import StageTimer

    logger = RunLogger("date-parsing-compatibility", args.root / "logs")
    started = datetime.now(UTC).isoformat()
    args.work.mkdir(parents=True, exist_ok=True)
    before = verify_inputs(args.root, args.bundle, args.raw, args.manifest)
    builder = args.root / "src/home_credit/features/builder.py"
    source_sha = digest(builder)
    for phase in ("candidate", "baseline"):
        command = [
            sys.executable,
            "-I",
            "-W",
            "error",
            str(Path(__file__).resolve()),
            "--bundle",
            str(args.bundle),
            "--raw",
            str(args.raw),
            "--manifest",
            str(args.manifest),
            "--work",
            str(args.work),
            "--phase",
            phase,
        ]
        with (
            StageTimer(logger, phase, heartbeat_seconds=15),
            (args.work / f"{phase}.log").open("w") as output,
        ):
            subprocess.run(command, check=True, stdout=output, stderr=output, timeout=180)
        if phase == "candidate" and "warning" in (args.work / "candidate.log").read_text().lower():
            raise ValueError("candidate emitted a warning; inspect candidate.log")
    if before != verify_inputs(args.root, args.bundle, args.raw, args.manifest):
        raise ValueError("verified inputs changed during comparison")
    if digest(builder) != source_sha:
        raise ValueError("candidate builder changed during comparison")
    report: dict[str, Any] = json.loads((args.work / "comparison.json").read_text())
    report.update(
        schema_version=1,
        scope="public_example_compatibility_comparison",
        started_utc=started,
        completed_utc=datetime.now(UTC).isoformat(),
        python=sys.version.split()[0],
        candidate_builder_sha256=source_sha,
        verifier_sha256=digest(Path(__file__)),
        frozen_bundle_sha256=digest(args.bundle / "bundle.json"),
        raw_manifest_sha256=digest(args.manifest),
        verified_files=len(before),
        candidate_warnings=0,
        model_fits=0,
        kaggle_submitted=False,
        hidden_test_executed=False,
        frozen_bundle_modified=False,
    )
    (args.work / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "raw", "manifest", "work"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--phase", choices=("candidate", "baseline"))
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[1]
    for name in ("bundle", "raw", "manifest", "work"):
        setattr(args, name, getattr(args, name).resolve())
    if args.phase:
        source = args.bundle / "source" if args.phase == "baseline" else args.root / "src"
        sys.path.insert(0, str(source))
        worker(args)
    else:
        print(json.dumps(verify(args), indent=2))


if __name__ == "__main__":
    main()
