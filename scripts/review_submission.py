#!/usr/bin/env python3
"""Execute the submission notebook in review mode; Kaggle runs offline inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from home_credit.modeling.checkpoints import atomic_write
from home_credit.modeling.portfolio_report import write_notebook
from home_credit.modeling.release_workflow import ReleaseLogger
from home_credit.runtime.notebooks import execute_notebook

CELLS = [
    (
        "markdown",
        """
    # Generate, validate and download your submission

    The frozen native bundle has been fitted on all 1,526,659 labeled applications.
    This notebook runs the same raw feature transformations and encoders as training.
    On Kaggle, attach the **Home Credit Frozen Inference** dataset and the competition
    data, select CPU with **Internet off**, then Save & Run All. This notebook discovers
    the attached inputs, installs hash-locked wheels into an isolated offline environment,
    and writes `/kaggle/working/submission.csv` plus a lineage receipt. Submit the saved
    notebook version through Kaggle so it reruns on the hidden test data.

    Outside Kaggle, review mode remains the default. Local exports can be enabled below.
    The ten public example cases validate integration; they are not a leaderboard result.
    No training, feature selection or calibration occurs in this notebook.
    """,
    ),
    (
        "code",
        """
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from IPython.display import FileLink, display

root = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file()),
    Path.cwd(),
)
release_key = "80f9edcb4d995cfdcb7ea3841aaf1f67c6bb31c386dc75f7bd4c3e9168192c79"
BUNDLE_DIRECTORY = Path(os.environ.get(
    "HOME_CREDIT_BUNDLE_DIRECTORY", str(root / "artifacts/model_release" / release_key / "bundle")
))
RAW_TEST_DIRECTORY = Path(os.environ.get(
    "HOME_CREDIT_RAW_TEST_DIRECTORY", str(root / "data/raw/parquet_files/test")
))
SAMPLE_SUBMISSION = root / "data/raw/sample_submission.csv"
DESTINATION = root / "submissions/submission.csv"
ON_KAGGLE = Path("/kaggle/input").is_dir()
GENERATE_SUBMISSION = ON_KAGGLE
runtime_sha256 = "RUNTIME_MANIFEST_SHA256"
if ON_KAGGLE:
    competition_roots = [
        p.parent for p in Path("/kaggle/input").rglob("sample_submission.csv")
        if p.parent.name == "home-credit-credit-risk-model-stability"
    ]
    if len(competition_roots) != 1:
        raise ValueError("Attach the Home Credit competition data exactly once")
    RAW_TEST_DIRECTORY = competition_roots[0] / "parquet_files/test"
    SAMPLE_SUBMISSION = competition_roots[0] / "sample_submission.csv"
    DESTINATION = Path("/kaggle/working/submission.csv")

bundle_sha256 = "71f338a66b15d0f8b549c3a9c9defe4defdd4169f5f57be3c65036ac9bcbf84a"
print("Expected frozen bundle identity:", bundle_sha256)
print("CSV generation enabled:", GENERATE_SUBMISSION)
""",
    ),
    (
        "markdown",
        """
    ## Run with durable prediction batches

    Inputs are fingerprinted before scoring. Each completed batch has a hash receipt;
    rerunning unchanged inputs reuses it. Missing tables, duplicated IDs, changed schemas,
    altered code, corrupt models and nonfinite probabilities raise explicit errors.
    Logs include UTC timestamps, stage/total elapsed time and long-stage heartbeats.

    The automated release check can exercise raw inference with CSV generation still
    disabled. Review mode below requires no raw data or cloud account.
    """,
    ),
    (
        "code",
        """
verify_raw = os.environ.get("HOME_CREDIT_VERIFY_INFERENCE", "0") == "1"
if ON_KAGGLE:
    matches = [
        p for p in Path("/kaggle/input").rglob("runtime.json")
        if hashlib.sha256(p.read_bytes()).hexdigest() == runtime_sha256
    ]
    if len(matches) != 1:
        raise ValueError("Attach the verified Home Credit Frozen Inference dataset exactly once")
    assets = matches[0].parent
    manifest = json.loads(matches[0].read_text())
    runner = assets / "kaggle_inference.py"
    member = next(m for m in manifest["files"] if m["path"] == runner.name)
    if hashlib.sha256(runner.read_bytes()).hexdigest() != member["sha256"]:
        raise ValueError("Offline inference launcher digest mismatch")
    subprocess.run([
        sys.executable, "-I", str(runner), "--assets", str(assets),
        "--runtime-sha256", runtime_sha256, "--raw", str(RAW_TEST_DIRECTORY),
        "--sample", str(SAMPLE_SUBMISSION),
        "--work", str(Path("/kaggle/temp/home_credit") / runtime_sha256),
        "--output", str(DESTINATION.parent),
    ], check=True)
    receipt = json.loads(DESTINATION.with_suffix(".json").read_text())
    print("Validated submission:", receipt["rows"], "rows; SHA-256:", receipt["sha256"])
elif GENERATE_SUBMISSION or verify_raw:
    import pandas as pd

    source_directory = BUNDLE_DIRECTORY / "source"
    if not source_directory.is_dir():
        raise FileNotFoundError("Extract the verified bundle and set BUNDLE_DIRECTORY first")
    sys.path.insert(0, str(source_directory))
    from home_credit.modeling.checkpoints import canonical_json_bytes, sha256_bytes
    from home_credit.modeling.inference import export_submission, local_test_records, predict_raw
    from home_credit.modeling.release_workflow import ReleaseLogger

    predictions = predict_raw(
        BUNDLE_DIRECTORY,
        RAW_TEST_DIRECTORY,
        root / "artifacts/submission_predictions",
        expected_bundle_sha256=bundle_sha256,
        logger=ReleaseLogger("notebook-inference", root / "logs"),
    )
    print("Raw inference passed for", len(predictions), "cases.")
    if GENERATE_SUBMISSION:
        raw_records = local_test_records(RAW_TEST_DIRECTORY)
        input_sha256 = sha256_bytes(canonical_json_bytes({r.file: r.sha256 for r in raw_records}))
        export_submission(
            pd.read_csv(SAMPLE_SUBMISSION), predictions, DESTINATION,
            lineage={"bundle_sha256": bundle_sha256, "input_sha256": input_sha256},
            owner_confirmed=True,
        )
    print("Submission generated:", GENERATE_SUBMISSION)
else:
    print("Review mode: inference and export are disabled. Notebook 09 contains verified evidence.")
""",
    ),
    (
        "markdown",
        """
    ## Download after validation

    The exported CSV must have exactly `case_id,score`, one row per supplied sample
    case, the same case order as the sample, and finite probabilities in [0, 1].
    The saved file is read back and checked. A different existing CSV is preserved
    rather than silently overwritten. Its JSON receipt records the bundle, input and
    output hashes. Kaggle preserves these files in the saved notebook version.
    """,
    ),
    (
        "code",
        """
if GENERATE_SUBMISSION:
    display(FileLink(os.path.relpath(DESTINATION), result_html_prefix="Download validated CSV: "))
    display(FileLink(os.path.relpath(DESTINATION.with_suffix(".json")),
                     result_html_prefix="Download lineage receipt: "))
else:
    print("No submission CSV was generated. No Kaggle upload was performed.")
""",
    ),
]


_runtime_policy = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/kaggle_submission.json").read_text()
)
CELLS = [
    (kind, source.replace("RUNTIME_MANIFEST_SHA256", _runtime_policy["runtime_sha256"]))
    for kind, source in CELLS
]


def review(root: Path, *, force: bool = False, write_only: bool = False) -> bool:
    directory = root / "artifacts/submission_review"
    directory.mkdir(parents=True, exist_ok=True)
    notebook = directory / "10_submission.ipynb"
    write_notebook(notebook, CELLS)
    reused = False
    if not write_only:
        reused = execute_notebook(
            root,
            notebook,
            ReleaseLogger("submission-review", root / "logs"),
            force=force,
            dependencies=[
                Path(__file__),
                root / "uv.lock",
                root / "configs/portfolio_review.json",
                root / "src/home_credit/modeling/inference.py",
                root / "configs/kaggle_submission.json",
            ],
            receipt_path=directory / "receipt.json",
            execution_root=root,
        )
    atomic_write(root / "notebooks/10_submission.ipynb", notebook.read_bytes())
    return reused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    review(Path(__file__).resolve().parents[1], force=args.force, write_only=args.write_only)


if __name__ == "__main__":
    main()
