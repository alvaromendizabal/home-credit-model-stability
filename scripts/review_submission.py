#!/usr/bin/env python3
"""Execute the owner-controlled submission notebook with export disabled."""

from __future__ import annotations

import argparse
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
    It never uploads to Kaggle. **Submission generation is off by default.**

    Set the bundle, raw test and sample-submission paths below. When you are ready,
    change `GENERATE_SUBMISSION` to `True` and run all cells. The output is validated,
    saved with a lineage receipt and presented as a download link.

    On Kaggle, point `RAW_TEST_DIRECTORY` to that run's competition input and attach
    the verified model bundle as an input dataset. The ten downloadable public example
    cases are only a local integration fixture; they are not the hidden evaluation set.
    """,
    ),
    (
        "code",
        """
import os
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
GENERATE_SUBMISSION = False

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
if GENERATE_SUBMISSION or verify_raw:
    import sys

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
    output hashes. Your notebook environment must use persistent storage for this path.
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
