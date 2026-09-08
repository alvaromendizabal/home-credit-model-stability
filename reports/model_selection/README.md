# Development model selection: accepted evidence

Open [notebook 08](../../notebooks/08_model_selection.ipynb) first. It compares all
15 fixed candidates, displays the selected candidate's fold deltas and shows
interactive Plotly charts with static GitHub fallbacks. Download and open
[report.html](report.html) for a self-contained interactive review.

## Result and limitation

The mean official fold stability objective selects 90% tuned LightGBM and 10%
original LightGBM: **0.6018988903**, compared with **0.6012378499** for the tuned
single model. Four folds improve, but worst-fold stability falls from **0.4791999815**
to **0.4725135463**. The gain of **0.0006610403** is small and has not been confirmed
on the locked holdout. No claim of statistical significance or leaderboard standing
is made. Weights chosen using earlier folds alone select the tuned single model for
folds 2-4 and the 90/10 blend for fold 5; base-model tuning is not nested.

The evaluation population contains **727,187 development cases**, weeks 33-72, with
five locked expanding folds. Week 68 has only 825 cases. The primary objective is
mean fold weekly-Gini stability, not pooled OOF stability. The report also shows
worst fold, ROC AUC, average precision, Brier score and log loss.

## Provenance and execution

`selection.json` is the exact 40,389-byte aggregate S3 publication. Its SHA-256 is:

```
be4b6b7a7468755312128e94aacbaf73a71909350066b5a6ad0a12b141e6c97e
```

The account-neutral [review policy](../../configs/model_selection_review.json)
records the immutable S3 key and source commit. The file itself records all scoring
source/configuration/lock digests, library versions, candidate metrics, 75 fold
records and aggregate diagnostics. It contains no borrower identifiers or individual
predictions. `publication.json` describes the separately reproducible notebook/HTML/SVG
render, without changing the original experiment evidence.

Managed processing job `home-credit-model-selection-20260908-0113` completed at
`6623d00e9118f9847d69c0caaedd71b10e9b32fa`. It passed 287 tests, Ruff and strict mypy,
restored the completed 15-candidate ledger, verified four prediction objects and
alignment, rebuilt diagnostics, executed notebook 08 and published verified S3
artifacts. It reused completed candidates: it did not perform 15 new experiments.
The managed launcher completed successfully in 223 seconds; no Home Credit model
was fitted and weeks 73-91 were not read.

## Reproduce without AWS or training

```bash
uv run --locked python scripts/review_model_selection.py --force
uv run --locked python scripts/review_model_selection.py
```

The first invocation executes all four code cells; the second verifies their receipt
and reuses the successful notebook. Both retain the canonical filename. Publication
fails on changed evidence, changed scoring lineage, inconsistent weights/metrics,
holdout contamination or incomplete population support. Unexpected cell errors
preserve the previous canonical notebook. CI checks this review under the locked
environment and retains the executed outputs.

Calibration fitting, final holdout evaluation, final model refit, train/test feature
parity and competition inference remain separate, unfinished release work. No CSV
submission is created here. The final submission must be generated, validated,
saved and downloaded by the owner inside the final notebook, never automatically
uploaded to Kaggle.
