# Reproducing the public evidence

The public review can be run from a clone with Python 3.12 and no third-party
packages, cloud credentials, borrower data or training job:

```bash
python3 scripts/review_public_evidence.py --output artifacts/public-review
```

Open the resulting `index.html` locally. `review.json` contains the same review in
machine-readable form. Output is deterministic: repeated runs over the same verified
inputs produce the same bytes, with no machine-specific path or current timestamp.
The canonical example is [reports/public_review](../reports/public_review).

## What that command verifies

The script consumes only previously published aggregate reports:

- benchmark acceptance evidence and metric corrections;
- controlled feature-family ablation results;
- the frozen future-period evaluation summary, release intent and weekly discrimination values.

It checks pinned input hashes before calculation and validates the expected report
structure, population counts and comparison alignment. It recalculates the weekly-Gini
stability components and comparison summaries where the public aggregates support
those calculations. Changed or inconsistent inputs cause a nonzero exit.

A published AUC, average precision, Brier score or log loss is not independently
recomputed without the underlying labels and predictions. The report distinguishes
recomputed aggregate statistics from source-reported metrics. Checksums establish
which published bytes were reviewed; they do not independently establish the truth
of the original private experiment.

## Reproducibility matrix

| Layer | Available publicly | What can be verified |
|---|---|---|
| Lightweight evidence review | Standard-library script, aggregate input reports, deterministic JSON/HTML | Input integrity, supported aggregate calculations and consistent evaluation scope |
| Executed research reviews | Python source, lockfile, notebooks and published outputs | Full review execution, lint/type checks, tests and byte-for-byte output reproduction |
| Temporal methodology | Fold definitions, historical release policy and decision records | Selection/evaluation separation and the documented use of confirmation periods |
| Historical model research | Benchmark, feature removals, calibration and later aggregate outcomes | Published comparisons and both acceptance and rejection decisions |
| Raw training and full inference | Historical implementation and artifact contracts | Implementation inspection; exact replay additionally requires restricted data and model artifacts |
| Active research | Mechanism-level summaries after reconciliation | Published aggregate findings; exact active recipes and checkpoints are not distributed |

## Full notebook and source checks

The canonical locked environment uses Python 3.12.14:

```bash
uv sync --locked --group dev
uv run --locked python scripts/review_model_benchmark.py --force
uv run --locked python scripts/review_model_release.py --force
bash scripts/check.sh
```

The [CI workflow](../.github/workflows/ci.yml) specifies the complete notebook
execution and reproduction sequence. It runs review generation before quality gates,
checks the exact proposed commit and compares canonical output bytes. Notebook
execution reviews saved aggregate evidence; it does not start private training.

To reproduce the lightweight checked-in report directly:

```bash
python3 scripts/review_public_evidence.py --output reports/public_review
git diff --exit-code -- reports/public_review
```

## Disclosure boundary

The public repository retains previously published historical source and evidence.
The public review adds only explicitly selected aggregate results. It does not load,
export or redistribute:

- borrower-level source records, labels or prediction arrays;
- private feature matrices or intermediate caches;
- trained research checkpoints, optimizer state or active feature identities;
- current ensemble specifications or runtime dependency bundles;
- owner execution returns, internal handoff prompts or cloud credentials.

Those artifacts remain in the research environment. This boundary allows code and
scientific review while protecting restricted data and unpublished implementation
details. Historical parameters already present in Git remain historical evidence;
this update does not rewrite repository history or assert that old disclosures have
been erased.

## Interpretation limits

The original future-period evaluation is now observed and belongs to its frozen
release. Later research uses development periods and is labeled accordingly. A
successful report build verifies the public review, not a new trained model or a
production deployment. Lending-policy utility, protected-group fairness and an
operational monitoring service remain outside the validated scope.

For a short tour, read the [employer review guide](employer_review.md),
[case study](case_study.md) and [model card](../MODEL_CARD.md).
