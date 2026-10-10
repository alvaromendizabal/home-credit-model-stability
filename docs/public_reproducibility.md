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

## Interactive demo

The [browser demo](https://home-credit-stability-lab.tartmacaw2.chatgpt.site) is a buildless, local-first explanation of
temporal model evaluation. It uses a deterministic synthetic weekly curve, not
borrower-level records or trained-model predictions. All controls and calculations
run in the browser without a service, API key or analytics request.

Pin the current scenario as a baseline, then change the candidate's decline or
residual variation. Both use the same 24 ordinal weeks, mean Gini of 0.6 and
deterministic residual pattern. The chart overlays the two curves, while the
comparison shows candidate-minus-baseline stability and its additive mean, trend
and variation components. These differences are arithmetic explanations, not
evidence of a better lending model.

Replace or clear the baseline explicitly. Reset restores the pinned baseline, or
the steady preset when none is pinned. With a baseline pinned, CSV exports paired
weekly rows and JSON also includes component differences and consistency checks.
Without a baseline, exports retain the single synthetic fixture.

To serve the demo from a clone:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory demo
```

Open http://127.0.0.1:8000. The numerical checks and actual UI-handler tests require only Node.js:

```bash
node demo/model.test.mjs
node demo/app.test.mjs
```

The synthetic explorer is not part of the observed evidence table. Its mathematical
behavior is checked independently from the published aggregate report. The [Public demo workflow](../.github/workflows/public-demo.yml) runs both checks.
The research CI separately preserves the broader notebook and source gates.

## What the evidence-review command verifies

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

## Dated owner-return receipt

The [E97 summary](../reports/latest_execution/summary.json) is a separate sanitized
operational receipt. Verify its fixed identity and evidence contract with:

```bash
python tools/summarize_execution.py --check
```

It records a memory-guard stop, 13 of 42 tasks complete and no new fit or submission.
Its original local archive integrity was verified; referenced remote artifacts and
private predictions were not independently replayed. It does not update the frozen
metrics or reverify the previously reported Kaggle scores. The [current status](project_status.md)
explains the distinction. The source owner-return archive remains private.

## Later AWS execution snapshot

A separate [E98 snapshot](../reports/latest_execution/aws_snapshot.json) records a
read-only inspection of the later completed run: 42 tasks and five fits, with the
selected candidate rejected on later-period stability. It records no new submission
or score. The snapshot preserves aggregate decisions and source identities; it does
not distribute the private model, configuration or predictions. Its observation time
is not the run's completion time. [Current status](project_status.md) explains the
selection/confirmation distinction and keeps the E97 archive separate. Check this
public snapshot without cloud access:

```bash
python tools/summarize_aws_snapshot.py --check
```

This validates the committed aggregate contract. Regenerating it from the original
inspected receipt bytes additionally requires private evidence retained outside Git.

## Reproducibility matrix

| Layer | Available publicly | What can be verified |
|---|---|---|
| Interactive metric demo | Static HTML/CSS/JavaScript and deterministic synthetic fixtures | Metric behavior and interaction; no trained-model or lending-outcome claim |
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
