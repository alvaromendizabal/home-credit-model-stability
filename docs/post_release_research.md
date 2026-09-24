# Post-release frontier research

The evaluated September 2026 release remains frozen. This document tracks later
research intended to close the gap to stronger public competition systems without
rewriting the historical release or converting exploratory development evidence into
a new independent-test claim.

## Current verified state

The frozen release remains a 90% tuned / 10% original LightGBM blend. Its accepted
development mean stability is 0.601899, its reserved weeks 73–91 local stability is
0.729674, and the recorded late Kaggle evaluation is 0.56062 public / 0.47429
private. These evaluation settings are not interchangeable.

A deterministic repository check,
`python scripts/review_post_release_frontier.py`, recomputes the current frontier
summary from committed CSV/JSON evidence.

## Frontier round 1 — categorical identity

[Notebook 13](../notebooks/13_categorical_identity_frontier.ipynb) independently
tested native LightGBM categorical identity and identity+frequency representations.
Folds 1–3 selected `blend_native`; the candidate was frozen before folds 4–5.
Confirmation then produced only +0.000062 mean stability versus the saved champion,
with one win and one loss, so the predeclared promotion gate failed.

**Decision:** reject the candidate, preserve the frozen release, and close this
direction.

## Frontier round 2 — previous-application occurrence histograms

The [histogram report](../reports/applprev_histogram/README.md) tests a mechanism
reported by a strong public solution: per-applicant counts/shares for low-cardinality
`applprev_1` categories. The vocabulary was learned only on weeks 0–32 and frozen.
Thirty-nine histogram features were added to the existing 700-feature snapshot.

The registered standalone `hist_augmented` model passed folds 1–3 but failed the
frozen folds 4–5 confirmation gate versus the saved champion: +0.000106 mean
stability and -0.000378 mean AUC. It was not promoted.

A 25% histogram / 75% saved-champion blend had already been part of the registered
candidate grid. After the standalone candidate failed, that blend was chosen
post-selection for an explicitly exploratory external transfer probe because it
improved stability on both confirmation folds while retaining nonnegative mean
confirmation AUC/Gini deltas. This is not valid internal promotion evidence.

Across all five development folds the 25% blend improves stability on 5/5 folds,
with mean deltas of +0.003966 stability, +0.000449 AUC and +0.000943 mean Gini.
One all-label histogram fit completed and portable private Kaggle overlay assets were
prepared. The committed evidence does not claim a new hidden-test score.

## Zero-fit ensemble audit

A follow-up used saved predictions only and evaluated 47 champion/histogram/XGBoost/
CatBoost combinations with zero new model fits. The strongest descriptive candidate
remained exactly 75% saved champion / 25% histogram, with zero weight on XGBoost and
CatBoost. A prequential diagnostic remained positive at +0.004193 stability,
+0.000447 AUC and +0.000956 Gini, but base models were themselves selected on
development data, so this is not nested unbiased validation.

## Frontier round 3 — DenseLight

The [DenseLight contract](../reports/denselight_frontier/README.md) was frozen before
any DenseLight result existed. It independently recreates the DenseLight tabular
neural mechanism documented by leading public work and measures complementarity to
the tree champion.

Selection uses weeks 33–56 (three fits). Only a passing candidate proceeds to weeks
57–72 (two confirmation fits). A single all-label refit is allowed only after
confirmation. Weeks 73–91 are prohibited for selection. The maximum substantive
budget is six fits and there is no artificial wall-clock cutoff.

The readiness record verifies the 700-feature snapshot and prior-probe identities.
At publication time the inspected CPU environment had no CUDA/Torch/LightAutoML
available, so the contract remains `registered_not_executed` in committed evidence.

## Promotion discipline

Post-release candidates remain development research unless a separately registered
procedure and genuinely independent evaluation population justify a new release
claim. The observed September holdout cannot be reused as a fresh untouched test.
An after-deadline Kaggle hidden-test measurement is recorded separately as an
external transfer probe; it does not retroactively make post-selection development
choices internally confirmed.

## Public solution sources

- 59th-place solution:
  https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508202
- First-place solution:
  https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337
- LightAutoML DenseLight implementation:
  https://github.com/sb-ai-lab/LightAutoML
- Final leaderboard:
  https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/leaderboard
