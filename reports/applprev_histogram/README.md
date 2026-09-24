# Previous-application histogram frontier

This directory publishes compact evidence from the 23 September 2026 post-release
study of low-cardinality per-category occurrence histograms built from
`applprev_1`. It is **development research**, not a replacement for the frozen
September release.

## Experiment design

The vocabulary was fit only on cases from weeks 0–32, then frozen. Thirty-nine
count/share features were added to the existing 700-feature snapshot. One augmented
LightGBM was fit per temporal fold using the already-selected trial parameters and
fixed fold-specific iteration budgets. Folds 1–3 were the selection phase; folds
4–5 were held back for the registered confirmation decision.

The registered standalone `hist_augmented` model passed selection but **failed
confirmation versus the saved champion**: +0.000106 mean stability and -0.000378
mean AUC across folds 4–5. It was therefore not promoted.

## External transfer probe

A 25% histogram / 75% saved-champion blend had been included in the original
candidate grid and passed the folds 1–3 gate. After the standalone model failed
confirmation, the 25% blend was chosen post-selection for an explicitly exploratory
hidden-test transfer probe because it improved stability on both confirmation folds
and retained nonnegative mean AUC/Gini deltas. This is **not valid internal promotion
evidence**.

Across all five development folds, the 25% blend improved stability on 5/5 folds,
with mean deltas of +0.003966 stability, +0.000449 AUC and +0.000943 mean Gini
versus the saved champion. One all-label histogram fit was completed and the
portable private Kaggle overlay was prepared; the committed evidence does not claim
a new hidden-test score.

A zero-fit follow-up searched 47 champion/histogram/XGBoost/CatBoost combinations
using saved OOF predictions. The strongest descriptive candidate remained exactly
75% champion / 25% histogram, with zero weight on XGBoost and CatBoost.

## Files

- `fold_metrics.csv`: all 30 saved fold/model rows used for the published comparisons.
- `confirmation_decision.json`: frozen standalone confirmation result.
- `external_probe_decision.json`: explicit post-selection disclosure and probe policy.
- `../../configs/applprev_histogram_frontier.json`: registered experiment contract.

Run `python scripts/review_post_release_frontier.py` to recompute the compact
frontier summary from these committed artifacts.
