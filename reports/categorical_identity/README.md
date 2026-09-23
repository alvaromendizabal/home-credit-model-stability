# Categorical identity frontier — accepted negative evidence

This report records the **22 September 2026 post-release research experiment** that tested whether preserving categorical identity in LightGBM could improve the frozen development system. It is deliberately kept separate from the evaluated release: the experiment reused previously explored development weeks, did not read new independent labels, did not alter the frozen model, and did not create a Kaggle submission.

## Decision

**Rejected. Do not refit or submit this candidate.** Folds 1–3 selected `blend_native`, but the candidate failed the predeclared confirmation gate on folds 4–5 versus the saved 90/10 LightGBM champion.

| Confirmation comparison | Mean stability delta | Wins | Worst fold delta | Gate |
|---|---:|---:|---:|---|
| vs saved champion | +0.000062 | 1/2 | -0.002671 | **FAIL** |
| vs matched frequency control | +0.008452 | 1/2 | -0.002553 | PASS |

The descriptive moving-block bootstrap interval for the confirmation delta versus the saved champion was **[-0.013200, +0.036385]**. This is not an independent-test confidence interval.

## Experimental design

- Frozen 700-feature snapshot, including 67 categorical features.
- Five expanding temporal development folds.
- Folds 1–3 used for candidate selection; the selected candidate was frozen before folds 4–5.
- Frequency-only, native categorical identity, identity+frequency, and fixed blends were compared under a matched tree schedule.
- Saved champion predictions were reused instead of retraining the release.
- All raw public-fixture feature and prediction parity checks passed before training.
- The run completed **13 substantive fits** in **2.53 hours** and preserved checkpoints throughout.

## Why this negative result matters

The selection phase suggested that categorical identity might improve the official stability objective, but confirmation showed essentially no mean gain against the saved champion and a losing fold. Standalone native-category models also sacrificed ranking quality in several periods. The experiment therefore closes a plausible mechanism rather than treating an unstable development improvement as progress.

This is the intended research discipline for the repository: promising hypotheses advance only when they survive a frozen later-window gate.

## Reproducible evidence

- [Executed notebook](../../notebooks/13_categorical_identity_frontier.ipynb)
- [`fold_metrics.csv`](fold_metrics.csv) — all model/fold metrics
- [`selection_decision.json`](selection_decision.json) — frozen folds 1–3 selection decision
- [`confirmation_decision.json`](confirmation_decision.json) — folds 4–5 promotion gate
- [`conditional_uncertainty.json`](conditional_uncertainty.json) — descriptive paired moving-block bootstrap
- [`run_manifest.json`](run_manifest.json) — source/environment/run lineage and stage receipts
- [`reproduction_matrix.csv`](reproduction_matrix.csv) — leading-solution mechanism coverage

The original frozen holdout and the late Kaggle score remain separate evidence. This post-release experiment does not change either one.
