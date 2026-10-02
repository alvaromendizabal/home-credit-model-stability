# October 2026 post-release frontier

This report records the post-release work completed after the 23 September publication.
It is intentionally **employer-facing and aggregate-only**: validation design, model
families, fit counts and observed metrics are published, while borrower-level matrices,
private checkpoints, runtime bundles, exact recovered-feature identities and active
competition recipes remain outside Git.

The evaluated September release remains frozen. Weeks 73-91 are already observed and
are not reused as a fresh model-selection population.

## External transfer result

A later ensemble combined the frozen tree champion, the previous-application histogram
challenger and a multi-seed DenseLight neural component. Its five-fold development
mean stability was **0.618940**, compared with **0.601899** for the frozen September
release.

The candidate was then measured through the competition's late-submission path:

| Submission | Public | Private |
|---|---:|---:|
| Frozen September inference refit | 0.56062 | 0.47429 |
| Later tree + histogram + DenseLight ensemble | 0.56035 | **0.47652** |

The later candidate improved the private score by **0.00223** while changing the
public score by -0.00027. This is useful external-transfer evidence, but it does not
turn the previously observed local holdout into a new independent test.

## DenseLight frontier

The preregistered DenseLight direction was executed on GPU and then extended to a
three-seed bagged neural component. The final internal candidate used a robust
probability aggregation across seeds and passed the locked folds 4-5 confirmation
gate before the all-label inference artifact was built.

Engineering evidence included:

- fixed expanding temporal folds with weeks 73-91 excluded from selection;
- GPU inference benchmarking on an NVIDIA L4;
- deterministic checkpoint and preprocessing identities;
- CPU/GPU probability parity below 3e-8 maximum absolute error;
- resumable training and hash-verified model artifacts.

The later Kaggle result above shows that the large internal stability gain transferred
only modestly to the private leaderboard. The exact DenseLight architecture/weights
and private checkpoints remain runtime artifacts rather than public competition code.

## Enriched tree frontier

A subsequent controlled tree study added a source-aware feature expansion and compared
fresh LightGBM/CatBoost variants under the same temporal protocol.

The unweighted expanded LightGBM improved over its same-family control on the
selection folds by:

- **+0.009229 mean stability**;
- **+0.001354 mean AUC**.

A recency-weighted/stability-filtered variant reduced mean stability by **0.014187**
relative to the unweighted expansion, so that weighting direction was closed.

The selected blend passed folds 1-3 but failed the frozen folds 4-5 confirmation
gate. All 14 planned fits completed and no model was promoted.

## Categorical and sparse ceiling-escape study

The next bounded study independently recreated several public-solution mechanisms,
including a larger categorical representation, depth-two payment-history summaries,
a competition risk-assessment field and an early-screened sparse-feature family.

The direct payment/risk feature branch underperformed the prior expanded LightGBM:

- **-0.031755 mean stability**;
- **-0.002055 mean AUC**.

The sparse recovery branch was the strongest new signal. Relative to that direct
LightGBM it improved selection by:

- **+0.037661 mean stability**;
- **+0.001951 mean AUC**.

Recovered categorical representation also materially improved CatBoost relative to
the earlier CatBoost representation, but neither the selected blend nor the
unweighted expanded-LightGBM rescue passed the locked confirmation gate.

The selected sparse-enhanced blend reached **+0.011655 mean stability** with 3/3
selection wins, then failed confirmation with **-0.000577 mean stability** and 1/2
wins. It was correctly rejected. All 14 substantive fits completed.

## Current frontier

The next registered research direction is deliberately different from another
near-duplicate tree sweep. It combines four missing capabilities in one shared
experiment:

1. temporally stable sparse-feature recovery;
2. explicit max-minus-min range features inspired by strong public aggregation
   strategies;
3. a fresh XGBoost challenger on the sparse/range representation;
4. recovered categorical features for CatBoost without the rejected direct-feature
   branch.

This frontier is **planned, not yet reported as executed** in this publication.

## Reproducibility boundary

The public repository preserves:

- experiment decisions;
- validation boundaries;
- aggregate fold-level outcomes;
- rejection/promotion decisions;
- model-family and representation descriptions;
- reproducibility contracts.

Private AWS runtime artifacts preserve:

- raw competition matrices;
- exact recovered feature lists;
- model checkpoints;
- portable Kaggle runtime bundles;
- active frontier implementation details.

This split keeps the portfolio reviewable and scientifically auditable without
publishing the full competition recipe.
