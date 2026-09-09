# Model card: frozen Home Credit research release

This project studies whether credit-risk ranking remains useful in later
application periods. It is an evaluated research pipeline with a reproducible
inference artifact. It has not been validated for real lending decisions.

## Model and data

| Item | Released specification |
|---|---|
| Task | Binary default-risk prediction using the competition's target definition |
| Source | Home Credit – Credit Risk Model Stability competition data |
| Representation | 700 selected features from 17 relational groups |
| Candidate selection | 2,508 original candidates; fit weeks 0–24, screen weeks 25–32 |
| Predictors | 90% tuned LightGBM, 10% original LightGBM |
| Fixed rounds | 1,852 tuned; 1,355 original |
| Probability calibration | None in the released model |
| External information | None |
| Reproducibility | Pinned Python 3.12.14 lock; native models, encoders, hashes and source identities |

The [competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
defines the target and data scope. Case-local histories become numeric aggregates,
application-relative dates, recency counts, category diversity and missingness
measures. Categorical frequency maps fit only the corresponding training population.
Identifiers, target values and base absolute-time fields are excluded as predictors.

The [frozen policy](configs/model_release.json) and
[release record](reports/model_release/state.json) identify the actual specification.

## Evaluation and artifact distinction

The development comparison uses five expanding temporal folds and 727,187 aligned
out-of-fold cases. It includes LightGBM, XGBoost, CatBoost and a logistic SGD
baseline, followed by controlled feature-family removals, bounded LightGBM tuning
and a fixed blend comparison. Selection reuses the same development periods;
development scores are not independent generalization estimates.

The frozen blend was then trained on **1,323,314 applications from weeks 0–72**
and evaluated on **203,345 applications from weeks 73–91**. Features, weights,
calibration choice and training rounds were fixed before those labels were opened.
The evaluation did not use holdout early stopping.

| Held-out-period metric | Observed result |
|---|---:|
| Weekly Gini stability | 0.729674 |
| ROC AUC | 0.875759 |
| Average precision | 0.193755 |
| Brier score | 0.019282 |
| Log loss | 0.080695 |
| Observed default prevalence | 2.1879% |

The official metric penalizes declining weekly discrimination and residual
variation. AUC, average precision and probability diagnostics answer different
questions. Changes in prevalence and period length prevent direct comparisons
between this score and development means. No statistical-significance or
state-of-the-art claim follows from the small development blend improvement.

A **separate all-label refit** uses 1,526,659 applications from weeks 0–91 for
inference. The table above evaluates the development-trained models, not that
all-label artifact. The holdout is now observed and cannot serve as a new
untouched test for subsequent work.

## Known limitations that affect interpretation

- The original selected features include sex and birth-related fields. These are
  explicit inputs, not an audited fairness control. Feature importance describes
  predictive association and does not justify a lending decision or a causal claim.
- Competition snapshots define availability. Source group indices do not prove
  chronological order, and this is not production event-time lineage certification.
- Temporal shifts are substantial. A high pooled score can hide differences across
  weeks, subpopulations and probability ranges; the notebooks expose weekly results.
- No operational decision threshold, lending-cost function, policy impact or
  protected-group fairness evaluation has been validated.
- The saved Kaggle notebook reproduced the verified ten-example integration export
  offline. Its submitted hidden-test execution is running; no leaderboard result
  or completed hidden-test resource measurement is claimed yet. See the
  [submission record](reports/kaggle_submission/execution.json).

Application periods define the folds; production event-time availability and label
maturity have not been certified.

The later [feature research](notebooks/11_feature_research.ipynb) tested 4,617 new
hypotheses and completed 20 comparison fits. Its engineered condition improved
pooled AUC while reducing temporal stability. It was not promoted. A separate
raw-history study tested 524 median/IQR/p90/skew candidates, retained 96 and
completed ten full comparisons. The 796-feature condition reaches 0.584195 mean
stability (-0.000994 vs the original control); removing its 13 skew features
reaches 0.588678 (+0.003490).

Both history conditions improve the weakest fold but win on only three of five
folds. Their fold-omission ranges cross zero; without skew, the range is
-0.002966 to +0.007128. The modest mean benefit depends on which periods
are included. Together with the weaker engineered extension and the fragile
1,400-feature gain, this supports closing the registered search without further
ad hoc expansion or retuning. It does not establish exhaustive discovery of every
possible feature. Any later promotion requires its own protocol and new independent
evaluation evidence.

All 1,454,374 predictions were replayed from saved native models. These are
post-release development results, not a new independent evaluation; they do
not alter the released recipe. The history view has no raw-test feature contract.
The accompanying audit records 33 date fields across 14 sources and explicitly
excludes chronological features without a verified availability contract.

The [calibration research](notebooks/12_calibration.ipynb) uses development
data only. Neither tested map improved pooled Brier or log loss; it does not
change this frozen release. Further research requires its own promotion decision
and independent evaluation population.

## Reproduction and inference

[Notebook 09](notebooks/09_model_release.ipynb) reproduces the release evidence
without private data or model fitting. Independent verification checked 62 objects,
recomputed all eight saved holdout metrics, reloaded the native bundle and verified
raw-feature parity and unchanged prediction-batch reuse.

[Notebook 10](notebooks/10_submission.ipynb) rebuilds features from the supplied
raw test files and validates the model bundle, exact case coverage, order and
finite probabilities. CSV generation is off by default. The owner enables and
downloads a file explicitly; there is no automatic Kaggle upload.

See the [release runbook](docs/model_release.md) for artifact lineage and commands,
and [the project status](PROJECT_PLAN.md) for the completed research record.
