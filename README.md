# Home Credit Model Stability

An evaluated credit-risk ML system built around one question: **does predictive
performance survive a move into future application periods?** The project combines
relational feature engineering, temporal model comparison, controlled ablations,
resumable AWS execution and a portable inference pipeline.

## Review in five minutes

1. [Feature engineering](notebooks/02_feature_engineering.ipynb): 2,508 candidates,
   training-only screening, 700 retained features and measured feature-family value.
2. [Model comparison](notebooks/05_benchmark_review.ipynb): four model families on
   five expanding temporal folds.
3. [Frozen future-period evaluation](notebooks/09_model_release.ipynb): the final
   result, reliability, native model lineage and tested inference boundaries.

For the full experimental trail, read the [feature ablations](reports/feature_ablation/06_feature_ablation.ipynb),
[tuning](notebooks/07_model_tuning.ipynb) and [ensemble selection](notebooks/08_model_selection.ipynb).
The later [feature study](notebooks/11_feature_research.ipynb) tests a wider budget
and 256 engineered additions with SHAP, grouped permutation and redundancy diagnostics.
The [calibration study](notebooks/12_calibration.ipynb) tests whether probability
mappings transfer across development periods. Both retain their negative findings.
The notebooks embed readable tables, interactive Plotly figures and static GitHub
fallbacks. Reading the evidence requires no cloud account or borrower-level data.
The [model card](MODEL_CARD.md) summarizes intended use, artifact distinctions and
the observed limits relevant to interpreting the results.
The [completion record](docs/completion.md) maps the finished work to its verification evidence.

For the current state and optional AWS monitoring, see [project status](docs/project_status.md).
The published release is evaluated, while the expanded feature-research completion
gate remains open for raw-history distributions, event-time availability and a new
promotion protocol. Completed experiments do not imply that these avenues were tested.

## Final frozen evaluation

| Diagnostic | Reserved weeks 73-91 |
|---|---:|
| Official weekly Gini stability | **0.729674** |
| ROC AUC | **0.875759** |
| Average precision | **0.193755** |
| Raw Brier score | **0.019282** |
| Log loss | **0.080695** |
| Applications | **203,345** |

The evaluated model is a 90% tuned / 10% original LightGBM blend, trained on
**1,323,314 applications from weeks 0-72**. Before evaluation, the release froze
features, weights, no calibration, seeds and development-derived iteration budgets.
The immutable intent identifies the exact two models and encoders. No holdout early
stopping or model selection was permitted.

A **separate inference refit** used all **1,526,659 labeled applications**. The holdout
score above belongs to the development-trained models, not to that all-label refit.
It is a local temporal evaluation, not a Kaggle leaderboard score. Differences in
period length and population mean it should not be read as a gain over development
fold scores.

## What the experiments show

- **Features:** 2,508 candidates; 2,034 structurally eligible; 700 retained. Screening
  used weeks 0-32, before model-validation weeks 33-72. Stored statistics account for
  all 1,808 rejections, including 1,334 eligible features below the computation limit.
- **Ablations:** removing credit bureau A, previous applications or depth-two history
  reduced mean development stability by **0.093813**, **0.031440** and **0.015762**.
  These are controlled removal comparisons, not individual-feature causal effects.
- **Models:** LightGBM led XGBoost, CatBoost and a logistic SGD baseline across five
  expanding folds. The accepted benchmark contains 20 model-fold evaluations and
  aligned out-of-fold predictions for 727,187 cases.
- **Tuning:** eight LightGBM candidates across five folds completed 40 new fits.
  The selected trial improved mean stability from 0.585188 to 0.601238.
- **Blending:** 15 fixed candidates reused saved predictions. The selected blend
  reached 0.601899 mean development stability; its small gain of 0.000661 came with
  a weaker worst fold. It is not evidence of statistical significance.
- **Expanded feature research:** 4,617 additional hypotheses, 256 retained, 20 new
  comparison fits. The full extension improved pooled AUC but reduced mean stability
  by **0.025284**. Doubling the original budget to 1,400 gave a **0.001290** mean gain
  and a weaker worst fold. These post-release development results did not change the release.
- **Calibration:** eight past-fold sigmoid/isotonic fits evaluated 544,611 later
  development cases. Neither improved pooled Brier score or log loss. This four-fold
  comparison has a different population from the five-fold model-selection study.

The official metric is:

`mean weekly Gini + 88 * min(weekly slope, 0) - 0.5 * residual standard deviation`

Gini is `2 * ROC AUC - 1`. Probability metrics and reliability complement the
ranking metric. No calibrated production default-probability claim is made.

## Engineering and reproducibility

The feature engine builds 34 train/test blocks from 17 relational groups. Saved
artifacts bind raw-data, feature, validation, configuration, code and dependency
identities. Global frequency maps are learned only on the relevant fit population.

Expensive runs checkpoint to S3 with content hashes, conditional writer leases and
verified read-back. Valid completed fits and prediction batches are reused. UTC logs
include stage progress, stage/total elapsed time and heartbeats. Reviews do not
retrain models. Native model reloads reproduce the saved probabilities.

CI runs Ruff, strict mypy, tests with warnings treated as errors, real notebook
execution, output reproduction and unchanged-notebook reuse. Generated exports are
classified separately from authored Python and notebooks in GitHub language statistics.
Canonical filenames are edited in place.

```bash
bash scripts/start_here.sh --require-persistent-storage
uv run --locked python scripts/review_model_benchmark.py --force
uv run --locked python scripts/review_model_tuning.py --force
uv run --locked python scripts/review_model_selection.py --force
uv run --locked python scripts/review_model_release.py --force
uv run --locked python scripts/review_submission.py --force
uv run --locked python scripts/review_feature_research.py --force
uv run --locked python scripts/review_calibration.py --force
```

The dependency lock uses Python 3.12.14. The [operating contract](AGENTS.md),
[frozen release policy](configs/model_release.json) and
[release runbook](docs/model_release.md) describe the validation boundaries.
Do not rerun training merely to view results.

## Inference and owner-controlled submission

[Notebook 10](notebooks/10_submission.ipynb) rebuilds features from the raw test files
supplied to the run, checks the native bundle, resumes predictions and lets the owner
explicitly generate, validate, save and download a `case_id,score` CSV. It checks exact
sample coverage/order, unique IDs, finite probabilities, saved-file round trips and
artifact lineage. **CSV generation defaults to off; nothing uploads to Kaggle.**

The ten public competition example cases have passed raw-feature parity and packaged
inference checks. They are an integration fixture. Hidden-test execution, hidden-test
resource limits and a leaderboard score have not been verified.

## Research scope

This is an evaluated, bounded research portfolio release. The expanded study tests
ratios, dispersion, category interactions, recency, household comparisons,
missingness and training-only peer statistics, with all rejections accounted for.
It includes five-fold interpretation and native-model replay. A finite search does
not establish exhaustive discovery: raw-history quantiles/skew, verified event-time
trends, neural challengers and fully nested promotion studies were not performed.
Peer quantiles describe training populations; they are not new per-customer history
quantiles. Native categorical target statistics were evaluated through CatBoost.

The original holdout is now observed and bound to the frozen release. Further feature
or model exploration must use development data and be labeled accordingly; it cannot
reuse this holdout as a new untouched test. Production deployment, lending-policy
fairness validation and an operational monitoring service are outside the tested scope.
