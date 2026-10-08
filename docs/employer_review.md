# Five-minute technical review

This project evaluates credit-risk models across changing application periods and
carries the selected model into verified offline inference. It covers **1,526,659
labeled applications from 17 relational source groups**. The public evidence is
readable without AWS access, private data or model retraining.

## The review path

| Time | Open | What to inspect |
|---|---|---|
| 0:00–1:00 | [Case study](case_study.md) | Problem, design decisions, measured outcomes and limits. |
| 1:00–2:00 | [Feature engineering notebook](../notebooks/02_feature_engineering.ipynb) | How 2,508 candidates became 700 features, with training-only screening and explicit rejection accounting. |
| 2:00–3:00 | [Benchmark notebook](../notebooks/05_benchmark_review.ipynb) | Four model families on the same five expanding folds; inspect weekly stability alongside AUC and probability metrics. |
| 3:00–4:00 | [Release notebook](../notebooks/09_model_release.ipynb) | The frozen later-period evaluation, model lineage and the separate all-label inference refit. |
| 4:00–5:00 | [Public reproducibility guide](public_reproducibility.md) | What can run from a clean checkout, what the checks establish and which assets remain private. |

**Measured release result:** weekly-Gini stability **0.729674**, ROC AUC
**0.875759**, average precision **0.193755**, Brier **0.019282** and log loss
**0.080695** on **203,345 later-period applications**. The
[aggregate evaluation](../reports/model_release/evaluation.json) records the
population and weekly results. This evaluation belongs to the development-trained
release. It does not evaluate the separate model refit on all labels.

## Follow a claim into code

| Review question | Implementation | Evidence or check |
|---|---|---|
| Can future periods leak into model selection? | [Temporal protocol](../src/home_credit/validation/protocol.py), [encoding](../src/home_credit/modeling/encoding.py) | [Protocol tests](../tests/unit/test_validation_protocol.py), [encoding tests](../tests/unit/test_modeling_encoding.py) |
| Does each data source justify its cost? | [Controlled ablations](../src/home_credit/modeling/ablation.py) | [Executed ablation notebook](../reports/feature_ablation/06_feature_ablation.ipynb) |
| Can interrupted work resume safely? | [Checkpoint handling](../src/home_credit/modeling/checkpoints.py), [experiment store](../src/home_credit/modeling/experiment_store.py) | [Checkpoint tests](../tests/unit/test_modeling_checkpoints.py), [store tests](../tests/unit/test_experiment_store.py) |
| Is inference consistent with the trained representation? | [Inference pipeline](../src/home_credit/modeling/inference.py) | [Inference tests](../tests/unit/test_inference.py), [offline execution receipt](../reports/kaggle_submission/execution.json) |
| Are published results maintained as executable evidence? | [CI workflow](../.github/workflows/ci.yml) | [Publication and review boundary](public_reproducibility.md) |

## Three decisions worth discussing

- **Optimize for temporal reliability.** Tuning improved mean fold stability from
  **0.585188 to 0.601238**, but only three of five folds improved. The
  [tuning record](../reports/model_tuning/metrics.json) makes that concentration
  visible rather than presenting the mean as a universal gain.
- **Measure feature value.** Removing the credit-bureau-A source reduced mean
  development stability by **0.093813**. The controlled
  [ablation results](../reports/feature_ablation/comparison.json) connect data
  engineering work to predictive value; they do not establish causal effects.
- **Keep negative results.** An expanded feature representation improved pooled
  AUC while reducing mean stability by **0.025284**. The
  [feature study](../notebooks/11_feature_research.ipynb) explains why it was not
  promoted. The [calibration study](../notebooks/12_calibration.ipynb) similarly
  preserves unsuccessful probability-mapping comparisons.

For further depth, review [model selection](../notebooks/08_model_selection.ipynb),
[research engineering](research_engineering.md) and the aggregate
[post-release research record](post_release_research.md). These extend the evidence;
they do not redefine the observed release evaluation as an untouched test.

## Scope of the evidence

This is an evaluated research and inference system. It does not claim a deployed
lending service, measured financial benefit or validated lending policy. Fairness,
production event-time availability, label maturity and operational decision thresholds
require additional work. See the [model card](../MODEL_CARD.md).

Public code and aggregate reports support technical review. Borrower-level matrices,
private trained artifacts and current research recipes are intentionally withheld;
review reproduction and full training reproduction have different requirements.
