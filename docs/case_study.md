# Case study: credit-risk modeling across changing application periods

## Problem and scope

A credit-risk model can discriminate well across a pooled historical sample and still
weaken in the periods when it will be used. This project asks whether improvements
survive a move into later application periods, and whether the resulting model can
be exported with the same features, predictions and artifact lineage.

I built and integrated relational data processing, feature selection, temporal
validation, model comparison, calibration studies, AWS execution, recovery and
offline inference.
The data contains **1,526,659 labeled applications across 17 relational source groups**.
The historical release uses **700 selected features**. Later research is recorded
separately so the original evaluation remains interpretable.

## Design decisions

### Fit the validation design to the intended use

The development benchmark compares four model families on five expanding temporal
folds, with **727,187 aligned out-of-fold cases**. Feature screening precedes the
model-validation periods, and categorical frequency maps fit the relevant training
population. The implementation separates ranking quality, probability quality and
variation over time.

The stability metric combines mean weekly Gini with penalties for declining weekly
performance and residual variation. AUC, average precision, Brier score and log loss
remain visible so improvement in one objective does not hide deterioration in another.

After development choices were frozen, the release trained on **1,323,314
applications from weeks 0–72** and evaluated **203,345 applications from weeks
73–91**. Model selection and early stopping did not use those evaluation labels.
These weeks have since been observed and are never described as a fresh test for
later research.

Evidence: [benchmark notebook](../notebooks/05_benchmark_review.ipynb),
[temporal protocol](../src/home_credit/validation/protocol.py),
[metric implementation](../src/home_credit/metrics/stability.py) and
[release evaluation](../reports/model_release/evaluation.json).

### Make feature complexity earn its place

The original feature system produced **2,508 candidates**, of which **2,034** were
structurally eligible and **700** were retained under the frozen computation budget.
Screening records account for every exclusion. Controlled source-family removal
experiments measured what the additional joins and historical aggregates contributed.

Removing credit bureau A reduced mean development stability by **0.093813**;
removing previous applications reduced it by **0.031440**; removing depth-two
history reduced it by **0.015762**. These are matched predictive comparisons, not
causal estimates about borrowers or the underlying data sources.

More features did not consistently improve the objective. A later extension screened
**4,617 additional hypotheses** and retained **256**. It improved pooled AUC but
reduced mean stability by **0.025284**, so it did not replace the evaluated release.
That finding shaped the research process: compare additions against saved controls,
inspect period-level effects and retain unsuccessful experiments.

Evidence: [feature engineering](../notebooks/02_feature_engineering.ipynb),
[controlled ablations](../reports/feature_ablation/comparison.json) and
[expanded feature study](../notebooks/11_feature_research.ipynb).

### Separate model improvement from evidence strength

The accepted tuning study completed **40 new model-fold fits** while reusing five
control fits. The selected model improved mean development stability from
**0.585188 to 0.601238**, with improvements on three of five folds. Most of the gain
came from the earliest fold; the complete fold table remains available.

Subsequent selection reused saved predictions rather than fitting models again.
The release froze its model choices and used no calibration before the historical
evaluation. Post-release calibration research then tested whether mappings fitted on
earlier development predictions transferred to later development periods. Neither
tested mapping improved pooled Brier score or log loss. It did not change the release.
Later representation and neural studies have their own selection and confirmation
records, without changing the meaning of that release result.

Evidence: [tuning metrics](../reports/model_tuning/metrics.json),
[selection notebook](../notebooks/08_model_selection.ipynb),
[calibration notebook](../notebooks/12_calibration.ipynb) and
[post-release research](post_release_research.md).

### Make recovery and inference part of the deliverable

The AWS workflow binds source, configuration, dependency, data and feature identities
to saved artifacts. Checkpoints and immutable manifests allow valid completed work to
be reused after interruption. Logs expose progress, elapsed time and resource use;
tests cover checkpoint validation and failure paths.

Inference is a separate release step. It validates raw-feature construction, expected
schema, unique application IDs, case order and finite probabilities. A distinct refit
uses all labeled applications for inference, while the historical evaluation continues
to refer to the development-trained models.

The recorded release verification checked **62 objects**, recomputed all **eight**
saved evaluation metrics and reused unchanged prediction batches. The saved offline
notebook reproduced the verified AWS CSV bytes on **10 supplied public examples**,
with zero model fits. An external hidden-test execution subsequently completed. The
10-example parity check establishes integration consistency on those examples; it
is not a comprehensive population-level equivalence test.

Evidence: [checkpoint code](../src/home_credit/modeling/checkpoints.py),
[inference code](../src/home_credit/modeling/inference.py),
[release verification](../reports/model_release/verification.json) and
[offline execution receipt](../reports/kaggle_submission/execution.json).

## Observed release outcome

| Metric | Frozen later-period evaluation |
|---|---:|
| Weekly-Gini stability | 0.729674 |
| ROC AUC | 0.875759 |
| Average precision | 0.193755 |
| Brier score | 0.019282 |
| Log loss | 0.080695 |
| Applications | 203,345 |

These results evaluate one frozen development-trained release. They are not a measured
business outcome or an evaluation of the all-label inference refit. The period length
and default prevalence differ from the development folds, so the holdout value should
not be treated as a directly comparable improvement over their mean.

I delivered a model evaluation and inference system with an inspectable
experimental record: source-level feature evidence, temporal comparisons, failed
hypotheses, native model lineage and offline execution verification.

## Preserve the meaning of a release

I keep the delivered release separate from subsequent experiments and incomplete
executions. The [current status](project_status.md) and its linked receipts record
that later work. An operational checkpoint or successful report does not, by itself,
establish a new trained model, scientific result or submission score.

## What a reviewer can reproduce

The public repository includes implementation code, aggregate evidence, executed
review notebooks and CI contracts. The [reproducibility guide](public_reproducibility.md)
distinguishes local review checks from full data processing and training. Reading the
results and reproducing the public review layer do not require starting a paid AWS
instance. Private borrower-level artifacts and current research recipes remain outside
the repository.

## Limits and production follow-through

The project has not established production lending readiness. It has not measured
financial lift, validated decision thresholds or completed protected-group fairness
analysis. Competition snapshots do not certify production event-time availability or
outcome maturity. Expanding folds share training data, and repeated development
selection is not an independent generalization estimate.

A production extension would need an approved data-availability contract, independent
prospective evaluation, subgroup analysis, decision-policy validation and monitoring
with ownership and escalation procedures. Those are future requirements, not delivered
capabilities. The [model card](../MODEL_CARD.md) records the current boundary.
