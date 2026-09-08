# Project Plan

## 1. Environment foundation
Create one deterministic Python 3.12 CPU environment with locked dependencies, observability, compatibility smoke tests, and CI.

## 2. External connectivity
Verify AWS, Kaggle competition access, Git, and GitHub independently of the Python package bootstrap.

## 3. Repository baseline
Initialize Git, install pre-commit, create the public GitHub repository, and require CI to pass.

## 4. Raw data acquisition
Download the official Home Credit competition archive with timestamped logs and validate archive integrity.

## 5. Data provenance
Extract immutable raw data, generate file inventories and SHA-256 manifests, and snapshot raw data to S3.

## 6. Data catalog and contracts
Inventory every relevant table, schema, row count, depth, `case_id` cardinality, missingness, and date/categorical structure. Enforce these expectations with tests.

## 7. Temporal protocol lock
Define expanding-window out-of-time folds and a final locked holdout before model optimization. Serialize and hash the protocol.

## 8. Point-in-time feature system
Build lazy Polars feature pipelines for depth-0/1/2 tables with leakage checks, cardinality tests, and deterministic feature manifests.

## 9. Classical model benchmark
Train logistic regression, LightGBM, CatBoost, and XGBoost with identical locked temporal evaluation, OOF predictions, calibration metrics, and runtime/cost metadata.

Completed: four model families x five temporal folds. The published run was restored
and accepted with independent metric recomputation, 70 verified files, cross-model
OOF alignment, weekly diagnostics, and an aggregate-only report. LightGBM leads the
development selection metric. Fit-stage timings are available; AWS dollar costs have
not been established. Final holdout evaluation remains pending.

## 10. Optimization and ablation
Completed: four controlled LightGBM feature conditions x five folds. Every removal
reduced mean fold stability; retain all 700 features. See the
[accepted ablation evidence](reports/feature_ablation/README.md).

Completed: a bounded Optuna TPE study with eight new LightGBM candidates, all five
development folds, the reused accepted control, an authoritative S3 trial ledger,
conditional writer leases, model-fold resume, and executed reports. All 40 new fits
completed; `trial_006` improved mean fold stability from 0.585188 to 0.601238. See [the tuning runbook](docs/model_tuning.md).

## 11. Neural challenger
Install the optional GPU stack and evaluate TabM under the same protocol. Keep it only if it adds performance, stability, or ensemble diversity.

## 12. Ensemble and calibration
Completed: 15 predeclared fixed candidates on 727,187 hash-verified and aligned
saved OOF rows, with zero new model fits. The 90% tuned / 10% original LightGBM
blend leads mean stability at 0.601899, but its 0.000661 gain comes with a worse
weakest fold. The [executed review](notebooks/08_model_selection.ipynb) includes
Plotly/static charts, exact fold deltas, model correlations and reliability.

Next experiment: compare an uncalibrated reference with a bounded, temporally
cross-fitted probability calibrator. Fit only on earlier development folds and
evaluate later folds; report Brier/log loss and preserve the official ranking
objective. Base-model tuning used all development folds, so this diagnostic must
not be presented as fully nested validation. Final holdout remains untouched.

## 13. Drift and robustness
Measure weekly predictive performance, calibration, feature drift, missingness drift, prediction drift, and subgroup robustness.

## 14. SageMaker productionization
Completed: a bounded managed SageMaker Processing execution at source commit
`6623d00e9118f9847d69c0caaedd71b10e9b32fa`, with CloudWatch logs and verified S3
artifacts. It passed 287 tests and reused the completed selection ledger. The job
terminated successfully and did not alter the active development workspace.
Generalized managed training and experiment tracking remain future work, not a
reason to repeat valid fits. AWS dollar costs have not been established.

## 15. Portfolio release
Completed: executed benchmark, ablation, tuning and selection reviews with a short
README navigation path. The selection publication has pinned aggregate/scoring
lineage, deterministic offline HTML/SVG, Plotly and PNG notebook representations,
explicit corruption guards, and tested execution reuse. CI executes actual published
reviews and checks reproduction. Final model/data cards must wait for an accepted
inference release; do not invent final-test, cost or deployment claims.

## 16. Kaggle inference and submission
After selection and final holdout evaluation, refit the inference pipeline and validate
test feature parity, row coverage, probability bounds and `case_id,score` schema.
Implement the final notebook so the owner explicitly generates predictions, validates
sample row count/order and finite ranges, saves the CSV durably and downloads it.
Never generate a real submission outside that notebook or upload to Kaggle on the
owner's behalf. Final refit and competition inference are not yet validated end to
end. A Hugging Face model card/demo is optional only when a publishable model exists
and data/licensing/privacy constraints have been verified.
