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
Run persisted Optuna studies and controlled feature/model ablations. Prefer evidence over model complexity.

## 11. Neural challenger
Install the optional GPU stack and evaluate TabM under the same protocol. Keep it only if it adds performance, stability, or ensemble diversity.

## 12. Ensemble and calibration
Learn ensemble weights from OOF predictions only; compare calibration approaches on temporally valid data.

## 13. Drift and robustness
Measure weekly predictive performance, calibration, feature drift, missingness drift, prediction drift, and subgroup robustness.

## 14. SageMaker productionization
Move expensive stages into disposable SageMaker processing/training jobs with S3 artifacts, CloudWatch logs, checkpoints, and managed MLflow when the modeling interfaces are stable.

## 15. Portfolio release
Publish reproducible benchmark tables, architecture diagrams, model/data cards, cost report, failure analysis, and a concise employer-facing README.
