# Home Credit Model Stability

Research-grade credit-risk modeling under temporal distribution shift, built as a reproducible AWS SageMaker portfolio project.

## Current benchmark

![Temporal benchmark overview](reports/benchmark/overview.svg)

The September 6 development benchmark completed **20 model folds** across four model
families. Independent acceptance verified **70 artifacts**, recomputed all fold, pooled,
and weekly metrics, and confirmed aligned out-of-fold predictions for **727,187 cases**.

| Model | Mean fold stability | Worst fold stability | OOF AUC | OOF average precision | OOF Brier |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.585188 | 0.393682 | 0.846894 | 0.204525 | 0.032425 |
| XGBoost | 0.558892 | 0.343432 | 0.846576 | 0.204701 | 0.032441 |
| CatBoost | 0.546330 | 0.297149 | 0.842745 | 0.201164 | 0.032512 |
| Logistic SGD | -0.120484 | -0.586889 | 0.675897 | 0.102612 | 0.036354 |

**LightGBM is the development leader.** Selection uses mean fold stability, not pooled
OOF stability. Development folds inform early stopping and selection. Final holdout
evaluation on weeks 73-91 remains pending; these are not final-test or production claims.
The boosting models use 700 screened features. The logistic SGD baseline uses the
first 256 screened features with training-only imputation and standardization; this
is a lightweight sanity baseline, not an optimized linear-model comparison.

See the [benchmark evidence](reports/benchmark/README.md),
[machine-readable acceptance](reports/benchmark/acceptance.json), and
[interactive report](reports/benchmark/report.html). Download the HTML and open it in
a browser; JavaScript is embedded and no server is required.

### Reproduce acceptance of the completed run

```bash
uv sync --locked
bash scripts/check.sh
uv run --locked python scripts/accept_model_benchmark.py --download --bucket YOUR_ARTIFACT_BUCKET
```

The last command restores the pinned S3 publication into `artifacts/benchmark_acceptance`,
checks SHA-256 identities, recomputes metrics, and writes `reports/benchmark/`. It trains
no models and creates no AWS compute resources. It needs read access to the existing
S3 objects and approximately 310 MB of free disk for the cache and reserve. Repeated
runs reuse verified downloads. UTC console/JSONL logs include stage timings, a 15-second
heartbeat, per-fold progress, failures, and total elapsed time.
Supply the existing bucket locally with `--bucket`, or set `HOME_CREDIT_ARTIFACT_BUCKET`.
The acceptance policy contains no AWS account ID or account-specific bucket address.

For an already restored bundle, omit `--download` and supply `--benchmark-dir PATH`.
The bundle must contain the pinned `checkpoint_manifest.json`. S3 restoration is
covered by injected-client tests; acceptance was also executed against all real artifacts.

The next modeling step is controlled LightGBM/XGBoost tuning and feature-block
ablation, preceded by investigation of weak folds and the screening adversarial AUC
of 0.998944. Keep the outer holdout locked until model and calibration choices are frozen.

## Engineering principles

- Python 3.12 with a project-local `uv` environment and committed `uv.lock`.
- CPU-first modeling with LightGBM, CatBoost, and the CPU-only XGBoost distribution.
- A separate GPU environment will be introduced later for the neural challenger; CUDA dependencies do not belong in the CPU foundation.
- Point-in-time-safe feature engineering and locked out-of-time validation.
- UTC structured logging, stage timings, total runtime, persistent JSONL logs, and heartbeats for long-running stages.
- Explicit unit and integration tests before expensive experiments.
- Canonical filenames only. Source history lives in Git, not `fix`, `repair`, `v2`, or `final` filenames.
- Immutable raw-data manifests and S3 provenance before modeling.

## Start here in SageMaker

From the extracted repository root:

```bash
bash scripts/start_here.sh
```

The entry point first performs a dependency-free source-integrity check. This catches missing internal modules, forbidden filenames, Python syntax errors, shell syntax errors, and required dependency mistakes **before** installing the ML environment.

If bootstrap succeeds:

```bash
bash scripts/connectivity_check.sh
```

Before any Kaggle download, the project enforces a local storage safety floor. The
competition download is blocked unless at least 30 GiB of ephemeral staging capacity is free by default. Override only deliberately with `MIN_STAGING_FREE_GIB=<value>` after verifying an alternate storage plan.

Kaggle is project-managed and invoked through the locked `uv` environment; it does not depend on a global `kaggle` executable.

## CPU foundation

The initial environment includes:

- NumPy / SciPy / pandas
- Polars / PyArrow
- scikit-learn
- LightGBM
- CatBoost
- XGBoost CPU
- Optuna
- SHAP
- boto3
- official Kaggle CLI
- Ruff / mypy / pytest / coverage / pre-commit

The model smoke gate fits and predicts with LightGBM, CatBoost, and XGBoost on deterministic synthetic data before any Home Credit training begins.

## Project sequence

1. Repository and environment integrity.
2. AWS/Kaggle/Git connectivity.
3. Git/GitHub baseline and CI.
4. Kaggle acquisition and immutable raw-data manifest.
5. S3 raw snapshot.
6. Data catalog and schema contracts.
7. Locked expanding-window validation protocol.
8. Point-in-time-safe feature system.
9. Logistic sanity baseline.
10. LightGBM benchmark.
11. CatBoost benchmark.
12. XGBoost benchmark.
13. Optuna optimization and feature/model ablations.
14. Separate neural challenger environment.
15. OOF ensemble, calibration, drift, robustness, SageMaker jobs, MLflow, and final portfolio reporting.

See `PROJECT_PLAN.md` for the detailed roadmap.


## External authentication

Kaggle credentials are intentionally not stored in this repository. Authenticate interactively
inside SageMaker with `uv run --locked kaggle auth login`, then rerun
`scripts/connectivity_check.sh`. Git author identity is also repository-local configuration rather
than project source; set `git config user.name` and `git config user.email` before the first commit.

Raw competition files are staged one at a time through `/tmp` and uploaded to the conventional
SageMaker S3 bucket (`sagemaker-<region>-<account-id>` by default). The ingestion script does not
require bucket-administration permissions. Every uploaded data object is explicitly encrypted with
SSE-S3 and carries its SHA-256 digest as object metadata for post-upload verification.

## SageMaker storage layout

The project intentionally keeps source code and small logs under the persistent JupyterLab home
filesystem while placing the reproducible `.venv` target and uv cache under `/tmp`. The `.venv`
entry in the repository is a symlink created by `scripts/start_here.sh`. This prevents the model
stack from exhausting a small SageMaker home mount; official competition data is likewise staged
one file at a time under `/tmp` and persisted to S3.
