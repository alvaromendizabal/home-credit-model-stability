# Home Credit Model Stability

Research-grade credit-risk modeling under temporal distribution shift, built as a reproducible AWS SageMaker portfolio project.

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
