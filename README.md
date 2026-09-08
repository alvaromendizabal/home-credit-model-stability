# Home Credit Model Stability

Credit-risk modeling under temporal distribution shift: point-in-time features,
expanding-window validation, four model families, controlled feature ablations,
bounded tuning and reproducible SageMaker execution.

## Start with the evidence

Read [the executed tuning review](notebooks/07_model_tuning.ipynb) for the latest result.
The [benchmark review](notebooks/05_benchmark_review.ipynb) explains the research question,
metric and original model comparison; the
[feature ablation notebook](reports/feature_ablation/06_feature_ablation.ipynb) explains
why all 700 features were retained. Tables and static figures are embedded for GitHub
viewing. No AWS account or raw data is required to read the published reviews.

## Latest completed result

The September 7 tuning study completed **eight new candidates across five temporal
folds: 40 new model fits**, plus five reused control folds. The selected LightGBM
configuration is **`trial_006`**.

| Development diagnostic | Reused control | Tuned LightGBM |
|---|---:|---:|
| Mean official fold stability | 0.585188 | **0.601238** |
| Worst-fold stability | 0.393682 | **0.479200** |
| Pooled OOF ROC AUC | 0.846894 | **0.849286** |
| Pooled OOF average precision | 0.204525 | **0.207351** |
| Pooled OOF Brier score | 0.032425 | **0.032347** |
| Pooled OOF log loss | 0.126805 | **0.126222** |

The winner improves **three of five folds**, not all five; most of the stability gain
comes from fold 1. Trial 004 has slightly better Brier and log loss, but trial 006 wins
the declared stability objective. The [aggregate excerpt](reports/model_tuning/metrics.json)
identifies the immutable full S3 study and training commit. It is not a new training run.

**These are development-selection results, not final-test or leaderboard claims.**
Weeks **73-91 remain locked**. Earlier development folds inform early stopping and
selection; repeated optimization can overfit them.

## What has been completed

The original benchmark evaluated LightGBM, XGBoost, CatBoost and a lightweight logistic
SGD baseline on five expanding folds. Acceptance verified 70 artifacts and aligned
OOF predictions for **727,187 cases**. Its
[metric audit](reports/benchmark/metrics.json) recomputes ranking metrics from unclipped
predictions; the older acceptance record preserves its historical clipping policy.

The [feature ablation](reports/feature_ablation/README.md) completed 20 model-fold fits.
Every tested feature-block removal reduced stability, so the tuned model keeps all
**700 screened features**. More features are not automatically better: candidate blocks
need point-in-time justification, training-only screening and ablation evidence.

## Run the next stage

The next implemented stage is [development model selection](docs/model_selection.md):
**15 fixed single-model/blend candidates and zero new Home Credit model fits**, using
hash-pinned saved predictions. Full-data blend results are not yet published.

Run from the existing persistent SageMaker project:

```bash
bash scripts/start_model_selection.sh --bucket YOUR_ARTIFACT_BUCKET
```

The launcher checks persistent storage, reconciles the existing locked environment,
runs compatibility smoke tests and quality gates, then performs the comparison.
Compatibility smoke tests fit tiny synthetic models; the completed Home Credit models
are not retrained. No new AWS compute resource is provisioned.

Every completed candidate is saved to S3 before the study advances. Rerunning the same
command restores verified inputs and reuses completed evaluations. A successful run
writes `notebooks/08_model_selection.ipynb`, aggregate JSON and an offline HTML review,
with the full run and logs retained locally and in S3. Publishing these real results
to GitHub is a separate reviewed results change. See the runbook for a detached launch.

**Do not restart the completed tuning study just to view its results.**

## Method and safeguards

Selection uses the mean of five official fold stability scores:

`mean weekly Gini + 88 * min(temporal slope, 0) - 0.5 * residual standard deviation`

Gini is `2 * ROC AUC - 1`. Worst-fold stability, weekly support, pooled ROC AUC,
average precision, raw-probability Brier score and log loss provide supporting diagnostics.
Only log loss clips probabilities to `[1e-7, 1-1e-7]`.
The [frozen protocol](configs/validation_protocol.json) defines the expanding folds and
locked holdout. No random train/test shuffle is used for temporal model selection.

The OOF comparison rejects duplicated/missing case IDs, changed targets or fold/week
assignments, holdout rows, invalid probabilities and artifact identity mismatches.
Earlier-fold-only blend-choice diagnostics are explicitly **not unbiased nested
validation**, because base-model tuning already used all development folds.

## Reproduce and review

```bash
bash scripts/start_here.sh --require-persistent-storage
uv run --locked python scripts/review_model_benchmark.py
uv run --locked python scripts/review_model_tuning.py
```

Reviews use committed aggregate evidence and do not train competition models. Add
`--force` to reexecute their cells. Valid execution receipts reuse unchanged notebooks;
failed execution preserves the last successful notebook. Logs show UTC timestamps,
per-cell progress, stage durations, heartbeats and total runtime.

To independently restore and verify the accepted benchmark from S3:

```bash
bash scripts/start_here.sh --require-persistent-storage \
  --accept-benchmark --bucket YOUR_ARTIFACT_BUCKET
```

The environment is locked to Python 3.12.14 and `uv.lock`. `.venv` is a real directory;
managed Python, caches, artifacts and logs live on the project volume, not `/tmp`.
Startup requires 12 GiB of free space for installation and reserve. EBS persistence
is not a backup against deleting the SageMaker space: source history is in GitHub and
published experiment artifacts are in S3. Credentials and raw loan data are not committed.

CI runs Ruff, strict mypy, tests with warnings treated as errors, and notebook execution.
Historical benchmark and ablation evidence are preserved in place. No repair/fix
filename variants or blanket warning suppression are used.

## Remaining release work

Run and inspect the fixed OOF comparison, then freeze remaining model/ensemble/calibration
choices before evaluating the holdout once. A bounded neural challenger is optional
research and requires its own compatible environment and evidence.

Final refit, train/test feature parity, full competition-compatible inference and
notebook-based submission export remain separate work. The owner must explicitly
**generate, validate, save and download** a submission from notebook code.
**Nothing is automatically submitted to Kaggle, and this stage creates no submission CSV.**

The [operating contract](AGENTS.md), [tuning record](docs/model_tuning.md),
[selection runbook](docs/model_selection.md) and [project plan](PROJECT_PLAN.md)
provide the implementation and research boundaries.
