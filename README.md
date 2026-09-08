# Home Credit Model Stability

Credit-risk modeling under temporal distribution shift: point-in-time features,
expanding-window validation, four model families, controlled feature ablations,
bounded tuning and reproducible SageMaker execution.

## Start with the evidence

Read [the executed model-selection notebook](notebooks/08_model_selection.ipynb)
for the latest comparison, fold-level tradeoffs, model diversity and reliability.
The [tuning review](notebooks/07_model_tuning.ipynb) explains the selected LightGBM;
the [benchmark review](notebooks/05_benchmark_review.ipynb) introduces the question,
official metric and original model families. The
[feature ablation notebook](reports/feature_ablation/06_feature_ablation.ipynb)
explains why all 700 screened features were retained.

Notebook 08 includes Plotly figures with static GitHub fallbacks. Its
[offline interactive report](reports/model_selection/report.html) embeds JavaScript;
download the HTML and open it in a browser. No AWS account or loan-level data is
needed to read or reproduce the published reviews.

## Latest completed result

The bounded selection study compares **15 fixed candidates on 727,187 development
cases**, using saved, hash-verified predictions and **zero new Home Credit model fits**.
The declared mean-fold stability objective selects **90% tuned LightGBM + 10% original
LightGBM**. The complete results are in [selection.json](reports/model_selection/selection.json).

| Development diagnostic | Original LightGBM | Tuned LightGBM | Selected 90/10 blend |
|---|---:|---:|---:|
| Mean official fold stability | 0.585188 | 0.601238 | **0.601899** |
| Worst-fold stability | 0.393682 | **0.479200** | 0.472514 |
| Pooled OOF ROC AUC | 0.846894 | 0.849286 | **0.849407** |
| Pooled OOF average precision | 0.204525 | 0.207351 | **0.207859** |
| Pooled OOF Brier score | 0.032425 | 0.032347 | **0.032336** |
| Pooled OOF log loss | 0.126805 | 0.126222 | **0.126176** |

The blend gains only **0.000661 mean stability** over the tuned single model. It
improves four of five folds, but **worsens the weakest fold**. This is a small
observed development gain, not proof of a robust future improvement. The two
LightGBM prediction vectors have Pearson correlation **0.9712**. The compared
XGBoost/CatBoost blends do not win the declared objective, even when some secondary
probability metrics improve.

**These are development-selection results, not final-test or leaderboard claims.**
Weeks **73-91 remain locked**. Earlier development folds inform early stopping and
selection; repeated optimization can overfit them. The earlier-fold-only blend
choice diagnostic is not fully nested, because base-model tuning already used all
five development folds. Weekly population support drops to **825 cases in week 68**;
late-fold variability should not be mistaken for a reliably estimated trend.

## What has been completed

The original benchmark evaluated LightGBM, XGBoost, CatBoost and a lightweight logistic
SGD baseline on five expanding folds. Acceptance verified 70 artifacts and aligned
OOF predictions for 727,187 cases. Its [metric audit](reports/benchmark/metrics.json)
recomputes ranking metrics from unclipped predictions; the older acceptance record
preserves its historical clipping policy.

The [feature ablation](reports/feature_ablation/README.md) completed 20 model-fold fits.
Every tested feature-block removal reduced stability, so the model keeps all **700
screened features**. Candidate blocks need point-in-time justification, training-only
screening and ablation evidence; more features are not automatically better.

The September 7 [tuning study](reports/model_tuning/metrics.json) completed eight new
candidates across five temporal folds: **40 new fits**, plus five reused control folds.
It selected `trial_006`, improving mean stability from 0.585188 to 0.601238. Tuning
improves three of five folds; most of that gain comes from fold 1.

A subsequent managed SageMaker validation run completed at source commit
`6623d00e9118f9847d69c0caaedd71b10e9b32fa`: **287 tests passed**, Ruff and strict mypy
passed, benchmark/tuning reviews executed, and notebook 08 executed successfully.
That run restored all 15 already-completed selection candidates instead of rescoring
them. It verified input hashes/alignment, rebuilt diagnostics, and published the
review and logs to S3. The publication upgrade is validated separately by current CI.

## Reproduce the latest review

From the prepared project environment:

```bash
uv run --locked python scripts/review_model_selection.py
```

This command uses committed aggregate evidence only: it verifies the original S3
content digest, every recorded scoring-source/configuration/lock digest, all 75
candidate-fold records, selected weights, ranking and population support. It runs no
Home Credit training, reads no holdout data, and requires no AWS credentials.

Add `--force` to reexecute all notebook cells. Unchanged verified notebook executions
are reused. Regenerated HTML/SVG and notebook outputs are published only after a
successful execution; the previous canonical notebook survives a failed cell.
UTC JSONL logs include stage and total elapsed time, cell counts and heartbeats.
CI executes the real-data review, verifies reuse and checks byte-for-byte reproduction.

To verify or restore the original S3-backed comparison rather than just read it:

```bash
bash scripts/start_model_selection.sh --bucket YOUR_ARTIFACT_BUCKET
```

The launcher checks persistent storage and the locked environment, runs compatibility
smoke tests and quality gates, then restores verified predictions and completed
candidate evaluations. It does not retrain completed Home Credit models. Compatibility
smoke tests fit tiny synthetic models. See the [runbook](docs/model_selection.md).

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
A renewable S3 writer lease and a local process lock prevent duplicate scoring writers;
each completed candidate is committed before the next begins.

## Environment and durability

```bash
bash scripts/start_here.sh --require-persistent-storage
uv run --locked python scripts/review_model_benchmark.py
uv run --locked python scripts/review_model_tuning.py
```

The environment is locked to Python 3.12.14 and `uv.lock`. `.venv` is a real directory;
managed Python, caches, artifacts and logs live on the project volume, not `/tmp`.
Startup requires 12 GiB free for installation and reserve. EBS persistence is not a
backup against deleting the SageMaker space: source history is in GitHub and
published experiment artifacts are in S3. Credentials and raw loan data are not committed.

CI runs Ruff, strict mypy, tests with warnings treated as errors, and real notebook
execution. Canonical filenames are edited in place. Historical accepted evidence
is retained, and unexpected warnings are not globally suppressed.

## Remaining release work

The next release must freeze the model/ensemble/calibration policy and verify the
final train/test feature contract before the locked holdout is evaluated once.
The ensemble margin is small: retain the tuned single-model result as an explicit
reference rather than presenting the blend as an unqualified improvement.

Final refit, full competition-compatible inference and notebook-based submission
export are **not yet implemented and validated end to end**. The owner must explicitly
**generate, validate, save and download** the submission from the final notebook.
**Nothing is automatically submitted to Kaggle; the review stages create no submission CSV.**

The [operating contract](AGENTS.md), [tuning record](docs/model_tuning.md),
[selection runbook](docs/model_selection.md) and [project plan](PROJECT_PLAN.md)
provide the implementation and research boundaries.
