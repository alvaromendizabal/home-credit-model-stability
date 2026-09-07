# Bounded LightGBM tuning

The completed feature ablation supports keeping all **700 selected features**. The
control's mean fold stability is **0.585188**; every tested block removal reduced it.
The next question is whether capacity and regularization changes improve performance
over time. This phase implements that study; full-data tuning results are pending.

## Run once, resume with the same command

From the repository root on the existing persistent SageMaker machine:

```bash
bash scripts/start_model_tuning.sh --bucket YOUR_ARTIFACT_BUCKET
```

The launcher runs the locked environment and quality gates, then one capped smoke
trial and the full study. Smoke and full results have separate storage and never
share tuning observations. The full study reuses the accepted control's predictions
and trains **eight new candidates x five folds = 40 new fits**. It creates no new AWS
compute resource. Keep this commit checked out throughout the run.

To keep it running after disconnecting from the terminal:

```bash
bash <<'BASH'
set -euo pipefail
mkdir -p logs
CONSOLE_LOG="logs/model-tuning-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup bash scripts/start_model_tuning.sh --bucket YOUR_ARTIFACT_BUCKET \
    > "$CONSOLE_LOG" 2>&1 < /dev/null &
TUNING_PID=$!
printf 'Worker PID: %s\nLog: %s\n' "$TUNING_PID" "$CONSOLE_LOG"
printf 'Ctrl+C stops this viewer; the worker continues.\n'
tail --pid="$TUNING_PID" -n 40 -F "$CONSOLE_LOG"
wait "$TUNING_PID"
BASH
```

The prior full-feature control took approximately 45 minutes for five folds on the
current six-thread machine. Budget **roughly 5-10 hours** for eight candidates;
tree size and early stopping can change this substantially. This is an estimate,
not a deadline. The study stops after its declared budget and does not automatically
expand the search, evaluate the holdout, or train a neural model.

## What changes and how selection works

The plan in `configs/model_tuning.json` fixes the search space before training:

| Parameter | Search |
|---|---|
| Leaves per tree | 15, 31, 47, 63 |
| Minimum observations per leaf | 100, 120, 250, 500, 1000 |
| Fraction of features per tree | 0.65-1.0 |
| Fraction of rows per bag | 0.7-1.0 |
| L1 regularization | 0.001-10, log scale |
| L2 regularization | 0.1-50, log scale |

Learning rate stays at 0.03; maximum depth at 7; the cap at 2,200 trees; patience at
120 rounds; training uses six CPU threads and the original model seed. Each fold
constructs a fresh training dataset, so feature prefiltering does not leak between
trials that use different leaf-size settings. Feature selection and preprocessing
are unchanged and fingerprinted alongside the dependency lock and training code.

Optuna's stable TPE sampler observes the original control and two initial random
proposals before making guided proposals. A deterministic seed is assigned to each
trial. There is **no pruning across folds**: discarding candidates after the earliest,
weakest period would distort comparisons across time. The eight-trial budget is an
initial search, not evidence of convergence or a claim of state-of-the-art performance.

Selection maximizes the mean of the five official fold stability scores. Tie-breakers
follow the locked protocol: worst fold, mean weekly Gini, temporal slope, residual
standard deviation, and Brier score. An exact tie retains the original control.
The official formula is mean weekly Gini + 88 x min(slope, 0) - 0.5 x residual std.
The report also shows OOF ROC AUC, average precision, raw-probability Brier score,
and log loss with the documented clipping convention. Early stopping still uses
development-fold AUC; Optuna selects parameters using mean fold stability.

These are development-selection results. Repeated optimization can overfit these
folds. Weeks 73-91 remain locked for one evaluation after model and ensemble decisions.
The reused control retains its original commit provenance. Training implementation,
feature inputs and protocol are unchanged. The new plan pins the current dependency
lock and verifies the original training/reporting package versions against the accepted
environment record. The unused SHAP plotting dependency has been removed because its
colormap calls emit pending Matplotlib deprecations. LightGBM's native SHAP contribution
API is covered by a real model smoke test that reconstructs raw prediction scores.
There is no warning suppression or installed-package modification; quality gates fail
on warnings from the active pipeline.

## Persistence and progress

Local work lives under `artifacts/model_tuning/<commit>/<smoke-or-full>/`. S3 uses
`home-credit-model-stability/model-tuning/<commit>/<smoke-or-full>/`.

| Artifact | Purpose |
|---|---|
| `study.json` | Authoritative trial parameters, observations, provenance, selection and budget |
| `history/<digest>.json` in S3 | Immutable ledger history, published before advancing the current ledger |
| `configs/trial_001.json`, etc. | Exact input to the canonical benchmark runner |
| Trial checkpoints | Models, predictions, native training logs and verified fold receipts |
| `progress.json` | Local whole-study progress, updated every 15 seconds during training |
| `report/report.html` | Offline interactive report refreshed after each completed trial |
| `report/overview.svg` | Static report figure |
| `report/07_model_tuning.ipynb` | Executed final notebook with embedded static figures |

Every proposal is committed to S3 **before** training. Every completed fold is uploaded
by the existing supervisor before the next fold. Every completed trial is verified
from aligned predictions and committed before the next proposal. Completed trials
are replayed through Optuna's public `create_trial`/`add_trial` API; the persisted JSON
ledger is the source of truth. Rebuilding the small in-memory optimizer does not repeat
training and requires no pickled RNG or transient database. Pending proposals resume
with their original parameters. A machine replacement restores the ledger, downloads
only missing verified inputs, and restores the pending trial's model-fold checkpoints.

The recovery unit is a **completed model-fold**. An interrupted native fit restarts
that fold. If a machine disappears between local completion and successful S3 commit,
that uncommitted fold may need to run again. An ordinary process failure on the same
persistent disk preserves its verified local fold. Notebook failures rerun only the
inexpensive report execution; they do not retrain completed models.

A local process lock and a renewable conditional S3 lease prevent duplicate writers.
The lease renews every 30 seconds and expires after five minutes following an abrupt
machine loss. A second launch reports the remaining lease time. Losing the lease stops
the active process group before publishing more results. S3 compare-and-swap commits
reject stale writers. Credentials use the existing AWS role/provider chain.

Human-readable and JSONL logs include UTC timestamps, stage duration, invocation total
time, process heartbeat/CPU/memory and whole-study completed-fit counts. Fit percentage
is explicitly a count, not a time estimate. Model logs are saved with fold checkpoints;
study logs are uploaded after trials and at clean completion. If S3 is unreachable,
diagnostics remain on persistent disk and the command exits rather than advancing an
uncommitted study. `MODEL_TUNING_COMPLETED` is printed only after the final report and
notebook upload succeeds.

## Tests and release gates

Tests exercise real CPU LightGBM training, interruption after a completed fold,
reuse without retraining, exact proposal replay past TPE startup, input-hash changes,
all selection tie-breakers, aligned temporal evaluation, interrupted transfers,
corrupt objects, S3 commit conflicts, duplicate writers and lease loss. CI executes
the notebook through Jupyter, verifies an embedded PNG and tests receipt-based reuse.
Ruff, formatting, strict mypy and the full existing suite are required before merge.

## Path to Kaggle and the portfolio release

1. Review the tuning results and decide whether an additional bounded search is justified.
2. Evaluate complementary challengers, including XGBoost and the planned TabM neural
   model, under the same temporal protocol. Keep improvements or useful ensemble diversity.
3. Select ensemble/calibration behavior from development predictions, then freeze it.
4. Evaluate the locked holdout once; publish uncertainty, weekly diagnostics, calibration
   and failure analysis alongside the official score and probability/ranking metrics.
5. Refit the selected inference pipeline, validate exact train/test feature parity,
   unique `case_id`, row coverage, finite probabilities and `case_id,score` output.
   Create and execute the competition-compatible Kaggle inference notebook and retain
   its artifact hashes, submission CSV, kernel version and Kaggle receipt where submission
   is available. Competition availability must be checked at that stage; the 2024 contest
   has ended, so creating a valid artifact does not guarantee a new leaderboard score.
6. Publish a readable model card and demo on Hugging Face, with GitHub evidence links.
   Public reports contain aggregate metrics rather than customer-level predictions.

References: [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html),
[Optuna ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html),
[Optuna trial replay](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.create_trial.html),
[official competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability).
