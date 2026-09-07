# Controlled LightGBM feature ablation

**Completed September 7, 2026:** all 20 fits succeeded; retain the 700-feature control.
See [the accepted evidence](../reports/feature_ablation/README.md) and
[the next tuning stage](model_tuning.md). The instructions below reproduce the
historical experiment from its original commit.

Phase 5B completed successfully on SageMaker. The accepted benchmark covers four
model families and five temporal folds, with 727,187 out-of-fold cases per model.
LightGBM leads: mean fold stability 0.585188; OOF ROC AUC 0.846894; average precision
0.204525; Brier score 0.032425. These are development results, not final test scores.

## Why this experiment

The first LightGBM fold has mean weekly Gini 0.691125 but stability 0.393682: its
negative slope contributes a -0.292516 penalty. The screening adversary separates
early and later samples with AUC 0.998944. That signals temporal shift; it does not
identify its cause or prove that a particular feature block is harmful.

Before adding model complexity or tuning many hyperparameters, test how the
current leader depends on major feature sources and higher-depth aggregates.

| Condition | Retained features | Question |
| --- | ---: | --- |
| `control` | 700 | Can the same settings establish a control under the current code? |
| `without_credit_bureau_a` | 427 | What changes when bureau-A depth-1 and depth-2 features are removed? |
| `without_previous_applications` | 516 | What changes when previous-application features are removed? |
| `without_depth2` | 651 | Do higher-depth features earn their complexity? |

All conditions use the same LightGBM parameters, six CPU threads, seed, and five
expanding temporal folds. Feature screening is not repeated. Removal preserves
the original feature order and does not fill vacancies with different predictors.
The control is retrained under the same source commit as its challengers, avoiding
a comparison confounded by code changes. The accepted benchmark remains intact.

`configs/benchmark_features.json` contains only names, types, and provenance;
it contains no customer-level values or targets. Its 700 entries were extracted
from the S3 screen whose SHA-256 is
`9cd20cf4a831f3acc8dcb73b8afedb281b8a61cd13758f87a0d87f57322e9c2d`.
The original object's bytes and exact extraction were independently verified.

## Run on the existing SageMaker Studio CPU instance

Update the repository to the merged commit, then run this entire block:

```bash
bash <<'BASH'
set -euo pipefail
cd "$HOME/home-credit-model-stability"
mkdir -p logs
CONSOLE_LOG="logs/feature-ablation-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup bash scripts/start_feature_ablation.sh \
    --bucket sagemaker-us-west-2-560403859723 \
    > "$CONSOLE_LOG" 2>&1 < /dev/null &
ABLATION_PID=$!
printf 'Worker PID: %s\nLog: %s\n' "$ABLATION_PID" "$CONSOLE_LOG"
printf 'Ctrl+C stops this viewer; the worker continues.\n'
tail --pid="$ABLATION_PID" -n 40 -F "$CONSOLE_LOG"
wait "$ABLATION_PID"
BASH
```

The canonical launcher verifies persistent storage, the locked environment and
quality gates. It then runs four capped smoke conditions before the full suite.
Success ends with `FEATURE_ABLATION_COMPLETED` and an exit code of zero.
A file lock rejects a second launch. Do not change branches or source files while
training; the recorded source commit is part of every resume identity.

The previous LightGBM benchmark spent about 46 minutes in fitting across five
folds. Four full conditions therefore suggest roughly three hours of fitting on
similar hardware, plus feature loading, smoke tests, reporting and transfers.
This is a planning estimate, not an upper bound or measured ablation runtime.
Studio compute remains billable while its app is running; this script neither
creates a GPU job nor automatically shuts down Studio.

## Progress and recovery

- UTC timestamps, stage durations, total duration, CPU/RSS telemetry and 15-second
  heartbeats are written to console and JSONL logs.
- Each fresh child process fits at most one model-fold, limiting accumulated native
  library memory. S3 receives hash-verified, encrypted, content-addressed checkpoints
  after every completed fold; a manifest and latest pointer commit the snapshot.
- Feature caches and model outputs live under `artifacts/`, not `/tmp`. Interrupted
  feature downloads preserve valid files. Restores write files atomically and publish
  state last. A newer verified local fold is retained if upload was interrupted.
- Run the same block at the same commit to resume. Completed verified model-folds
  are reused. An unfinished native model fit restarts its current fold; it does not
  resume mid-tree. A disconnected terminal does not stop the detached worker. A
  stopped Studio app does stop it, so restart the app and rerun the block afterward.
- Existing raw-data/feature snapshots, the accepted benchmark, and the final holdout
  are retained. No `fix`, `repair`, or numbered replacement source files are created.

Inspect `logs/ablation-full-*.jsonl` for individual condition progress and the
`logs/feature-ablation-*.jsonl` suite log for condition transitions.

## View the results

After full completion, download and open:

- `artifacts/feature_ablation/full/report/report.html`: an offline interactive chart
  and exact metric table. All Plotly assets are embedded.
- `artifacts/feature_ablation/full/report/06_feature_ablation.ipynb`: executed notebook
  with the comparison table, change-vs-control chart and temporal fold chart.
- `artifacts/feature_ablation/full/report/comparison.json`: aggregate metrics and
  source/checkpoint provenance. Keep it beside the notebook when rerunning cells.

The notebook logs cell progress and is published only after successful execution.
Derived reports also upload to S3; their exact object keys appear in the log.
Smoke reports live in a separate directory and are explicitly marked as smoke.
Only aggregate reports should be committed to the public portfolio. Model files,
per-case predictions and raw data remain in private S3 storage.

The comparison recomputes the competition-style stability score, ROC AUC, average
precision, Brier score and log loss from hash-verified fold predictions. It requires
identical case/label/week coverage across conditions and rejects duplicate cases,
nonfinite probabilities, missing weeks and holdout predictions. The report shows
mean fold stability, worst-fold stability and changes from control. No ablation
outcome is claimed before training completes.

## Selection limits and following work

Weeks 73-91 remain locked. Early stopping and experiment selection use development
folds, so repeated comparisons can overfit development; the final holdout is the
future check after feature/model/tuning decisions are frozen. Feature removal
measures sensitivity, not a causal explanation of drift.

Review the weakest-fold and probability-metric changes before deciding what to
retain. Next, add persisted LightGBM/XGBoost Optuna studies, then evaluate ensemble,
calibration and a neural challenger under the same protocol. A linked Hugging Face
account does not require a GPU model for this ablation phase.

Metric reference: [Home Credit competition evaluation](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/evaluation).
