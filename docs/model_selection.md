# Development model selection

## Why this is next

The completed tuning study selected `trial_006`: mean fold stability **0.6012378499**,
versus **0.5851883724** for the reused control. It wins three of five folds; most of the
improvement comes from fold 1. This is a development result, not a final holdout score.
The [executed tuning review](../notebooks/07_model_tuning.ipynb) preserves the full nine-row
comparison and the control/winner fold comparison. Its JSON is a documented aggregate
excerpt; the complete study remains in immutable S3 storage.

Compare a fixed grid of saved predictions before spending time on more model training.
Do not rerun `start_model_tuning.sh` to read the completed study. Its training identity
includes the source commit; a new commit is not permission to redo the completed fits.

## Run in the existing SageMaker project

```bash
bash scripts/start_model_selection.sh --bucket YOUR_ARTIFACT_BUCKET
```

The launcher validates persistent storage, reconciles the existing locked environment,
runs compatibility smoke checks and all quality gates, then launches model selection.
The compatibility smoke checks fit tiny synthetic models; no Home Credit model is retrained.
No new AWS compute resource is provisioned.

For a terminal-disconnection-safe launch from the repository root:

```bash
bash <<'BASH'
set -euo pipefail
mkdir -p logs
LOG="logs/model-selection-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
: > "$LOG"
nohup bash scripts/start_model_selection.sh --bucket YOUR_ARTIFACT_BUCKET \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
printf 'Worker PID: %s\nLog: %s\n' "$PID" "$LOG"
printf 'Ctrl+C stops this viewer; the worker continues.\n'
tail --pid="$PID" -n 40 -F "$LOG"
wait "$PID"
BASH
```

Creating the log before `tail` avoids the launch-time missing-file message. It was a
viewer race, not evidence that the completed tuning job failed.

## Frozen experiment

`configs/model_selection.json` pins the full completed tuning study, two source checkpoint
manifests, the feature manifest, the temporal protocol and four OOF file hashes.
Only the saved prediction files are downloaded, not model binaries or raw loan tables.
The expected population is **727,187 cases**, covering development weeks **33-72**.

There are **15 predeclared candidates**: four single models (tuned LightGBM, original
LightGBM, XGBoost and CatBoost); nine blends pairing the tuned incumbent with each other
model at incumbent weights 50%, 75% and 90%; equal-weight blends of the three model
families and of all four predictors. Weights are not repeatedly optimized against the
same outcomes. Selection uses the locked mean-fold objective and secondary tie-breaks.
An exact remaining tie keeps the tuned single-model incumbent.

Each input must have complete and unique case IDs, aligned targets/weeks/folds, the exact
week population and finite probabilities in [0,1]. No inner join may silently discard
cases. The incumbent's recomputed metrics must match the original tuning study.

## Recovery and progress

The source/configuration/lock/metric identity determines the study, not unrelated README
commits. A local process lock and renewable S3 writer lease prevent duplicate writers.
Every candidate is committed to an immutable S3 snapshot before the conditional latest
pointer advances. A failed upload cannot mark a candidate complete locally.

Rerun the same command after interruption. Verified input files and completed candidates
are reused. Only the interrupted or uncommitted candidate needs rescoring. No model fit
is repeated. UTC events include invocation elapsed time, candidate progress, stage times
and 15-second heartbeats. Logs remain on the project volume and are published to S3.
An abrupt machine loss may leave the latest unuploaded log tail only on the volume; the
already committed candidate ledger remains authoritative in S3.

## Results and interpretation

A successful run prints `MODEL_SELECTION_COMPLETED` and the canonical review location:

- `notebooks/08_model_selection.ipynb`: executed tables and static figures.
- `reports/model_selection/selection.json`: aggregate metrics, exact weights and provenance.
- `reports/model_selection/report.html`: self-contained offline review.

The run directory under `artifacts/model_selection/<identity>/` retains the local notebook
execution receipt. S3 retains candidate receipts, aggregate evidence, executed notebooks,
figures and published logs. A lost local notebook receipt triggers inexpensive report
execution, not model retraining or repeated candidate scoring. Canonical copies are written
only after all report artifacts are successfully uploaded and verified. Committing the
real results to GitHub is a separate reviewed results change; a successful local run
alone does not mean they have appeared on GitHub.

Inspect mean/worst-fold stability, weekly Gini, reliability and prediction correlations.
The earlier-fold-only weight-choice diagnostic is **not unbiased nested validation**:
the base model's hyperparameters were selected using all development folds. No calibrator
is fitted here. Model selection, calibrated probability estimation and final validation
must not be conflated.

## Completion boundary

The full-data blend comparison must actually run before choosing a new incumbent.
Then freeze the remaining modeling decisions and evaluate weeks 73-91 once. A neural
challenger is optional research, not a prerequisite to reporting the completed benchmark,
and must use a separate compatible environment and its own bounded evaluation.

Final refit, train/test feature parity, competition-compatible inference and notebook
submission generation remain a separate release. That notebook must let the owner
explicitly generate, validate, save and download `submission.csv`; it must not upload
anything to Kaggle automatically. No submission CSV is created by this stage.
