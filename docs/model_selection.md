# Development model selection

## Completed development comparison

The completed tuning study selected `trial_006`: mean fold stability **0.6012378499**,
versus **0.5851883724** for the reused control. It wins three of five folds; most of the
improvement comes from fold 1. This is a development result, not a final holdout score.
The [executed tuning review](../notebooks/07_model_tuning.ipynb) preserves the full nine-row
comparison and the control/winner fold comparison. Its JSON is a documented aggregate
excerpt; the complete study remains in immutable S3 storage.

The 15-candidate fixed comparison is complete. Mean fold stability selects 90% tuned
LightGBM and 10% original LightGBM: **0.6018988903**, only **0.0006610403** above the
tuned single model. Four folds improve, but the worst fold declines from **0.479200**
to **0.472514**. This is a small development gain, not proof of a robust improvement.
The [executed selection notebook](../notebooks/08_model_selection.ipynb) shows the
complete evidence and that tradeoff.

A bounded managed SageMaker run verified the saved comparison and executed the
original review without new model fitting. It reused all 15 completed candidate
records. The later GitHub publication retains the exact S3 aggregate bytes and adds
interactive Plotly charts with static GitHub fallbacks; its rendering has separate
provenance from the original scoring identity.

## Read or reproduce without AWS

```bash
uv run --locked python scripts/review_model_selection.py
```

This verifies the committed aggregate digest, scoring-source/configuration/lock
lineage, all 75 candidate-fold records, selected weights and population coverage.
Add `--force` to reexecute the four code cells. An unchanged successful notebook is
reused only when its execution receipt and dependency/output hashes match. Failed
execution does not replace the previous canonical notebook. The offline HTML embeds
Plotly JavaScript, and CI checks deterministic reproduction and receipt reuse.
This path requires no AWS credentials, raw loan records or model fitting.

Do not rerun `start_model_tuning.sh` to read the completed study. Its training identity
includes the source commit; a new commit is not permission to redo the completed fits.

## Restore or verify the S3-backed study in SageMaker

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

- `notebooks/08_model_selection.ipynb`: executed tables and Plotly/static figures.
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

The full-data blend comparison and subsequent frozen release have completed; do not
restart tuning or model fits merely to view them. The release froze no calibration
before evaluating weeks 73-91. Its reliability and raw probability metrics remain
explicit diagnostics, without a production probability claim. The subsequent
[calibration study](calibration.md) fitted earlier development folds and evaluated
later folds. Neither tested map improved pooled Brier or log loss, and neither
used the observed holdout or changed the release. Neural
challengers are optional research and were not part of the accepted four-family benchmark.

The separate [release](model_release.md) completed development and all-label refits,
raw feature parity and packaged inference. Notebook 10 lets the owner explicitly
generate, validate, save and download `submission.csv`. Its real raw-input path was
executed with CSV generation disabled. No automatic Kaggle upload occurs. These
public-example checks do not establish hidden-test execution or a leaderboard score.
