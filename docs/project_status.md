# Project status and optional monitoring

**The bounded employer-facing research portfolio is complete.** The frozen model
has a future-period evaluation, the additional feature studies are executed and
verified, and the canonical notebooks expose the evidence. Completion means the
registered scope is addressed; it does not mean every possible feature or model
has been tried.

## What is finished

| Work | Evidence |
|---|---|
| Original representation | 2,508 candidates; 700 retained; 1,808 explained rejections |
| Engineered extension | 4,617 additions; 256 retained; 4,361 rejected; 20 full fits |
| Raw-history distributions | 524 additions; 96 retained; 428 rejected; ten full fits |
| Event-date audit | 33 date fields across 14 sources; unsupported chronology excluded |
| Temporal comparison | Four model families, five expanding folds; 727,187 OOF cases |
| Release | Frozen 90/10 LightGBM blend; 203,345 later applications evaluated |
| Inference | Native models and raw-feature parity tested on ten public examples |
| Recovery | Completed studies resume from verified checkpoints with zero new fits |

The three screens account for **7,649 hypotheses**. The 256 engineered additions
and 96 raw-history additions are experimental representations; the evaluated
release remains 700 features. [Notebook 11](../notebooks/11_feature_research.ipynb)
compares additions and removals on the same development cases.

## What the results mean

Removing credit bureau A, previous applications or depth-two history lowered mean
development stability by 0.093813, 0.031440 and 0.015762. The first 956-feature
extension lowered stability by 0.025284 despite improving pooled AUC. A wider
1,400-feature representation gained 0.001290 but weakened the worst fold.

The 796-feature history condition reaches 0.584195 mean stability (-0.000994 vs
the original control); removing its 13 skew features reaches 0.588678 (+0.003490).

Both history conditions improve the weakest fold but win on only three of five
folds. Their fold-omission ranges cross zero; without skew, the range is
-0.002966 to +0.007128. The modest mean benefit depends on which periods
are included. Together with the weaker engineered extension and the fragile
1,400-feature gain, this supports closing the registered search without further
ad hoc expansion or retuning. It does not establish exhaustive discovery of every
possible feature. The frozen release is preserved; any later promotion requires
its own protocol and new independent evaluation evidence.

Full results include each fold, pooled probability metrics and omission sensitivity.
The latter removes one fold at a time from the mean difference; it is descriptive, not a confidence
interval. Expanding-window fits share training cases and are not independent.

The reserved-period stability score is **0.729674**. It belongs to the frozen
development-trained blend, not the separate all-label inference refit or a Kaggle
leaderboard. Weeks 73–91 have been observed and cannot become a new untouched test.

## What comes next

The next step is employer review: read notebooks **02 → 05 → 09**, then use
notebooks 11–12 for the feature and calibration research. No further paid training
is required to present this release. The completion record links the actual
verification and cloud execution receipts.

An optional Kaggle submission remains an owner action in notebook 10. Its CSV
generation defaults to off. Hidden-test execution and a leaderboard score are
unverified. A later research iteration would use the separately registered
[future promotion protocol](../configs/future_promotion.json), which is **not
executed**, and a new independent evaluation population for a new release claim.

The feature gate in `configs/research_gate.json` is closed only by hash-pinned
evidence. The date audit records explicit exclusions instead of interpreting
historical-record indices as time. It does not certify production event-time or
outcome-maturity controls. Native CatBoost categorical statistics were benchmarked;
the LightGBM release uses training-only frequency maps.

## Optional monitoring

Run from an existing, prepared checkout. Nothing needs to run to keep the saved
results valid; there is no need to start a paid Studio instance just to monitor.

```bash
.venv/bin/python scripts/project_status.py
```

This validates published identities, feature accounting, results and the saved
notebook inventory without AWS or model fitting. Successful output includes
`Published release and completed development studies: VERIFIED` and
`Expanded feature completion gate: PASSED`. A missing or inconsistent artifact
returns exit code 1 and `PROJECT_STATUS_FAILED`.

```bash
.venv/bin/python scripts/project_status.py --cloud
```

With existing AWS credentials, this adds paginated live processing/training job
status in `us-west-2`, including completed, stopped and failed jobs. Access failures
propagate. The scope excludes Studio apps and other AWS services; it does not
launch or stop anything.

Your earlier `--status-equals InProgress` command returning `[]` meant only that
no matching jobs were active. It neither started an experiment nor proved that one
had succeeded. To inspect the actual distribution study directly:

```bash
aws sagemaker describe-processing-job \
  --region us-west-2 \
  --processing-job-name home-credit-history-research-20260908-2248 \
  --query '{Status:ProcessingJobStatus,Ended:ProcessingEndTime,Failure:FailureReason}'
```

For optional repeated checks from a Linux terminal:

```bash
watch -n 60 .venv/bin/python scripts/project_status.py --cloud
```

Use Ctrl+C to exit. This is a user-run monitor, not a notification service.

```bash
.venv/bin/python scripts/project_status.py --verify-cloud --json
```

This additionally streams and hashes the unique release, feature-study,
raw-history and calibration checkpoints referenced by the validated reports.
It can read substantial data and takes longer than checking job states. UTC stage
logs and 15-second heartbeats go to stderr and `logs/project-status-*.jsonl`;
JSON status goes to stdout. Bodies are closed on success and failure. It verifies
artifact identity and does not fit models or perform a new evaluation.
