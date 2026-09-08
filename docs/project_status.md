# Project status and optional monitoring

The frozen release and the first expanded feature study are complete. The owner's
broader feature-research gate is **open**. Both statements describe the same project:
finishing a declared experiment does not establish diminishing returns for untested
raw-history distributions or verified event-time features.

The current release contains 700 features. The original screen considered 2,508
candidates, retained 700 and rejected 1,808. The extension considered another 4,617,
retained 256 for comparison and rejected 4,361. Together these are 7,125 hypotheses;
the 256 additions were not promoted into the release.

## Decisions supported by experiments

Credit bureau A, previous applications and depth-two histories helped the original
representation: removing them lowered mean development stability by 0.093813,
0.031440 and 0.015762. The 956-feature extension lowered mean stability by 0.025284.
The 1,400-feature condition improved it by 0.001290 but weakened the worst fold.
Those comparisons hold the learner and five temporal folds fixed.

The status command also recomputes paired fold sensitivity. This removes one fold
at a time from the mean difference; it is a descriptive check, not a confidence
interval. Five expanding-window fits share training cases and are not independent.
The wider original condition wins on only two of five folds, and the sign of its
average benefit depends on which fold is omitted. The full extension remains worse
under every single-fold omission. These findings support preserving the current
release, not declaring every unexplored family unhelpful.

The reserved-period stability score is 0.729674 on 203,345 applications. It belongs
to the frozen development-trained blend. It is neither a leaderboard result nor
the performance of the separate all-label inference refit. Weeks 73-91 have been
observed and cannot serve as a new untouched test.

## Next research steps

1. Build a bounded raw-history distribution study: applicant-level median, IQR,
   upper-tail quantiles and skew for the strongest relational sources. Compare
   screened additions against the same 700-feature control on development data.
2. Audit actual event-date fields and availability before computing chronological
   lags, trends or acceleration. Source group indices alone are insufficient.
   Exclude unsupported histories and record the reason.
3. Predeclare a nested temporal development protocol for further selection. A
   claim about a newly validated release requires a new independent evaluation
   population; reshuffling or reusing the observed holdout cannot supply one.

These are unexecuted requirements, recorded in `configs/research_gate.json`. The
status command refuses an addressed status without hash-pinned evidence. Native
CatBoost categorical statistics were previously benchmarked, but the 700-feature
LightGBM release uses training-only frequency maps. Online target histories are
excluded without outcome-availability timestamps. New model tuning waits for the
feature gate; existing valid fits and the frozen release remain preserved.

## Optional commands

Run from an existing, prepared checkout of this repository. Nothing needs to run
to keep the saved results valid. Do not start a paid Studio instance solely to
monitor completed jobs.

```bash
.venv/bin/python scripts/project_status.py
```

This verifies published identities, feature accounting, results and saved notebook
outputs without AWS access or any model fitting. Successful output includes
`Published release and completed development studies: VERIFIED` and
`Expanded feature completion gate: OPEN`. A missing or inconsistent artifact
returns exit code 1 and `PROJECT_STATUS_FAILED`; it does not report success.

```bash
.venv/bin/python scripts/project_status.py --cloud
```

With the existing AWS credentials, this adds paginated live processing/training
job status in `us-west-2`. A stopped job remains `Stopped`; an access failure never
becomes a misleading zero-active-jobs result. The scope excludes Studio apps and
other AWS services. The command neither launches nor stops resources.

```bash
.venv/bin/python scripts/project_status.py --verify-cloud --json
```

This also streams and hashes all unique release, feature-study and calibration
checkpoints referenced by the validated reports. It reads bytes and may take
longer than a status check. UTC stage logs and 15-second heartbeats go to stderr
and `logs/project-status-*.jsonl`; the JSON status goes to stdout. Every body is
closed, including on corruption or a failed read. This verifies artifact identity,
not a new training run or a new independent evaluation.

For repeated optional job checks from a Linux terminal:

```bash
watch -n 60 .venv/bin/python scripts/project_status.py --cloud
```

Use Ctrl+C to exit. This is a user-run monitor, not a background notification service.
Kaggle CSV generation remains an explicit owner action in notebook 10.
