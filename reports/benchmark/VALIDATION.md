# Validation record

The acceptance implementation was executed against the actual published benchmark,
not only synthetic fixtures. No training was rerun and no AWS compute resource was created.

## Evidence checked

- Training commit: `414397e1f5731aa969b8599d5c11a58459a0e361`.
- Run key: `ea48ad26a735ef4044ebc6a0d2e2750cb54584a2926619ab9d52bcdc0f739438`.
- Manifest SHA-256: `4fb12c48f7af8a9f7f442b2519f17cc0c1eba274c9d34fa6bb76b59c485e58f6`.
- Summary SHA-256: `cb87d1ce0abd7419a115e3b47aa1d332aad27f4d476a36a0028c2c7b2af183f4`.
- 70 files / 250,504,699 bytes verified against the published manifest.
- 20 model-fold receipts, model bytes, prediction bytes, and feature counts verified.
- Fold metrics, pooled OOF metrics, weekly diagnostics, and model ranking recomputed.
- 727,187 distinct OOF cases per model; exact case/week/label agreement across all four models.
- OOF weeks 33-72; no weeks 73-91 in these prediction artifacts.
- Screening feature counts, excluded predictors, protocol hash, configuration hash,
  feature-screen identity, and training provenance checked.

## Code checks

`bash scripts/check.sh` runs Ruff lint, Ruff formatting, strict mypy, and the full
pytest suite with coverage. The expanded suite contains 154 tests, including 24 new
acceptance, report, S3 restoration, and heartbeat regression cases.

New cases cover known metric values; duplicate IDs; nonfinite and out-of-range
probabilities; wrong, missing, or single-class weeks; invalid targets; missing folds;
changed artifact bytes; a wrong metric inside a consistently rehashed publication;
holdout contamination; manifest identity; cached restoration; corrupted download
cleanup; report generation; and unavailable process telemetry.

S3 restoration uses an injected client in tests. The real S3 artifacts were downloaded
through the connected AWS account, then the local acceptance CLI verified them in the
locked project environment. The CLI's direct boto3 download path was not run inside
the user's SageMaker terminal during this review.

The run emits UTC console/JSONL events, per-fold counters, elapsed stage time, a
heartbeat, and total elapsed time. In containers that hide process telemetry, the
heartbeat continues with null CPU/RSS values and an explicit telemetry status.

## Visual checks and limits

The static SVG overview was rendered and visually inspected. The HTML contains four
Plotly figures and embeds its JavaScript. The report-generation test passes and its
tables use accepted aggregate evidence. Interactive browser QA could not be completed
because this environment's browser policy blocked the local preview routes.

Acceptance establishes artifact consistency and reproducible development metrics.
It does not establish final holdout performance, production readiness, causal
interpretability, fairness, raw-data point-in-time correctness, or actual AWS dollar costs.

## Continue in SageMaker

Run from the repository root on `feat/benchmark-acceptance`:

```bash
bash scripts/start_here.sh --require-persistent-storage \
  --accept-benchmark --bucket YOUR_ARTIFACT_BUCKET
```

Success ends with `PHASE_5A_ACCEPTANCE_COMPLETED`. Open `reports/benchmark/report.html`
in a browser. The command restores the completed run; it does not refit the models.

## Persistent startup regression coverage

The startup entry point now prepares a real project-local `.venv`, managed Python,
and persistent dependency caches before quality gates. Tests execute real uv against
a dangling link, check preservation of model files and old link targets, verify
interrupted creation recovery, reject low capacity and temporary mounts, exercise
silent-child heartbeats and failure propagation, and prevent recursive bootstrap.
The original 154-test result above records the earlier acceptance implementation;
current startup verification is recorded in GitHub CI and the startup logs.

On 2026-09-06, the SageMaker startup log confirmed the resized persistent XFS
filesystem at 127.9375 GiB with 124.642 GiB free before installation. Managed Python
3.12.14, all locked dependencies, and the environment/model smoke checks succeeded.
The quality gate then exposed a test assertion that assumed uv's `File exists`
diagnostic always occupied one line. Acceptance had not started in that terminal.

The regression now executes real uv 0.12.10 at three terminal widths, including the
exact word boundary that reproduced the failure. It checks exit status, normalized
`EEXIST` diagnostics, preservation of the dangling target, and a working real virtual
environment after recovery. The previous assertion failed in the forced wrapping
case; the updated assertion passes all three cases without skipping the error check.

The complete local `bash scripts/start_here.sh` run reused its installed environment
and passed 180 tests, Ruff lint/formatting, strict mypy (36 files), and the smoke checks
in 49.794 seconds. Its log recorded periodic heartbeats and total elapsed time. Local
verification used overlay storage; the SageMaker command retains the explicit
persistent-mount requirement. The README background launcher was syntax-checked and
executed with successful and failing workers: it created the log before the viewer
and propagated both exit statuses. No benchmark training was rerun.
