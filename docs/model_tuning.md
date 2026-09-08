# Completed bounded LightGBM tuning

The full study completed on **September 7, 2026 at 07:40:22 UTC**: eight new candidates,
40 new model-fold fits and five reused control folds. `trial_006` improved mean fold
stability from **0.5851883724 to 0.6012378499**. Weeks 73-91 remain untouched.

Read the [executed review](../notebooks/07_model_tuning.ipynb) and
[aggregate evidence](../reports/model_tuning/metrics.json). The committed JSON is an
aggregate excerpt of the immutable full S3 ledger; it is not the full ledger itself.

**Next: [run the fixed OOF comparison](model_selection.md). Do not rerun the training
launcher merely to read this study.** Its identity includes the source commit, so pulling
new reporting code is not permission to redo these completed fits under a new identity.

## Frozen design

All 700 screened features were retained after the completed feature-block ablation.
The original five expanding folds, feature inputs and preprocessing were unchanged.
`configs/model_tuning.json` declared the search space before training:

| Parameter | Search space |
|---|---|
| Leaves per tree | 15, 31, 47, 63 |
| Minimum observations per leaf | 100, 120, 250, 500, 1000 |
| Feature fraction | 0.65-1.0 |
| Bagging fraction | 0.7-1.0 |
| L1 regularization | 0.001-10, log scale |
| L2 regularization | 0.1-50, log scale |

Learning rate was 0.03, maximum depth 7, tree cap 2,200, early-stopping patience 120
rounds and CPU thread count six. Every fold constructs a fresh training dataset, so
feature prefiltering is not shared between trials with different leaf-size settings.
Optuna's stable TPE sampler observes the reused control and two initial random proposals,
then makes guided proposals. Trial seeds are deterministic. No candidate is pruned after
an unfavorable early fold. The eight-trial budget is not evidence of search convergence.

Selection maximizes mean official fold stability, with the frozen secondary metrics
and control-preserving tie-break. Early stopping uses fold AUC; parameter selection uses
mean fold stability. ROC AUC and average precision use unclipped ranks, Brier uses raw
probabilities, and log loss uses clipping to `[1e-7, 1-1e-7]`.

## Interpretation

The winner has stronger L1/L2 regularization than the original control, but these are
joint parameter changes, not an isolated causal experiment on regularization.
It improves three of five folds. Most of the stability gain comes from fold 1; folds
3 and 4 regress slightly. Trial 004 has slightly better probability-error metrics.
The official stability objective selects trial 006. Repeated development selection
is not an unbiased future-performance estimate or a leaderboard claim.

## Provenance and recovery

Training commit: `2e52dd908f2b30a783e37157ae9801e0679cc666`.
Full study SHA-256: `0de34e51d0a97bd8cd4064931f81135ba3a5299c30e76b3ca1f178f0cb3569dd`.

Local study: `artifacts/model_tuning/<training-commit>/full/`.
S3 prefix: `home-credit-model-stability/model-tuning/<training-commit>/full/`.
The full ledger, immutable history snapshots, exact trial configurations, models,
OOF predictions, fold receipts, logs and executed report remain in S3. Smoke and full
studies use separate prefixes and never share tuning observations.

The historical launcher was:

```bash
bash scripts/start_model_tuning.sh --bucket YOUR_ARTIFACT_BUCKET
```

Recovery must use the original training identity and verified state. It reuses completed
model folds, publishes receipts before advancing, protects writers with local/S3 locks
and emits UTC progress, stage/total elapsed time and heartbeats. A report-only replay
of the completed study can finish quickly because it does not repeat 40 model fits.
For the current review, use `scripts/review_model_tuning.py`, not the training launcher.

## Remaining work

The fixed OOF comparison is implemented; full-data blend results still need execution.
It must keep the final holdout locked. Final refit, train/test feature parity, optional
calibration and competition-compatible inference have their own validation boundary.
A later notebook must let the owner generate, validate, save and download a submission;
it must not upload to Kaggle automatically.
