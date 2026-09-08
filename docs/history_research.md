# Raw-history distribution study

This is a development-only study after the original release holdout was observed.
The original feature snapshot, model bundle and reported holdout evaluation retain
their identities. This extension is not a new deployable model.

## Question and mechanism

Means and extrema can hide asymmetric or heterogeneous histories. For each numeric
column in every depth-one and depth-two raw source, generate the within-applicant
median, interquartile range, 90th percentile and adjusted Fisher-Pearson skew.
Pool all shards before aggregation. These are applicant-history distributions,
distinct from the earlier training-population peer percentiles.

Quantiles use [linear interpolation](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.quantile.html).
Skew uses the [bias-corrected estimator](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.skew.html).
IQR and p90 require three finite observations. Skew requires five and positive
variance. Nonfinite or unsupported summaries remain missing. Empty histories do
not become zero balances. Dates, targets and source group indices are excluded.

## Registered experiment

[The plan](../configs/history_research.json) fixes definitions, feature count,
input hashes, contrasts, seeds and compute budget before full-data results.

1. Read only the supplied raw histories for development applicants in weeks 0-72.
   Verify the locked raw manifest and the entire byte digest of every input shard.
2. In weeks 0-24, remove constant, near-constant and exactly duplicate candidates.
   Fit the original early target/drift screen, using weeks 25-32 for its validation.
   Retain at most 96 candidates. Store every hypothesis and rejection reason.
3. Reuse the original 700-feature control predictions. Add the selected history
   features without changing the original LightGBM parameters or fold assignments.
4. Repeat the comparison after removing the selected skew features, without
   re-screening replacements. Ten full fits cover two conditions and five folds.
   If no skew survives screening, reuse the identical condition instead of fitting it.
5. Compare mean and worst fold stability, AUC, average precision, Brier and log loss
   on exactly the same 727,187 development cases. Show each fold and descriptive
   fold omission sensitivity. These overlapping folds do not support an independent
   five-sample significance test.
6. Replay the saved native models and independently recompute the metrics. Preserve
   negative results. Do not retune in response to these validation outcomes.

The source manifest records every candidate date field and its physical storage
types. A date suffix, raw row position or `num_group1`/`num_group2` does not establish
event meaning or availability time. Without a field-level contract in the locked
snapshot, chronological lag/trend/acceleration features are explicitly excluded.
Order-invariant summaries rely on the competition's supplied historical snapshot;
they do not certify production point-in-time availability.

## Execution and recovery

```bash
uv run --locked python scripts/run_history_research.py \
  --bucket sagemaker-us-west-2-560403859723
```

Run from the exact clean training commit recorded in the published study. A different
commit creates a different experiment; do not rerun training to view results.
Every 100,000-case partition, source block, early screen and model fit has a verified
S3 checkpoint. Conditional trial receipts recover fits completed just before a
coordinator interruption. Two workers share one renewable study lease; only the
coordinator updates the ledger. Logs report UTC, elapsed time and heartbeats.

The independent verifier accepts the study directory and bucket. It restores
verified checkpoints, replays native model predictions and uses pandas, sklearn
and NumPy polynomial regression to check the reported metrics without model fitting.

## Further promotion

Any future model change follows the separately registered
[promotion protocol](../configs/future_promotion.json). Weeks 73-91 are already
observed. An exploratory gain cannot replace the frozen release's independent
evaluation evidence. A new release claim needs a new independent population.
