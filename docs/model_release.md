# Frozen model release

## Decision made before the outer holdout

The completed 15-candidate development comparison selects 90% tuned LightGBM
(`trial_006`) and 10% original LightGBM by the existing mean-fold stability policy.
Mean stability is 0.6018988903 versus 0.6012378499 for the tuned single model.
Worst-fold stability declines from 0.4791999815 to 0.4725135463. The small mean
improvement is not evidence of statistical significance or state-of-the-art performance.

`configs/model_release.json` freezes the feature snapshot, 700-feature plan,
selection evidence, weights, dependency lock, seeds and temporal boundaries.
Iteration counts are the medians of the five saved development best iterations:
1852 for the tuned component and 1355 for the original component. They must be
recomputed from hash-verified source state before a fit is allowed.

No additional calibrator is selected. Raw Brier score, log loss and reliability
remain explicit diagnostics; these are not validated production default probabilities.
No model, feature, weight, iteration or calibration choice may be selected on weeks 73-91.

## Distinct artifacts and completion gates

1. Fit the frozen two-component ensemble on development weeks 0-72 with fixed rounds.
2. Evaluate weeks 73-91 once for the frozen decision, persist predictions and metrics,
   and retain the exact development-trained model and encoders used for those metrics.
3. Refit the same frozen specification on weeks 0-91 for inference. Never attach the
   earlier holdout score to this all-label model as though that model was evaluated there.
4. Verify inference against the known raw test files and frozen feature snapshot,
   then test multi-shard inputs, missing-history tables, row alignment and resumption.
5. Execute the final notebook in non-export mode. The owner alone enables CSV
   generation and downloads it; no automatic Kaggle upload or submission is implemented.

The downloadable competition test files contain ten public example cases. A CSV
for those cases is not evidence of hidden-test execution or a leaderboard score.
Competition execution must recompute features from the test files supplied to that run.

The configuration is a frozen plan, not evidence that these remaining stages have run.
Published execution receipts and reports, not this document, determine completion.

## Observed execution

The planned stages subsequently completed in managed SageMaker job
`home-credit-model-release-20260908-0240`, at source
`83ff36e2d324c8f72c99d024ba34720c69c771e1`. The frozen evaluation was recorded at
2026-09-08 03:04:22 UTC, and the full release completed at 03:31:07 UTC.
`reports/model_release/state.json` and `evaluation.json` preserve that original record.

Official holdout stability is 0.7296737944587718 on 203,345 cases. Both distinct
all-label native models completed, raw parity passed for 700 features on ten public
examples, and a complete rerun reused the saved holdout, models and prediction batches.
The raw-data examples are not hidden-test execution evidence.

The feature and release reviews run from committed aggregate evidence with no cloud
access. `scripts/verify_model_release.py --bucket ...` independently restores and
hash-checks all release members, recomputes saved prediction metrics, runs packaged
raw inference and executes notebook 10 with export disabled. It never refits a model.

For owner-controlled CSV generation, extract the verified `inference_bundle.zip`,
set the three input paths in notebook 10 and enable its generation flag. On Kaggle,
attach the model bundle and use that run's competition test files and sample schema.
The notebook can import the matching source included in the bundle without requiring
a repository checkout. The execution environment still needs compatible dependencies.
