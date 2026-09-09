# Kaggle submission

The research release and Kaggle evaluation are separate milestones. The frozen
700-feature, 90% tuned / 10% original LightGBM ensemble is already fitted on all
1,526,659 labeled cases. Submission runs perform inference only.

This is a [notebook competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability).
The original deadline was 27 May 2024. The account's late-submission dialog was
checked on 9 September 2026: notebook submissions were available, with direct
file upload disabled. A downloaded ten-row example CSV is not the hidden-test
submission. Kaggle must rerun the saved notebook on its supplied evaluation data.

## Current delivery status

The final 205.82 MB package passed two complete offline AWS export runs on
9 September 2026. Both produced the same ten-example CSV, reused four unchanged
prediction batches, and fitted zero models. A separately built archive matched
AWS byte for byte; all 78 manifest members were verified. The exact 21 dependencies
also installed and imported successfully in Kaggle with Internet disabled.

The updated package is published as version 2 of the owner's **private**
[Home Credit Frozen Inference dataset](https://www.kaggle.com/datasets/alvaromendizabal/home-credit-frozen-inference).
It contains native models, frozen source and wheels, without raw borrower files or
credential files. Saved notebook **version 1**, script version **348432382**, completed
with Internet disabled and zero model fits. Its ten-example CSV matches the AWS
export byte for byte: SHA-256
`c48d8111604d62895e9ffff643bcf42a52b59c2b110385d78a10d95389f3c799`.

The saved version was submitted on 9 September 2026. Kaggle reports
**Succeeded (after deadline)** with **0.56062 public / 0.47429 private** on the account's
[submissions page](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/submissions).
These are the completed hidden-test rerun's leaderboard scores, not the ten-example
integration output or the development-trained model's local holdout score.
The [execution receipt](../reports/kaggle_submission/execution.json) records the
observed stages separately.

The first dataset version is superseded: Kaggle expanded its nested ZIP, which
prevented inference. Version 2 publishes native model members explicitly.

## Reproduce the submission

1. Open [Home Credit — Frozen LightGBM Inference](https://www.kaggle.com/code/alvaromendizabal/home-credit-frozen-lightgbm-inference).
2. Attach version **2** of the **Home Credit Frozen Inference** dataset and the competition input.
3. Use Python 3.12, CPU, and **Internet off**. Save & Run All.
4. Confirm the output contains `submission.csv`, `submission.json`, and
   `SUBMISSION_VALIDATED` in the execution log.
5. Submit that saved notebook version and record Kaggle's returned status and score.

[Canonical notebook 10](../notebooks/10_submission.ipynb) discovers both supported
Kaggle input layouts. It verifies the runtime manifest and launcher before running
the packaged code. The launcher creates an isolated environment and installs only
the exact wheels selected from `uv.lock`, using `--no-index --require-hashes`.
Kaggle's base package versions are not used for scoring. The installer copies only
pip into a private directory and runs under `python -I -S`, avoiding the host's
startup hooks and dependency resolution. It does not require `venv` or `ensurepip`
on Kaggle. The worker uses `-W error`. The pinned Polars 1.44.1 runtime nevertheless
prints nonfatal String-to-Date deprecation messages from the frozen feature code;
these messages are preserved, not filtered. The validated CSV matches both AWS
exports. This release does not claim compatibility with Polars 2.x. Kaggle's notebook
HTML exporter also reports host-package syntax warnings after inference completes.

The native bundle retains its original source, feature recipe, encoders, models and
SHA-256 manifest. Every raw test shard is fingerprinted; completed prediction
batches can be reused only for identical inputs. Export enforces the supplied
sample's IDs, order and column names, checks finite probabilities in [0, 1], reads
the saved CSV back, and records its hash. A different existing CSV is preserved.

## Evidence and limits

The original ten-example integration and all-label model verification remain in
`reports/model_release/verification.json`. Offline package execution and Kaggle
acceptance are recorded separately; neither changes the original model metrics.
The development-trained model's observed holdout stability of 0.729674 is not a
Kaggle score, and does not evaluate the separate all-label inference refit.

The packaging command is `scripts/build_kaggle_assets.py --bundle PATH --destination PATH`
inside the locked project environment. It uses verified wheels from `uv.lock` and
the existing native archive; it never downloads new training data or fits a model.
The dataset contains model assets, source and dependency wheels, without raw
borrower records. Wheel files retain their distribution metadata and license files.

## Optional status check

From the configured SageMaker terminal, this read-only command reports the completed
package-verification job without launching work:

```bash
aws sagemaker describe-processing-job \
  --region us-west-2 \
  --processing-job-name home-credit-kaggle-offline-20260909-0314 \
  --query '{Status:ProcessingJobStatus,Failure:FailureReason,Started:ProcessingStartTime,Ended:ProcessingEndTime}' \
  --output json
```

Expected status is `Completed`, with no failure reason. Listing only `InProgress`
jobs and receiving `[]` proves that no matching job is running; it does not establish
that an export or Kaggle submission succeeded. The terminal Kaggle status and scores
are recorded above from the competition's submissions page.
