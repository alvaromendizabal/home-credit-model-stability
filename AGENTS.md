# Project operating contract

Preserve canonical filenames and edit them in place. Do not add fix, fixed, repair,
repaired, final-numbered or version-numbered alternatives. Preserve verified raw data,
models, checkpoints and locally executed notebooks; never reset or clean them blindly.

Keep the employer review path short: README, executed benchmark, feature ablation,
and tuning/selection notebooks. Publish aggregate evidence, not borrower-level data.
Separate observed results from planned work. Do not claim state-of-the-art performance
from a small development study, synthetic smoke test or incomparable leaderboard metric.

Use the frozen five expanding folds and the official weekly Gini stability formula.
Keep weeks 73-91 locked until model, ensemble and calibration choices are frozen.
Report mean fold stability, worst fold, ROC AUC, average precision, raw Brier and log loss.
Validate domain-informed feature blocks with training-only screening and controlled
ablations rather than maximizing feature count without evidence.

Every expensive stage needs explicit contracts, tests, UTC logs, stage and total elapsed
times, heartbeats, immutable input identities and verified S3 recovery. Resume valid work;
a changed report must not retrain a model. Never hide unexpected warnings globally.
Do not rerun the completed tuning launcher merely to review its results. The accepted
study is pinned to commit 2e52dd908f2b30a783e37157ae9801e0679cc666.

The owner generates and downloads Kaggle submissions from notebook code. Do not supply
pre-made submissions or automatically upload to Kaggle. A later inference release must
validate train/test feature parity, unique case IDs, exact sample schema/order, finite
probabilities, artifact lineage and durable export before displaying a download link.
Do not describe that release as implemented before its complete path is tested.

Use documented feature branches and pull requests. Run Ruff, strict mypy, tests with
warnings treated as errors, and real notebook execution before merging. Keep executed
review outputs in canonical notebooks. CI may commit regenerated tuning-review outputs
on feature branches only after quality gates pass; the PR head still needs its own CI.
