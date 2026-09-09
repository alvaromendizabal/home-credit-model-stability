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
That original boundary was crossed by the frozen release on 2026-09-08. Its holdout
is now observed. Subsequent feature/calibration research uses development data only;
never relabel those holdout weeks as untouched or use them for further selection.
Report mean fold stability, worst fold, ROC AUC, average precision, raw Brier and log loss.
Validate domain-informed feature blocks with training-only screening and controlled
ablations rather than maximizing feature count without evidence.

Every expensive stage needs explicit contracts, tests, UTC logs, stage and total elapsed
times, heartbeats, immutable input identities and verified S3 recovery. Resume valid work;
a changed report must not retrain a model. Never hide unexpected warnings globally.
Do not rerun the completed tuning launcher merely to review its results. The accepted
study is pinned to commit 2e52dd908f2b30a783e37157ae9801e0679cc666.

The owner explicitly authorized submission preparation and Kaggle submission on
2026-09-09. Generate from the frozen all-label inference release, verify the saved
notebook execution, and record Kaggle's actual acceptance or failure. The inference release must
validate train/test feature parity, unique case IDs, exact sample schema/order, finite
probabilities, artifact lineage and durable export before displaying a download link.
Do not describe that release as implemented before its complete path is tested.

Use documented feature branches and pull requests. Run Ruff, strict mypy, tests with
warnings treated as errors, and real notebook execution before merging. Generate and
execute review notebooks before linting, so newly generated code cannot bypass style
checks. Keep executed review outputs in canonical notebooks. CI may commit regenerated
tuning and selection review outputs on feature branches only after quality gates pass.
The exact resulting PR head must pass its own CI and byte-for-byte publication
reproduction before merge; a successful run on the pre-publication parent is insufficient.
