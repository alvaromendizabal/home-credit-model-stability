# Completed development feature research

Start with [the executed notebook](../../notebooks/11_feature_research.ipynb) and
[the research protocol and conclusions](../../docs/feature_research.md).

| Record | What it establishes |
|---|---|
| [screen.json](screen.json) | All 4,617 hypotheses, source columns, formulas, rationales and 4,361 rejection decisions |
| [comparison.json](comparison.json) | Five conditions, all 25 fold metrics, 20 new native-fit records and pooled comparisons |
| [study.json](study.json) | Completed conditional S3 ledger, screen identities and worker provenance |
| [interpretation.json](interpretation.json) | Five native-model replays and corrected diagnostic identities |
| `diagnostics_fold_1.json` through `diagnostics_fold_5.json` | Full feature importance, within-week grouped permutations, training redundancy, SHAP sampling and prediction parity |
| [verification.json](verification.json) | Independent recomputation from 21 prediction files, 3,635,935 predictions and 235 metric identities |
| [execution.json](execution.json) | Actual AWS job states, timings, interrupted-work recovery and zero-fit reuse log records |
| [acceptance.json](acceptance.json) | Publication-source CI, actual test coverage and visual review |
| [publication.json](publication.json) | Executed notebook hash, renderer identity and dependency-bound execution receipt |

The [review policy](../../configs/feature_research_review.json) pins the bytes and
SHA-256 of every research input used by the notebook. The loader reconciles the
catalog, model-fold grid, fit budget, metrics, native-model lineage, independent
verification and corrected sampling before displaying results. Semantic-corruption
tests also repin altered files to verify that checks extend beyond checksums.

Training source: `dd69a8cd935dc922c5055efbf5a35d9266de01e7`.
Parallel orchestration source: `1d7f85286e6aba12720f9836351235b93b650ad2`.
Corrected interpretation source: `e92dd56e486bb7f05b119e150f0cc0262fa413c1`.
Training study: `6e5fb0a67d28bf465142c2de8d8d1577632463bfa70d2d2413762c82ccc6b79e`.
Interpretation study: `33ee7ddb34c5bd80be6399339175a6e4de5e9b239787e44cc4cb6daf5d86d924`.

Only aggregate evidence is public. Borrower-level predictions, native models and
learned peer references remain in verified private storage. Interpretation links
the original diagnostic hashes and corrects the sampling without changing training.
This post-release study uses development weeks only and never changes the frozen
model or its now-observed holdout evaluation.
