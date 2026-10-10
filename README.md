# Home Credit Model Stability

[![CI](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/ci.yml)

**Credit-risk machine learning evaluated across time, with reproducible evidence and recoverable execution.**

A model can rank historical applicants well and still deteriorate in later periods.
This project treats temporal generalization as an engineering requirement: relational
feature construction, expanding-window validation, controlled ablations, probability
diagnostics, resumable AWS workloads and verified inference all share an explicit
artifact and evaluation contract.

Built by [Alvaro Mendizabal](https://github.com/alvaromendizabal).

**[Explore the interactive demo](https://home-credit-stability-lab.tartmacaw2.chatgpt.site)** ·
[Read the case study](docs/case_study.md) ·
[Verify the published evidence](docs/public_reproducibility.md) ·
[Inspect the model card](MODEL_CARD.md)

## Explore temporal stability

The [interactive demo](https://home-credit-stability-lab.tartmacaw2.chatgpt.site) makes the model-evaluation problem tangible:
adjust a synthetic weekly discrimination curve and inspect how its trend and
volatility affect the stability score. The chart, component breakdown and comparison
update together. It runs entirely in the browser, with no account or backend.

The demo uses clearly labeled **synthetic weekly aggregates**. It neither predicts
an applicant's risk nor represents a trained model, measured business impact or new
research result. Actual observed results are linked below and reproduced separately
by the evidence-review command.

## The project at a glance

| Evidence | Observed scope |
|---|---|
| Data scale | **1,526,659 applications** across **17 relational source groups** |
| Feature engineering | **2,508 initial candidates**, training-only screening, **700-feature frozen release** |
| Model comparison | **Four model families × five temporal folds**, with 727,187 aligned out-of-fold cases |
| Frozen future-period evaluation | **0.875759 ROC AUC**, **0.729674 weekly-Gini stability**, **203,345 applications** |
| Probability diagnostics | **0.193755 average precision**, **0.019282 Brier**, **0.080695 log loss** on that same future-period evaluation |
| Execution | AWS SageMaker, S3 recovery, content hashes, immutable manifests and CPU/GPU research |
| Verification | Executed notebooks, native-model replay, schema/order checks and deterministic report reproduction |

The future-period metrics belong to the development-trained frozen release. A separate
all-label inference refit and subsequent development studies have different evaluation
scopes. This is a research system; lending-policy impact and production fairness have
not been validated. [Model card](MODEL_CARD.md).

## Reproduce a public result in one command

From a clone of this repository, with **Python 3.12**:

```bash
python3 scripts/review_public_evidence.py --output artifacts/public-review
```

Open `artifacts/public-review/index.html` for the standalone report, or inspect
`artifacts/public-review/review.json` for machine-readable results. The command uses
only Python's standard library and committed aggregate evidence. It requires no
package installation, credentials, cloud account or model training.

It authenticates the input reports, recalculates stability from weekly Gini aggregates,
checks comparison populations and reproduces the model/ablation review. It fails on
changed or inconsistent evidence. It does not reconstruct borrower predictions or
recompute AUC from unavailable private labels; the report labels those source metrics
separately. The checked-in [public report](reports/public_review/index.html) and
[JSON result](reports/public_review/review.json) are regenerated and compared in CI.

[Reproducibility guide](docs/public_reproducibility.md) ·
[Five-minute review](docs/employer_review.md) ·
[Engineering case study](docs/case_study.md)

## Architecture

![Data, modeling, validation and inference architecture](reports/portfolio/system_architecture.svg)

| Layer | Design choice | Why it matters |
|---|---|---|
| Relational features | Case-local aggregates and training-population encoders | Makes joins, availability assumptions and category fitting inspectable |
| Evaluation | Expanding windows; frozen release choices before future-period evaluation | Separates model selection from the observed release evaluation |
| Experiments | Matched controls, feature-family removals and explicit rejection decisions | Tests whether added complexity contributes transferable signal |
| Execution | Hash-pinned manifests, checkpoint recovery and resource/cost telemetry | Reuses valid work and makes interrupted execution diagnosable |
| Inference | Native model reload, feature/schema parity and exact case ordering | Tests the path from source records to a portable prediction artifact |
| Public review | Aggregate reports, executable checks and a bounded disclosure policy | Lets reviewers verify published results without private data or runtime bundles |

## Three findings worth inspecting

**Feature families have measurable value.** Removing credit bureau A, previous
applications or depth-two history reduced mean development stability by **0.093813**,
**0.031440** and **0.015762**, respectively. These are controlled removal effects
conditional on the tested model and feature set; they are not causal effects.
[Feature ablation evidence](reports/feature_ablation/README.md).

**More features did not automatically improve temporal performance.** A 256-feature
extension improved pooled AUC but reduced mean stability by **0.025284**. The study
retains this negative result and the corresponding feature-selection/accounting
receipts. [Feature research](notebooks/11_feature_research.ipynb).

**Selection gains need later-period confirmation.** Later model and representation
studies include candidates that improved earlier development periods and weakened on
later confirmation periods. Those candidates were rejected. This establishes the
importance of the validation procedure within the tested studies; it does not turn
repeatedly explored development folds into independent test data.
[Research record](docs/post_release_research.md).

![Model-family comparison and controlled feature removals](reports/portfolio/overview.svg)

Both panels above use the five development folds. The future-period release metrics
at the top of this page refer to a separate population and are not directly comparable.

## Review the implementation

| Question | Entry point |
|---|---|
| How were features screened and joined? | [Feature notebook](notebooks/02_feature_engineering.ipynb), [feature code](src/home_credit/features) |
| How were models compared fairly? | [Executed benchmark](notebooks/05_benchmark_review.ipynb), [validation code](src/home_credit/validation) |
| What was frozen before release evaluation? | [Release notebook](notebooks/09_model_release.ipynb), [model card](MODEL_CARD.md) |
| How does work resume after interruption? | [Research engineering](docs/research_engineering.md), [runtime code](src/home_credit/runtime) |
| What can be reproduced without private artifacts? | [Public review script](scripts/review_public_evidence.py), [reproducibility guide](docs/public_reproducibility.md) |
| What did unsuccessful experiments teach? | [Calibration](notebooks/12_calibration.ipynb), [post-release research](docs/post_release_research.md) |

## Full review environment

The lightweight command above is sufficient to reproduce the public evidence report.
For the broader authored source and executed notebook suite, use the pinned Python
3.12.14 environment:

```bash
uv sync --locked --group dev
uv run --locked python scripts/review_model_benchmark.py --force
uv run --locked python scripts/review_model_release.py --force
bash scripts/check.sh
```

The [CI workflow](.github/workflows/ci.yml) executes the complete review sequence,
then checks Ruff, strict mypy, warnings-as-errors tests, notebook output reproduction
and verified reuse. Reviewing published results does not retrain the private models.
The [operating contract](AGENTS.md) and [release runbook](docs/model_release.md)
describe the full validation and publication gates.

## Public evidence and private research

Published: aggregate metrics, temporal evaluation design, historical research code,
executed reviews, architecture, data/model contracts and selected reproducibility
checks. Private: borrower-level records, derived matrices, trained research checkpoints,
active feature identities, current ensemble specifications and operational bundles.

The interactive demo illustrates the stability metric using synthetic data. The
separate public evidence review reproduces supported calculations from published
aggregates. The complete private training and serving environment requires assets
outside GitHub. See the [precise boundary](docs/public_reproducibility.md).

The evaluated release is preserved as a historical artifact. Further research is
tracked separately and does not rewrite its evidence. No cloud workload needs to
remain running to review this repository.
