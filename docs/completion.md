# Research portfolio completion record

The project is an evaluated, bounded credit-risk research release. Its standard is
reproducible evidence, clear decisions and honest limits; a subjective employer
rating cannot be guaranteed. Start with the [README](../README.md) and notebooks
[02](../notebooks/02_feature_engineering.ipynb),
[05](../notebooks/05_benchmark_review.ipynb) and
[09](../notebooks/09_model_release.ipynb). The full research trail is linked there.

## 1. What was completed

The work closes the feature-budget comparison, engineered-feature search, temporal
calibration, interpretation audit and independent verification. The canonical
notebooks and model card distinguish observed results, historical experiments and
future hypotheses. No leaderboard or production lending claim is made.

## 2. Original feature engineering

Seventeen relational groups produce 34 train/test blocks and 2,508 original
candidates. Early training-only screening leaves 2,034 eligible and 700 retained.
All 1,808 rejections are accounted for. Features cover numeric distributions,
relative dates, recency, categories, missingness and applicant/related-person context.
Group-index order is not presented as verified chronological order.

## 3. Expanded search and selection

The extension adds 4,617 candidate representations across eight families, making
7,125 combined hypotheses rather than independent signals. Structural and early
temporal screening retains 256, with every rejection recorded. The 50 selected peer
features contain 44 ranks and six conditional median/interquartile transforms.
Raw-history quantiles/skew were not recomputed. See [notebook 11](../notebooks/11_feature_research.ipynb).

## 4. Models and experiment budgets

The existing benchmark covers four model families and five folds, with 727,187
aligned OOF cases. Existing ablation evidence compares three family removals with
the control. Bounded LightGBM tuning completed eight candidates and 40 new fits;
15 fixed blend candidates reused predictions. This completion work preserves and
verifies those earlier results rather than claiming to have rerun their training.

The new feature study completed 20 native comparison fits plus two separate early
screening fits. Calibration completed eight calibrator fits with zero base-model
fits. Corrected interpretation replayed five saved native models with zero new fits.

## 5. Observed feature and calibration decisions

The original 700-feature control has mean development stability 0.585188. The
956-feature extension reaches 0.559904 despite better pooled ranking and probability
scores. Removing added ratios recovers mean stability to 0.584635. Wider original
features reach 0.586478 but weaken the worst fold from 0.393682 to 0.362256.
Feature importance does not establish improvement on the chosen objective.

Neither past-fold sigmoid nor isotonic calibration improves pooled Brier or log loss
on the 544,611 later development cases. These four calibration folds differ from
the five-fold selection population. These negative findings are retained; neither
study promotes a change to the frozen release.

## 6. Frozen release result and population

The 90% tuned / 10% original LightGBM blend was fixed before its reserved-period
evaluation. Development fitting used 1,323,314 cases, weeks 0-72, with fixed 1,852/
1,355 rounds and no holdout early stopping. On 203,345 cases in weeks 73-91 it scored
0.729674 stability, 0.875759 AUC, 0.193755 average precision, 0.019282 raw Brier and
0.080695 log loss. A separate all-label inference refit used 1,526,659 cases.

The holdout is now observed and cannot become a fresh independent test for later
research. Its score is not a Kaggle leaderboard result or directly comparable to
the shorter development-fold mean. The [model card](../MODEL_CARD.md) records the scope.

## 7. Independent verification

Release verification checked 62 objects / 33,186,864 bytes, all eight saved holdout
metrics, the 54-member bundle, 700 features, 36 raw test shards and four unchanged
prediction batches. Feature verification independently recomputed 235 metric
identities from 21 prediction files / 3,635,935 predictions; maximum error was below
6.78e-15. Calibration verification replayed 1,633,833 predictions and checked all
eight fold/pooled diagnostics. Five engineered native models reproduced all 727,187
validation predictions exactly. Receipts and verifier source identities are committed.

## 8. Durable recovery and actual cloud closure

The original feature driver hit its two-hour limit after eleven durable fits.
Three disjoint workers supplied eight fits. After the original writer stopped and
its lease expired, the collector resumed one unfinished fit. No completed fit was
retrained. Completed-fold recovery does not claim mid-tree resumption.

The collector's unchanged second invocation restored twenty records and reported
zero new fits. Corrected interpretation did the same for five records. S3 objects
are content-addressed and verified before conditional ledgers advance. Native
models, encoders, peer references and predictions remain durable and private.
The [execution record](../reports/feature_research/execution.json) includes actual
UTC starts/ends, all nine terminal AWS job states and explicit reuse log events.

## 9. Notebooks and visual review

The review path includes feature engineering, benchmark, archived family ablation,
tuning, selection, frozen release, owner-controlled inference, expanded research
and calibration. Interactive figures include static GitHub fallbacks. Tables expose
fold support, worst-fold behavior, probability metrics, rejected features and model
complexity. SHAP sampling was corrected after an audit found a biased sorted prefix;
the publication preserves the original diagnostic lineage and sampled week counts.

## 10. Quality and publication gates

CI uses the locked Python 3.12.14 environment, Ruff, strict mypy and warnings-as-errors
tests. It executes generated notebooks before linting, reproduces published outputs
byte for byte and proves unchanged review reuse. The operating contract requires
the actual PR head, including published notebook outputs, to pass before merge.
Local Python 3.12.13 verification is disclosed in its receipt; local Jupyter sockets
are unavailable, so CI provides the actual notebook execution gate.

The [publication source CI run](https://github.com/alvaromendizabal/home-credit-model-stability/actions/runs/34266850893)
passed all **434 tests with zero skips**, strict mypy on 75 source files, Ruff and
real notebook execution. Measured test line coverage was **77%**; this is not a
claim of exhaustive coverage, and managed cloud executions are outside that
instrumented test run. Eight canonical notebooks plus the archived ablation review
executed 35 code cells. All five published feature-research figures were visually
reviewed. The [acceptance record](../reports/feature_research/acceptance.json) pins
that source run and the resulting notebook publication. The resulting final PR
head must still pass its own reproduction gates before merge.

## 11. Known limits and employer interpretation

The project demonstrates hypothesis-driven ML, temporal evaluation, negative-result
reporting, reproducible execution and artifact verification. It does not demonstrate
fully nested inference, significant superiority, a neural challenger, operational
monitoring or production lending validation. Sex and birth-related predictors are
disclosed; importance is not a fairness audit or a causal explanation. Event-time
availability and outcome maturity need separate controls for deployment.

## 12. Owner-only boundary and future work

[Notebook 10](../notebooks/10_submission.ipynb) validates raw feature parity, native
artifacts, unique case IDs, exact sample coverage/order and finite probabilities.
Its real raw-data integration covers the ten public examples. Hidden-test execution
and its resource limits are unverified. CSV generation is off by default and nothing
uploads to Kaggle. The owner alone enables, generates, downloads and submits a file.

Further modeling should begin as a separately scoped study with a new promotion
decision and independent evaluation population. It is not necessary to add every
possible model family or engineered formula to close this research release.

One account-side presentation edit remains: the repository's About description still
mentions neural challengers and drift monitoring, which exceed the demonstrated
scope. The connected repository tools do not expose an About editor, and the secure
browser sign-in was interrupted. In the repository's About gear, use:

> Temporal credit-risk research: LightGBM, CatBoost, XGBoost, feature ablations, calibration studies, and verified SageMaker pipelines.
