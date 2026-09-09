# Research portfolio completion record

**The bounded release and expanded feature-research gate are complete.** The
raw-history study is executed and independently verified; the date audit records
unsupported chronology explicitly; the next promotion protocol is registered.
Registration does not claim that a fully nested study has run. See
[current status](project_status.md) for the completed scope and optional monitoring.

The project is an evaluated, bounded credit-risk research release. Its standard is
reproducible evidence, clear decisions and honest limits; a subjective employer
rating cannot be guaranteed. Start with the [README](../README.md) and notebooks
[02](../notebooks/02_feature_engineering.ipynb),
[05](../notebooks/05_benchmark_review.ipynb) and
[09](../notebooks/09_model_release.ipynb). The full research trail is linked there.

## Closeout checks — 9 September 2026

This bounded research release is closed as of 9 September 2026. The accepted
submission, frozen models, research metrics and executed visual reviews are
preserved. No further submission or model training is required for closure.

| Check | Observed result |
|---|---|
| Kaggle evaluation | Saved notebook v1, script version 348432382: **Succeeded (after deadline)**; **0.56062 public / 0.47429 private**, reconfirmed on 9 September 2026 |
| Submission identity | One accepted evaluation; private input dataset v2 is an input revision, not a second submission |
| Validated implementation | Commit [`75b4565`](https://github.com/alvaromendizabal/home-credit-model-stability/commit/75b45653e9631e10d281386b0fb49fe37f1a58f6), published through [PR #14](https://github.com/alvaromendizabal/home-credit-model-stability/pull/14) |
| Implementation CI | [Run 34382253954](https://github.com/alvaromendizabal/home-credit-model-stability/actions/runs/34382253954): **499 tests, zero skips**, Ruff, strict mypy, real notebook execution, byte-identical publication reproduction and unchanged-review reuse passed |
| Scoped AWS check | At **2026-09-09 19:07:17 UTC**, no `home-credit` processing or training jobs were `InProgress` or `Stopping` in `us-west-2`; all four list calls succeeded |
| Submission export job | `home-credit-kaggle-offline-20260909-0314` was `Completed`, ending at 2026-09-09 03:17:13 UTC |
| Follow-up | The completed Kaggle-result follow-up is paused; no repeat submission was made during closeout |

The AWS check excludes Studio apps and other services; it is not a statement that
all account charges have stopped. The CI record above identifies the implementation
revision checked at closeout. Subsequent documentation changes must pass their own
exact-head CI and notebook gates before merge.

The one known account-side presentation correction is the repository About text
described at the end of this record. It does not block the research release.
Additional robustness audits, fully nested promotion and production validation are
separately scoped future work, not remaining completion gates.

## 1. What was completed

The work closes the feature-budget comparison, engineered and raw-history searches,
temporal calibration, interpretation audit and independent verification. The canonical
notebooks and model card distinguish observed results, historical experiments and
future hypotheses. The separate Kaggle submission is recorded below; no production
lending claim is made.

## 2. Original feature engineering

Seventeen relational groups produce 34 train/test blocks and 2,508 original
candidates. Early training-only screening leaves 2,034 eligible and 700 retained.
All 1,808 rejections are accounted for. Features cover numeric distributions,
relative dates, recency, categories, missingness and applicant/related-person context.
Group-index order is not presented as verified chronological order.

## 3. Expanded search and selection

The extension adds 4,617 candidate representations across eight families, making
7,125 hypotheses in those first two screens rather than independent signals.
Structural and early temporal screening retains 256, with every rejection recorded.
The 50 selected peer features contain 44 ranks and six conditional median/interquartile transforms.
That first extension did not recompute within-applicant distributions. The subsequent
raw-history study adds 524 candidates from 131 numeric columns in 12 eligible
sources, bringing the total across screens to 7,649 hypotheses. It retains 96
(50 medians, 11 IQRs, 22 p90 values and 13 skew features) and explains all 428
rejections. Removing skew leaves 783 features without refilling the budget.
See [notebook 11](../notebooks/11_feature_research.ipynb).

## 4. Models and experiment budgets

The existing benchmark covers four model families and five folds, with 727,187
aligned OOF cases. Existing ablation evidence compares three family removals with
the control. Bounded LightGBM tuning completed eight candidates and 40 new fits;
15 fixed blend candidates reused predictions. This completion work preserves and
verifies those earlier results rather than claiming to have rerun their training.

The new feature study completed 20 native comparison fits plus two separate early
screening fits. The raw-history study adds ten full fits and two early-screen
fits; its five controls are reused. Calibration completed eight calibrator fits
with zero base-model fits. Corrected interpretation replayed five saved native
models with zero new fits.

## 5. Observed feature and calibration decisions

The original 700-feature control has mean development stability 0.585188. The
956-feature extension reaches 0.559904 despite better pooled ranking and probability
scores. Removing added ratios recovers mean stability to 0.584635. Wider original
features reach 0.586478 but weaken the worst fold from 0.393682 to 0.362256.
Feature importance does not establish improvement on the chosen objective.

The 796-feature history condition reaches 0.584195 mean stability (-0.000994 vs
the original control); removing its 13 skew features reaches 0.588678 (+0.003490).

Both history conditions improve the weakest fold but win on only three of five
folds. Their fold-omission ranges cross zero; without skew, the range is
-0.002966 to +0.007128. The modest mean benefit depends on which periods
are included. Together with the weaker engineered extension and the fragile
1,400-feature gain, this supports closing the registered search without further
ad hoc expansion or retuning. It does not establish exhaustive discovery of every
possible feature. The frozen release is preserved; any later promotion requires
its own protocol and new independent evaluation evidence.

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
validation predictions exactly. The raw-history verifier replayed 1,454,374
predictions from all ten models and independently checked 141 metric identities.
Its maximum prediction error was 0; maximum metric error was 4.66e-15.
Receipts and verifier source identities are committed.

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

The later raw-history driver completed its two-condition grid on one managed
processing instance. It then replayed all saved models and invoked the unchanged
study again, proving identical results with zero new fits. Its separate
[execution record](../reports/history_research/execution.json) records the actual
terminal state, source identity and reuse log events. Source partitions and
per-fit receipts support recovery after a coordinator interruption.

## 9. Notebooks and visual review

The review path includes feature engineering, benchmark, archived family ablation,
tuning, selection, frozen release, owner-controlled inference, expanded research
and calibration. Interactive figures include static GitHub fallbacks. Tables expose
fold support, worst-fold behavior, probability metrics, rejected features and model
complexity. SHAP sampling was corrected after an audit found a biased sorted prefix;
the publication preserves the original diagnostic lineage and sampled week counts.
The README adds a compact comparison of model families and controlled feature-block
removals. CI regenerates that SVG from accepted aggregate reports and requires a
byte-identical result. The eight canonical notebooks contain 22 static figures and
17 interactive charts, alongside the archived ablation review.

## 10. Quality and publication gates

CI uses the locked Python 3.12.14 environment, Ruff, strict mypy and warnings-as-errors
tests. It executes generated notebooks before linting, reproduces published outputs
byte for byte and proves unchanged review reuse. The operating contract requires
the actual PR head, including published notebook outputs, to pass before merge.
Local Python 3.12.13 verification is disclosed in its receipt; local Jupyter sockets
are unavailable, so CI provides the actual notebook execution gate.

The earlier [publication source CI run](https://github.com/alvaromendizabal/home-credit-model-stability/actions/runs/34266850893)
passed all **434 tests with zero skips**, strict mypy on 75 source files, Ruff and
real notebook execution. Measured test line coverage was **77%**; this is not a
claim of exhaustive coverage, and managed cloud executions are outside that
instrumented test run. Eight canonical notebooks plus the archived ablation review
executed 35 code cells. All five published feature-research figures were visually
reviewed. The [acceptance record](../reports/feature_research/acceptance.json) pins
that historical source run and its notebook publication. The raw-history
[training-source CI](https://github.com/alvaromendizabal/home-credit-model-stability/actions/runs/34287549682)
subsequently passed 456 tests with zero skips and strict mypy on 81 files. The
new publication adds evidence-contract tests and two figures. Its exact PR head
must pass execution, reproduction and unchanged-review reuse before merge; the
workflow itself enforces these checks. Test totals above are acceptance records
for their stated source revisions, not a claim about all future revisions.

## 11. Known limits and employer interpretation

The project demonstrates hypothesis-driven ML, temporal evaluation, negative-result
reporting, reproducible execution and artifact verification. It does not demonstrate
fully nested inference, significant superiority, a neural challenger, operational
monitoring or production lending validation. Sex and birth-related predictors are
disclosed; importance is not a fairness audit or a causal explanation. The 33-field
date audit found no field-level event/availability contract in the frozen inputs;
chronological lags, trends and acceleration are therefore excluded. This is a
scope decision, not evidence that those features are ineffective. Event-time
availability and outcome maturity need separate controls for deployment.

## 12. Submission delivery and future work

[Notebook 10](../notebooks/10_submission.ipynb) validates raw feature parity, native
artifacts, unique case IDs, exact sample coverage/order and finite probabilities.
Its real raw-data integration covers the ten public examples. On 9 September 2026,
the owner authorized submission preparation and submission. The offline delivery
package passed two export runs with four unchanged prediction batches reused.
Kaggle saved version 1 (script version 348432382) completed offline with zero fits
and reproduced the verified CSV bytes. Kaggle's submitted hidden-test rerun reports
**Succeeded (after deadline)** with **0.56062 public / 0.47429 private**. These
leaderboard scores evaluate the separate all-label inference refit and must not be
substituted for the development-trained model's 0.729674 local holdout stability.
See the [submission runbook](kaggle_submission.md) and `reports/kaggle_submission/`.

Further modeling should begin as a separately scoped study with a new promotion
decision and independent evaluation population. It is not necessary to add every
possible model family or engineered formula to close this research release.

The current-source date-parsing cleanup is independently verified on the original
public inputs: 93 file identities, 700 feature columns, identical encoded matrices
and ten identical predictions, with zero warnings from the candidate implementation.
The resulting CSV hash matches the accepted public integration run. The
[comparison receipt](../reports/kaggle_submission/date_parsing_verification.json)
identifies the exact source and verifier. This check ran locally under locked
Python 3.12.14 with zero model fits. It does not update the frozen Kaggle assets,
erase their historical warnings or claim a new hidden-test evaluation.

One account-side presentation edit remains: the repository's About description still
mentions neural challengers and drift monitoring, which exceed the demonstrated
scope. The connected repository tools do not expose an About editor, so that account
setting remains an owner-side edit. In the repository's About gear, use:

> Temporal credit-risk research: LightGBM, CatBoost, XGBoost, feature ablations, calibration studies, and verified SageMaker pipelines.
