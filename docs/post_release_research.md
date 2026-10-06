# Post-release frontier research

The evaluated September 2026 release remains frozen. This document tracks later
research that stress-tests temporal robustness, broadens model and representation
diversity, and improves the reproducibility of the research system without rewriting
the historical release or converting exploratory development evidence into a new
independent-test claim.

## Current verified state

The frozen release remains a 90% tuned / 10% original LightGBM blend. Its accepted
development mean stability is 0.601899, its reserved weeks 73–91 local stability is
0.729674, and the recorded late Kaggle evaluation is 0.56062 public / 0.47429
private. A later post-release ensemble, frozen after its own development confirmation
gate, scored 0.56035 public / 0.47652 private. These evaluation settings are not
interchangeable, and neither Kaggle score rewrites the historical release.

A deterministic repository check,
`python scripts/review_post_release_frontier.py`, recomputes the current frontier
summary from committed CSV/JSON evidence.

## Frontier round 1 — categorical identity

[Notebook 13](../notebooks/13_categorical_identity_frontier.ipynb) independently
tested native LightGBM categorical identity and identity+frequency representations.
Folds 1–3 selected `blend_native`; the candidate was frozen before folds 4–5.
Confirmation then produced only +0.000062 mean stability versus the saved champion,
with one win and one loss, so the predeclared promotion gate failed.

**Decision:** reject the candidate, preserve the frozen release, and close this
direction.

## Frontier round 2 — previous-application occurrence histograms

The [histogram report](../reports/applprev_histogram/README.md) tests a mechanism
reported by a strong public solution: per-applicant counts/shares for low-cardinality
`applprev_1` categories. The vocabulary was learned only on weeks 0–32 and frozen.
Thirty-nine histogram features were added to the existing 700-feature snapshot.

The registered standalone `hist_augmented` model passed folds 1–3 but failed the
frozen folds 4–5 confirmation gate versus the saved champion: +0.000106 mean
stability and -0.000378 mean AUC. It was not promoted.

A 25% histogram / 75% saved-champion blend had already been part of the registered
candidate grid. After the standalone candidate failed, that blend was chosen
post-selection for an explicitly exploratory external transfer probe because it
improved stability on both confirmation folds while retaining nonnegative mean
confirmation AUC/Gini deltas. This is not valid internal promotion evidence.

Across all five development folds the 25% blend improves stability on 5/5 folds,
with mean deltas of +0.003966 stability, +0.000449 AUC and +0.000943 mean Gini.
One all-label histogram fit completed and portable private Kaggle overlay assets were
prepared. The committed evidence does not claim a new hidden-test score.

## Zero-fit ensemble audit

A follow-up used saved predictions only and evaluated 47 champion/histogram/XGBoost/
CatBoost combinations with zero new model fits. The strongest descriptive candidate
remained exactly 75% saved champion / 25% histogram, with zero weight on XGBoost and
CatBoost. A prequential diagnostic remained positive at +0.004193 stability,
+0.000447 AUC and +0.000956 Gini, but base models were themselves selected on
development data, so this is not nested unbiased validation.

## Frontier round 3 — DenseLight

The [DenseLight contract](../reports/denselight_frontier/README.md) was frozen before
any DenseLight result existed. It independently recreates the DenseLight tabular
neural mechanism documented by leading public work and measures complementarity to
the tree champion.

Selection uses weeks 33–56 (three fits). Only a passing candidate proceeds to weeks
57–72 (two confirmation fits). A single all-label refit is allowed only after
confirmation. Weeks 73–91 are prohibited for selection. The maximum substantive
budget is six fits and there is no artificial wall-clock cutoff.

The readiness record verifies the 700-feature snapshot and prior-probe identities.
At publication time the inspected CPU environment had no CUDA/Torch/LightAutoML
available, so the contract remains `registered_not_executed` in committed evidence.

## Frontier round 3 — DenseLight execution and external transfer

The preregistered DenseLight contract was executed on GPU under the frozen temporal
protocol and then extended to a multi-seed neural component. A later tree + histogram
+ DenseLight ensemble reached **0.618940** mean development stability and passed the
locked folds 4–5 confirmation gate before its all-label inference artifact was built.

That candidate was subsequently measured through the late Kaggle submission path and
scored **0.56035 public / 0.47652 private**, compared with **0.56062 / 0.47429** for
the frozen September inference refit. The private improvement is real external-transfer
evidence, but it is modest relative to the internal stability gain and does not make the
observed weeks 73–91 a fresh test.

Engineering evidence includes L4 inference benchmarking, multi-seed checkpointing,
hash-pinned preprocessing and CPU/GPU probability parity below 3e-8 maximum absolute
error. Exact private checkpoints and the active competition recipe remain runtime
artifacts rather than committed source.

## Frontier round 4 — source-aware tree expansion

A bounded tree study then expanded the source-aware representation and compared fresh
LightGBM/CatBoost variants under the same selection/confirmation boundary.

The unweighted expanded LightGBM improved its same-family control on the selection
folds by **+0.009229 mean stability** and **+0.001354 mean AUC**. A recency-weighted
variant reduced stability by **0.014187** relative to the unweighted expansion, so that
weighting direction was closed.

The selected blend passed folds 1–3 but failed the frozen folds 4–5 confirmation gate.
All 14 planned fits completed and no model was promoted.

## Frontier round 5 — categorical and sparse ceiling-escape

A later study independently recreated a larger categorical representation, depth-two
payment-history summaries, a competition risk-assessment field and an early-screened
sparse-feature family.

The direct payment/risk feature branch underperformed the prior expanded LightGBM
(**-0.031755 mean stability**, **-0.002055 mean AUC**). In contrast, the sparse
recovery branch improved that direct LightGBM by **+0.037661 mean stability** and
**+0.001951 mean AUC** on selection.

Recovered categorical representation also materially improved CatBoost relative to
the earlier CatBoost representation, but the selected sparse-enhanced blend failed its
locked confirmation gate: **+0.011655 mean stability** with 3/3 selection wins became
**-0.000577 mean stability** with 1/2 confirmation wins. All 14 substantive fits
completed and the candidate was rejected.

That registered sparse/range/XGBoost/categorical frontier was subsequently executed;
its aggregate outcomes are summarized in the later rounds below and in the
[October frontier continuation](../reports/post_release_frontier/october_2026_continuation.md).

See the [October 2026 frontier report](../reports/post_release_frontier/october_2026.md)
for the earlier snapshot and the continuation for the later employer-facing evidence.

## Frontier rounds 6–10 — sparse, relational, categorical and broad temporal systems

Later bounded studies moved beyond the initial sparse/categorical ceiling-escape work.

A clean sparse branch retained useful signal but missed the confirmation threshold.
Explicit max-minus-min ranges weakened the tested LightGBM representation, and a
fresh XGBoost model on that range representation improved ordinary AUC while reducing
the official stability metric.

A broader relational study added underused source families and hierarchical
payment/contract summaries. Its selected auxiliary-relational LightGBM blend reached
about **+0.00971 mean selection stability** with 3/3 wins, then produced effectively
flat confirmation evidence. A recovered-categorical CatBoost frontier improved all
three selection folds but remained below the predeclared mean-stability threshold.

The broadest representation pass added **637 new features** across core, temporal,
first/last-difference and interaction families. Its selected wide-temporal LightGBM
blend reached about **+0.01836 mean selection stability** with 3/3 wins, but reversed
to about **-0.00128** on confirmation.

**Decision:** preserve the reusable representation work and negative evidence; do not
promote selection-only gains that fail later-period confirmation.

## Frontier rounds 11–13 — heterogeneous bridges and stability-aware training

A fixed LightGBM/CatBoost bridge reached about **+0.01154 mean selection stability**
with 3/3 wins and positive AUC change, then failed confirmation.

Week-balanced/stability-aware training produced another large selection result
(**+0.01631**) that also failed the later-period gate. A nested heterogeneous stack
reusing saved predictions reached about **+0.01393** on selection and likewise failed
confirmation.

These repeated reversals shifted the research emphasis from ordinary global weighting
toward explicit distribution-shift diagnosis and genuinely different model pipelines.

## Frontier rounds 14–15 — temporal drift diagnosis

An adversarial classifier distinguishing earlier from later periods reached
approximately **0.998 AUC**, quantifying substantial multivariate shift. Target-free
adversarial feature pruning and recent-likeness weighting were tested as controlled
responses.

The selected pruning blend improved selection by about **+0.00871** but failed later
confirmation, and recent-likeness weighting did not provide a robust alternative.

**Decision:** retain the drift diagnostic; close the exact pruning/weighting schemes.

## Frontier round 16 — portable temporal reconstruction

A refresh-date-based chronology reconstruction recovered held-out week structure with
approximately **0.9996 correlation**, **0.24-week MAE**, and 100% accuracy within one
week in the audited development periods.

Competition-specific score transformations tested against that reconstructed timeline
were harmful for this system and were rejected. Temporal reconstruction remains useful
engineering infrastructure, but those exact postprocessing recipes are not part of the
promoted predictive system.

## Frontier round 17 — reduced-sample XGBoost

A reduced-sample XGBoost study tested smaller training populations rather than another
full-data tree fit. The strongest XGB100-global blend reached approximately
**+0.00327 mean selection stability** with nearly neutral AUC change.

One fold missed the nonnegative worst-fold gate by roughly **6e-5**, so the candidate
was not promoted. The experiment also exposed an execution-path mismatch: requested
CUDA execution fell back to CPU in the active XGBoost/SageMaker stack. Later work
therefore verifies the effective device and preserves runtime consistency across folds.

## Frontier round 18 — learned heterogeneous Ridge stack

A leakage-aware Ridge second level combines existing heterogeneous prediction streams.
Its frozen selection-stage candidate reached approximately **+0.01874 mean stability**,
**3/3 selection wins** and a **+0.00080 worst-fold delta**.

Later confirmation rejected it. One confirmation fold was slightly positive and the
other materially negative, producing approximately **-0.00120 mean stability** overall.

**Decision:** reject the selected Ridge stack; preserve the heterogeneous OOF
infrastructure and selection evidence.

## Frontier round 19 — target-free stable-feature diversity

The registered fallback used a target-free stability mask with Extra-Trees-style
LightGBM, DART LightGBM and HistGradientBoosting.

The selected small blend improved all three selection folds at approximately
**+0.00329 mean stability**, then lost both confirmation folds at approximately
**-0.00039 mean stability**.

**Decision:** reject the exact stable-feature/model pool; preserve the target-free
screening and complementary-model evidence.

## Frontier round 20 — independent processing pipeline

A separate processor rebuilt missingness handling, categorical treatment, correlation
reduction and model inputs independently of the incumbent matrix. Six selection fits
and 12 blend candidates completed.

The least-negative candidate was approximately 95% incumbent plus 5% independent
LightGBM, with about **-0.00101 mean stability** and only one of three fold wins.

**Decision:** close this exact processing/model recipe. Processor diversity remains a
valid design principle, but this implementation did not improve the frozen metric.

## Frontier round 21 — learned payment-history representation

A matched study compared an order-free history control with a chronology-aware
representation while preserving the core feature system.

Chronology improved mean AUC by about **+0.00060** but reduced mean official stability
by about **-0.01562**. All registered blend candidates failed selection.

**Decision:** close the tested fixed chronological-history recipe. Retain the normalized
history cache and aggregation infrastructure for genuinely learned representations.

## Frontier round 22 — coherent GPU ranking objectives

A coherent GPU XGBoost study compared pointwise classification, global pairwise ranking
and within-week pairwise ranking on matched populations. All nine selection models and
16 candidate comparisons completed.

The least-negative candidate was approximately 95% incumbent plus 5% global pairwise
ranking: about **-0.00162 mean stability**, **-0.00040 mean AUC**, and **0/3 fold wins**.

**Decision:** close the tested ranking-objective recipe; do not spend another round on
nearby objective/weight sweeps.

## Frontier rounds 23–31 — neural, retrieval and robustness ceiling-escape

The next phase deliberately moved away from ordinary tree tuning and tested materially
different mechanisms under the same temporal promotion discipline.

### Efficient neural and learned-history systems

A matched neural batch compared a single-network control with efficient shared-weight
ensembling and learned historical representations. The experiments completed
successfully, but the selected streams did not survive the full later-period gate.

A subsequent numerical-representation study tested nonlinear encodings under matched
architecture and initialization controls. One candidate advanced from the earlier
selection period and then failed later confirmation.

Contract-preserving historical representations were then evaluated to retain information
that month-level pooling can discard. Again, the strongest earlier-period candidate did
not remain strong enough on confirmation.

**Decision:** preserve the reusable GPU, history and numerical-representation
infrastructure; close the exact tested recipes.

### Distribution support and self-supervised objectives

A zero-fit support audit measured missingness, clipping and feature-distribution changes
across the temporal folds, then built training-only rarity representations. The audit
confirmed substantial input shift, but the resulting rarity models did not produce a
confirmed gain.

The project then tested contrastive and denoising-style self-supervised pretraining
before the ordinary supervised fit. These experiments were bounded and controlled
against the same downstream architecture. Their selected candidate failed later-period
confirmation.

**Decision:** retain the support diagnostics and pretraining infrastructure; do not
promote the tested representations.

### Retrieval and learned retrieval

A retrieval study tested whether earlier labeled applicants could provide complementary
local information at prediction time. A follow-up replaced fixed neighbor weighting
with learned attention-like adapters and added query-only controls so that any gain could
be attributed to retrieval rather than merely to access to a newer labeled period.

The apparent selection gain did not beat its appropriate matched control robustly, so
confirmation was not warranted for the learned-retrieval candidate.

**Decision:** close the tested retrieval recipes while preserving the chronology,
reference-memory and inference-replay infrastructure.

### Source robustness and categorical-information recovery

Supervised source dropout and group-robust objectives were evaluated as direct responses
to the severe period shift. The selected source-dropout blend improved both later
confirmation folds slightly, but the mean gain remained below the frozen threshold and
did not beat the matched control robustly.

An independent validation-ceiling review then reproduced the relevant metrics and
calibration behavior, finding no scoring defect that could explain the plateau. That
review also identified repeated categorical-vocabulary saturation in the neural
preprocessing.

The next study therefore recovered additional fitting-supported categorical identities
and tested strictly past-only category statistics. Its selected candidate passed the
earlier selection gate but failed later confirmation.

**Decision:** keep the validation, categorical-recovery and recovery-engineering
capabilities; reject the exact model recipes.

### Aggregate lesson from the late frontier

More than one hundred bounded predictive or adapter fits now span the late frontier.
The common failure mode is not inability to improve an earlier development window. It
is **failure to transfer the improvement into later periods under strong distribution
shift**.

The public repository records mechanisms, aggregate fold decisions and reproducibility
contracts. Exact private feature identities, checkpoints and active competition runtime
recipes remain outside Git.

The aggregate continuation is published in
[October frontier continuation](../reports/post_release_frontier/october_2026_continuation.md).
The engineering contract is summarized in
[research engineering and reproducibility](research_engineering.md).

## Promotion discipline

Post-release candidates remain development research unless a separately registered
procedure and genuinely independent evaluation population justify a new release
claim. The observed September holdout cannot be reused as a fresh untouched test.
An after-deadline Kaggle hidden-test measurement is recorded separately as an
external transfer probe; it does not retroactively make post-selection development
choices internally confirmed.

## Public solution sources

- 59th-place solution:
  https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508202
- First-place solution:
  https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337
- LightAutoML DenseLight implementation:
  https://github.com/sb-ai-lab/LightAutoML
- Final leaderboard:
  https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/leaderboard
