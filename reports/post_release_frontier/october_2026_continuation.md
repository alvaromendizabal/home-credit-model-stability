# October 2026 frontier continuation

This report continues the employer-facing post-release record after the earlier
[October frontier snapshot](october_2026.md). It summarizes the later temporal,
representation and ensemble studies while preserving the same publication boundary:
aggregate evidence is public; exact active feature identities, borrower-level matrices,
private checkpoints and runtime bundles remain outside Git.

The evaluated September release remains frozen. The later confirmed tree + histogram +
DenseLight ensemble remains the strongest externally measured post-release artifact in
the repository at **0.56035 public / 0.47652 private**. Later studies below are reported
as research evidence unless they separately pass the locked promotion lifecycle.

## Clean sparse, ranges and fresh tree diversity

A clean sparse-feature study removed the previously harmful direct payment/risk branch
and retained the strongest temporally stable sparse candidates. A small sparse blend
passed selection with approximately **+0.00567 mean stability**, but later confirmation
improved by only about **+0.00035**, below the preregistered threshold.

Explicit max-minus-min ranges materially weakened the tested LightGBM representation.
A fresh XGBoost challenger on the sparse/range matrix improved ordinary AUC while
reducing the official stability metric, so that exact representation/model combination
was closed.

**Decision:** retain clean sparse features as reusable representation evidence; reject
the exact range/XGBoost branch.

## Relational source mining

The next study added new relational summaries from underused source families and
hierarchical payment/contract history. The selected auxiliary-relational LightGBM
blend reached approximately **+0.00971 mean selection stability** with 3/3 wins.

The gain did not persist on later confirmation periods: mean confirmation stability
was effectively flat at about **-0.00004**, with one win and one loss.

**Decision:** relational features contain useful signal, but this LightGBM formulation
was not robust enough to promote.

## Recovered categorical CatBoost

A separate CatBoost frontier combined the stable sparse base with a substantially
larger categorical representation and relational additions. The strongest robust
candidate improved all three selection folds and mean AUC, but its mean stability
gain was about **+0.00131**, below the +0.002 selection threshold.

**Decision:** preserve the recovered categorical representation as useful modeling
evidence; do not spend another round on near-duplicate CatBoost depth/iteration tuning.

## Broad snapshot and temporal representation

A wider feature pass expanded the candidate pool to roughly 3,000 representations and
materialized **637 new features** across broad/core, temporal, first/last-difference
and interaction families.

The selected wide-temporal LightGBM blend produced a large **+0.01836 mean selection
stability** with 3/3 wins, then reversed to **-0.00128** on confirmation.

This was an important result: feature breadth can create large apparent gains on earlier
periods without reliable transfer to later periods.

## Heterogeneous bridge and stability-aware training

A fixed LightGBM + CatBoost bridge produced **+0.01154 mean selection stability** with
3/3 wins and positive AUC change, but failed confirmation at about **-0.00014**.

Week-balanced/stability-aware training produced another strong selection result
(**+0.01631**) that again failed later confirmation (**-0.00071**).

**Decision:** close ordinary global weighting as a current direction and treat temporal
transfer, not training-set fit, as the central research constraint.

## Nested stacking and drift diagnosis

A nested heterogeneous stack used saved prediction streams rather than new base-model
fits. It reached approximately **+0.01393 mean selection stability** with 3/3 wins,
then failed confirmation at about **-0.00100**.

An adversarial classifier distinguishing earlier from later training periods reached
approximately **0.998 AUC**, showing substantial multivariate distribution shift.
Adversarial feature pruning and recent-likeness weighting were tested as targeted
responses. The selected pruning blend improved selection by about **+0.00871** but
again failed later confirmation.

**Decision:** preserve the shift diagnostic; close the exact pruning/weighting schemes.

## Temporal reconstruction and postprocessing audit

A refresh-date-based temporal reconstruction recovered held-out week structure with
approximately **0.9996 correlation**, **0.24-week MAE**, and 100% accuracy within one
week in the audited development periods.

The public competition-specific score-adjustment recipes tested against that reconstructed
timeline were harmful for this system and were rejected. This work is intentionally
reported separately from predictive-model research.

**Decision:** keep the portable temporal-reconstruction infrastructure; reject the exact
postprocessing recipes tested so far.

## Reduced-sample XGBoost

A public-solution-inspired reduced-sample XGBoost study tested smaller training populations
rather than another full-data tree fit. The strongest XGB100-global blend reached
approximately **+0.00327 mean selection stability** with nearly neutral AUC change.

One fold missed the nonnegative worst-fold gate by roughly **6e-5**, so the candidate
was not promoted. This is one of the strongest remaining complementary-model signals.

The run also exposed an execution-path issue: the installed XGBoost build requested
CUDA but effectively executed on CPU in the active SageMaker image. Later confirmation
work therefore preserves version/device consistency rather than mixing runtimes across
validation folds.

## Learned heterogeneous Ridge stack

A leakage-aware Ridge second level combined existing heterogeneous prediction streams.
Its frozen candidate reached approximately **+0.01874 mean selection stability** with
3/3 wins and a **+0.00080 worst-fold delta**.

Confirmation rejected it: one later fold was slightly positive and the other materially
negative, for approximately **-0.00120 mean confirmation stability**.

**Decision:** reject the selected stack; preserve the heterogeneous OOF infrastructure.

## Target-free stable-feature diversity

The registered fallback combined a target-free stable-feature mask with three
complementary learners. The selected small blend improved all three selection folds at
about **+0.00329 mean stability**, then lost both confirmation folds at about
**-0.00039 mean stability**.

**Decision:** reject the exact model pool while retaining the target-free stability
screen as reusable research infrastructure.

## Independent processing pipeline

A separate processor rebuilt missingness, categorical handling, correlation reduction
and model inputs independently of the incumbent representation. Six selection fits and
12 blend candidates completed.

The least-negative candidate was about 95% incumbent plus 5% independent LightGBM:
approximately **-0.00101 mean stability**, with one of three fold wins.

**Decision:** close this exact independent-processing recipe.

## Learned payment-history representation

A matched study compared order-free and chronology-aware historical representations.
The chronology-aware arm improved mean AUC by approximately **+0.00060** while reducing
mean official stability by approximately **-0.01562**.

All registered blend candidates failed selection.

**Decision:** close the tested chronology representation; preserve the normalized
history cache for genuinely learned set/sequence encoders.

## Coherent GPU ranking objectives

A matched GPU XGBoost study compared pointwise classification, global pairwise ranking
and within-week pairwise ranking. All nine selection models and 16 candidate comparisons
completed.

The least-negative candidate was about 95% incumbent plus 5% global pairwise ranking,
at approximately **-0.00162 mean stability**, **-0.00040 mean AUC**, and **0/3 fold
wins**.

**Decision:** close the tested ranking-objective recipe.

## Late frontier — neural, retrieval, robustness and category recovery

The later research program completed the previously registered neural frontier and
continued through several materially different representation classes.

A matched neural study tested efficient shared-weight ensembling together with learned
order-invariant and temporal historical representations. None of the tested recipes
survived the complete temporal promotion contract.

Subsequent studies covered:

- nonlinear numerical encodings with matched architecture controls;
- contract-preserving history encoders;
- training-only support and rarity representations;
- contrastive and denoising-style self-supervised objectives;
- fixed and learned retrieval over earlier labeled applicants;
- supervised source dropout and group-robust objectives;
- independent metric/calibration and validation-ceiling diagnostics;
- recovered categorical identities and strictly past-only category statistics.

Several candidates again produced positive earlier-period evidence but weakened on later
confirmation. One learned-retrieval candidate also failed its matched-control attribution
requirement, showing why a gain versus the incumbent alone is insufficient evidence for a
new mechanism.

The validation-ceiling review reproduced the relevant scoring and calibration behavior
and found no metric defect that explained the plateau. The dominant unresolved problem
remains transfer across substantial temporal distribution shift.

**Decision:** preserve the reusable preprocessing, GPU, recovery, retrieval and
representation infrastructure; close the exact rejected recipes and avoid near-duplicate
parameter sweeps.

The public report remains intentionally aggregate. Exact active feature identities,
private checkpoints, borrower-level matrices and competition runtime recipes stay
outside Git.

## Engineering lessons

Across these rounds, the project now demonstrates:

- strict temporal promotion gates that reject unstable early-period gains;
- heterogeneous tree, neural and meta-model research;
- target-free drift diagnostics and stable-feature screening;
- resumable model-fold checkpoints;
- immutable run manifests and lineage;
- runtime cost and resource telemetry;
- explicit effective-device verification;
- negative-result preservation rather than post-hoc promotion;
- a public/private boundary that keeps the work reviewable without releasing the
  full active competition recipe.

See [research engineering and reproducibility](../../docs/research_engineering.md) for
the system-design view.
