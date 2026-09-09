# Raw-history research evidence

The registered study is complete and independently verified. It asks whether
within-applicant distribution shape adds value beyond the original 700 features.
The comparison uses the same 727,187 development cases and five temporal folds.

## Observed results

| Condition | Mean stability | Worst fold | Change vs control | OOF AUC | OOF AP | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Without history skew (783) | 0.588678 | 0.422994 | 0.003490 | 0.847326 | 0.205255 | 0.032412 | 0.126706 |
| Original 700 | 0.585188 | 0.393682 | 0.000000 | 0.846894 | 0.204525 | 0.032425 | 0.126805 |
| Original + history shape (796) | 0.584195 | 0.409243 | -0.000994 | 0.847020 | 0.205200 | 0.032413 | 0.126752 |

Full fold results and omission sensitivity are in [comparison.json](comparison.json)
and executed [notebook 11](../../notebooks/11_feature_research.ipynb). Omission ranges
are descriptive, not confidence intervals. No result changes the frozen release or
reuses its already-observed holdout. This is a post-release exploratory study.

Both history conditions improve the weakest fold but win on only three of five
folds. Their fold-omission ranges cross zero; without skew, the range is
-0.002966 to +0.007128. The modest mean benefit depends on which periods
are included. Together with the weaker engineered extension and the fragile
1,400-feature gain, this supports closing the registered search without further
ad hoc expansion or retuning. It does not establish exhaustive discovery of every
possible feature. The frozen release is preserved; any later promotion requires
its own protocol and new independent evaluation evidence.

## What was tested

- 28 raw shards / 1,102,188,410 bytes verified against the frozen input hashes.
- 14 raw history sources audited; 12 have eligible numeric columns. The two
  remaining sources contain no eligible numeric history fields.
- 131 numeric columns generate 524 median, IQR, p90 and skew candidates.
- Early training-only structural screening leaves 378, then missingness leaves
  334 eligible. The fixed budget retains 96; all 428 rejections are explained.
- Retained: 50 medians, 11 IQRs, 22 upper-tail values and 13 skew features.
- Two contrasts complete ten full fits plus two separate early-screen fits.
  The five original control fits are reused. Removing skew does not refill slots.
- Every one of 33 resolved date fields has an explicit availability decision.
  Chronological lags/trends are excluded without a field-level contract. Supplied
  numeric calendar parts remain numeric candidates, not evidence of ordering.

## Provenance and verification

Training source: `5c18beceffbe5e79f2bf193ec4478eb71cd63f3e`.

Study: `dbae30f9c3b54ccbb694e4fa5f2b3ad0499dde474a5ff2ef7c76017f66d57b5b`.

[The registered plan](../../configs/history_research.json) fixes the definitions,
input hashes, feature budget, learners, seeds, folds and contrasts. The original
feature snapshot and release retain their identities. The history manifest is a
separate development-only view; it is not a deployable raw-test recipe.

[screen.json](screen.json) is the complete hypothesis and rejection catalog.
[history_manifest.json](history_manifest.json) records the actual raw schema audit,
source hashes and feature-block identities. [study.json](study.json) records every
partition and native-model checkpoint. [The publication policy](../../configs/history_research_review.json)
pins the exact bytes of all five evidence files.

Independent verification replayed 1,454,374 predictions from ten saved models and checked 141 metric identities. Maximum prediction error: 0; maximum metric error: 4.66e-15.

[verification.json](verification.json) contains the receipt. The verifier uses
pandas/sklearn/NumPy polynomial regression independently of the training stability
implementation. Verification and report generation perform zero model fits.

## Recovery and review

Each 100,000-case partition, source block, screen and fit is durable in S3 before
its completion flag advances. Conditional per-fit receipts survive a coordinator
interruption. The finished driver invokes the study again and requires identical
results with zero new fits. UTC logs include elapsed time and heartbeats.

The final processing state and reuse evidence are recorded in
[execution.json](execution.json). Models, encoders and borrower-level predictions
stay in the project bucket; Git contains aggregate research evidence.

The next model promotion would need the registered
[future protocol](../../configs/future_promotion.json) and a new independent
population. That protocol is registered, not executed. The current employer-facing
release remains the previously evaluated frozen blend.
