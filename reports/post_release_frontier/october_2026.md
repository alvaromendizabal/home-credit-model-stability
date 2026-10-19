# Final externally evaluated ensemble

This report records the post-release work completed after the 23 September publication.
It is intentionally **employer-facing and aggregate-only**: validation design, model
families, fit counts and observed metrics are published, while borrower-level matrices,
private checkpoints, runtime bundles, exact recovered-feature identities and active
competition recipes remain outside Git.

The evaluated September release remains frozen. Weeks 73-91 are already observed and
are not reused as a fresh model-selection population.

## External transfer result

A later ensemble combined the frozen tree champion, the previous-application histogram
challenger and a multi-seed DenseLight neural component. Its five-fold development
mean stability was **0.618940**, compared with **0.601899** for the frozen September
release.

The candidate was then measured through the competition's late-submission path:

| Submission | Public | Private |
|---|---:|---:|
| Frozen September inference refit | 0.56062 | 0.47429 |
| Later tree + histogram + DenseLight ensemble | 0.56035 | **0.47652** |

The later candidate improved the private score by **0.00223** while changing the
public score by -0.00027. This is useful external-transfer evidence, but it does not
turn the previously observed local holdout into a new independent test.

## Retained neural component

The preregistered DenseLight direction was executed on GPU and then extended to a
three-seed bagged neural component. The final internal candidate used a robust
probability aggregation across seeds and passed the locked folds 4-5 confirmation
gate before the all-label inference artifact was built.

Engineering evidence included:

- fixed expanding temporal folds with weeks 73-91 excluded from selection;
- GPU inference benchmarking on an NVIDIA L4;
- deterministic checkpoint and preprocessing identities;
- CPU/GPU probability parity below 3e-8 maximum absolute error;
- resumable training and hash-verified model artifacts.

The later Kaggle result above shows that the large internal stability gain transferred
only modestly to the private leaderboard. The exact DenseLight architecture/weights
and private checkpoints remain runtime artifacts rather than public competition code.

## Public reproduction boundary

This page documents the later externally evaluated ensemble, not unpromoted
follow-on experiments or a plan for new research. The original frozen release
and this later ensemble remain separate artifacts with separate evidence.

The [public reproduction guide](../../docs/public_reproducibility.md) exposes the
original feature, validation, inference and aggregate-review contracts. It does
not restore the private fitted ensemble or require an AWS workspace. Exact private
weights, feature identities and ensemble settings are not distributed.

The numerical results above are unchanged from the historical record; this edit
removes the unrelated follow-on experiment catalogue and its obsolete continuation
link. The earlier version remains in Git history. No new training or score is
claimed by this publication.
