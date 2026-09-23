# Post-release frontier research

The evaluated September 2026 release remains frozen. This document tracks later research that attempts to close the gap to stronger public competition systems without rewriting the historical release or converting exploratory development evidence into a new independent-test claim.

## Current verified state

The frozen release remains a 90% tuned / 10% original LightGBM blend. Its accepted development mean stability is 0.601899 and its reserved weeks 73–91 local stability is 0.729674. The recorded late Kaggle evaluation is 0.56062 public / 0.47429 private. Those evaluation settings are not interchangeable.

## Frontier round 1 — categorical identity

[Notebook 13](../notebooks/13_categorical_identity_frontier.ipynb) independently tested native LightGBM categorical identity and identity+frequency representations against both a matched frequency control and the saved champion. Folds 1–3 produced a promising candidate, so `blend_native` was frozen before folds 4–5. Confirmation then produced only +0.000062 mean stability versus the saved champion, with one win and one loss; the predeclared promotion gate failed.

**Decision:** reject the candidate, preserve the frozen release, and stop this direction. The experiment is useful because it removes a credible representation hypothesis from the search space.

## Leading-solution reproduction status

The project now has explicit evidence for relational feature engineering, applicant/related-person separation, tree-family benchmarks, fixed blend selection, robust-history summaries, calibration, and the categorical-identity hypothesis. Important transferable mechanisms are still incomplete; therefore the repository does not claim full reproduction of the strongest public systems.

The highest-value open gaps are:

1. **Previous-application category occurrence histograms.** The 59th-place writeup reports a +0.013 leaderboard gain from counting occurrences of each `applprev_1` category per applicant. The current aggregator keeps non-null count, number of unique values, first/last values and diversity ratio, but not a full per-category occurrence vector. This is a new feature capability, not another encoding variant.
2. **DenseLight neural challenger.** The first-place writeup identifies DenseLight (LightAutoML) as its DNN component and reports that tested transformer-style tabular alternatives did not beat it. The inspected release has no neural challenger. A neural tabular model is valuable only if it supplies complementary errors under the same temporal validation design.
3. **Heterogeneous ensemble after new signal exists.** Existing CatBoost/XGBoost blends did not beat the tuned LightGBM system. Another blend is justified only after a genuinely different feature or architecture family produces complementary out-of-fold predictions.

## Promotion discipline

Post-release candidates must remain development research unless a separately registered procedure and genuinely independent evaluation population justify a new release claim. The observed September holdout cannot be reused as a fresh untouched test.

## Public solution sources

- 59th-place solution: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508202
- First-place solution: https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337
- Final leaderboard: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/leaderboard
