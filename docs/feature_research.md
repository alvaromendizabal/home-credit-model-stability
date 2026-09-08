# Development feature research after the frozen release

The original holdout was opened before this extension. These are explicitly
exploratory development comparisons. They neither revise its reported score nor
establish a new unbiased final evaluation. The released 700-feature model and its
90/10 blend remain immutable.

The extension enumerates 4,617 additional hypotheses from the verified original
2,508-column cache. Together these form 7,125 candidate representations, not
7,125 independent signals. Selection uses only weeks 0–24 for fitting and 25–32
for early screening; five subsequent development folds cover weeks 33–72.

| Family | Candidates | Question |
|---|---:|---|
| Missingness | 2,179 | Does the absence of a measurement identify a different risk context? |
| Amount ratios | 1,137 | Does recorded activity relative to current credit, price or annuity help? |
| Dispersion | 546 | Are historical ranges and relative variability predictive? |
| Household comparisons | 193 | Does applicant versus related-person context matter? |
| Source-order changes | 182 | Do first/last source records differ meaningfully? |
| Recency | 164 | Does recent activity dominate the longer history? |
| Category interactions | 120 | Do training-fitted joint frequencies improve category representation? |
| Peer statistics | 96 | Do training-only percentiles, category medians and interquartile ranges help? |

Every candidate has a formula, source columns, rationale, stable identifier and
recorded rejection reason. Invalid or nonpositive ratio denominators produce null;
overflow never becomes infinity. Exact duplicates and near-constant columns are
removed using training rows only. Categorical combinations use an unambiguous
encoding. Peer groups require at least 50 training cases; unseen or unsupported
groups produce null. Their reference distributions and frequency maps are saved.

The plan fixes four conditions before execution: 1,400 original features;
700 original plus 256 screened additions; and removal of the added amount-ratio
or peer-statistics families. Each uses the same LightGBM parameters, seeds and
five full development folds as the archived 700-feature control. This is 20 new
fits plus five reused control fits. It tests breadth and major families without
simultaneously retuning the learner.

For the full extension, each fold also measures training gain, native TreeSHAP
contributions on 512 validation cases, three within-week grouped permutations on
12,000 validation cases, and high correlations among the leading training
predictors. SHAP additivity and saved-model prediction parity are asserted.
These diagnostics explain associations; they are not causal or fairness findings.

## Boundaries of the search

No finite search exhausts all possible formulas. This study deliberately covers
the major plausible families and reports its computational budget. It does not
equate feature count with sophistication or promise a gain from more features.
The 1,400-versus-700 comparison directly tests the earlier feature cap.

The cache contains case-level historical means, standard deviations, extrema,
counts and sums. New dispersion measures derive from those records. New peer
quantiles are training-population statistics; they are **not** newly computed
per-customer historical medians, quantiles or skew. Raw-history quantiles and
chronological event trends remain separate hypotheses. `num_group1/2` provide
source order, not proven event-time order, so source differences are never called
momentum or acceleration. Date-window shares use the original application-time
offsets and do not pool future cases.

Native categorical target statistics were already evaluated through CatBoost in
the four-family benchmark. This extension does not introduce online default-rate
histories without outcome-availability timestamps. Ranks, counts, combinations and
peer aggregates contain no target values. Generic text, graph and image embeddings
have no corresponding validated input modality here. External data are not added
without a reproducible and competition-compliant join. Monotonic transforms alone
do not create new feature ordering for trees; the emphasis is on relative and
joint information. Population-shift and fairness questions require separate
validation before any lending deployment.

The categorical leakage rationale is informed by
[Prokhorenkova et al., NeurIPS 2018](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf).
The implementation uses the native contribution interface in
[LightGBM's Booster documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html).
The task and its application-time data scope are described by
[the competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

## Reproducibility

`configs/feature_research.json` freezes the experimental conditions. Every completed
fold publishes its native model, predictions, encoder, peer references and exact
feature list. S3 objects receive full read-back hash verification before a
compare-and-swap ledger advances. A renewable writer lease and local process lock
prevent duplicate workers. Resuming the original source commit reuses verified
completed folds. UTC stage logs and 15-second heartbeats cover long work.

## Observed screening

The full early-window screen completed on 160,000 training and 80,000 later
early-window cases. Of 4,617 candidates, 2,159 survived duplicate/near-constant
pruning, 2,036 passed the additional missingness/cardinality filters, and 256
were retained. There were 934 constants, 1,458 exact duplicates, 66 near-constants,
97 mostly missing variables, 26 high-cardinality combinations and 1,780 eligible
features below the ranking budget. Every rejection appears in the committed catalog.

The selected additions comprise 95 amount ratios, 55 dispersion measures, 50
peer statistics, 39 category interactions, five source-order differences and four
each from household comparisons, recency and missingness. The screened feature
list was persisted before any later development-fold evaluation.

## Interpretation sampling audit

An audit found that the initial SHAP implementation took the first 512 rows after
sorting a larger random sample. With source-ordered cases, that can overweight
earlier observations. `configs/feature_interpretation.json` defines the correction:
sample 512 cases uniformly without replacement from the complete validation fold,
using an independent deterministic seed, and report the sampled week counts.

`scripts/review_feature_interpretation.py` restores the original native models,
encoders and learned peer references. It reconstructs every validation prediction
without fitting a model or refitting a peer map, checks the saved probabilities,
then regenerates SHAP with the corrected sampling. The original diagnostics remain
in the training ledger; the corrected publication links both identities. The
within-week permutation and training-correlation results are preserved.

The original training run must resume at its pinned source commit. A reporting or
interpretation correction must never restart that completed model work under a
different study identity.
