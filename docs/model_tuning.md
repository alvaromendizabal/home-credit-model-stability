# Bounded tuning behind the selected release

The completed study compared eight new LightGBM candidates across five expanding
folds, with five verified control folds reused. **40 new model-fold fits** completed.
The selected `trial_006` improved mean fold stability from **0.5851883724 to
0.6012378499**. It improved three of five folds; most of the gain came from the first.

[Executed analysis](../notebooks/07_model_tuning.ipynb) ·
[Original aggregate metrics](../reports/model_tuning/metrics.json) ·
[Subsequent model selection](model_selection.md)

## Why this comparison matters

All conditions used the selected 700-feature representation and the same expanding
folds. Early stopping used fold AUC; candidate selection used mean fold stability,
with the preregistered secondary metrics and control-preserving tie-break. Each
fold built a fresh training dataset. The eight-trial budget is not evidence of
search convergence, and a joint parameter change is not an isolated causal test.

Weeks 73–91 were withheld **during this study**. The subsequent frozen release
opened them once; they are now observed and cannot be treated as a fresh holdout
for later work. [Release evaluation](model_release.md)

## Reproduce the analysis, not the retired job launch

```bash
uv sync --locked --group dev
uv run --locked python scripts/review_model_tuning.py
```

The review verifies the committed aggregate input and executes the canonical
notebook. It does not launch a GPU, contact AWS, or repeat the historical fits.
The retained native tuning implementation is
[`src/home_credit/modeling/tuning.py`](../src/home_credit/modeling/tuning.py);
its recovery and selection contracts are exercised by the package tests.

The original training commit was `2e52dd908f2b30a783e37157ae9801e0679cc666`.
The original full-study digest was
`0de34e51d0a97bd8cd4064931f81135ba3a5299c30e76b3ca1f178f0cb3569dd`.
These identify the historical experiment, not a currently available cloud service.
The public JSON is an aggregate excerpt, not borrower-level predictions or the
complete private ledger. Project AWS resources have been retired; public review
uses only the repository and the locked environment.

## Final disposition

Tuning, the fixed OOF comparison, development-trained holdout evaluation, and the
all-label inference refit all completed. The later externally evaluated ensemble
is a separate artifact, documented in the [model card](../MODEL_CARD.md). No further
training or unpromoted experiment is part of this review path.
