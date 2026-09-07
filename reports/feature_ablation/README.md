# Accepted feature ablation

**Decision: retain all 700 features for the next LightGBM tuning study.** Every tested
block removal reduced mean fold stability. The first temporal fold remains the
weakest period and requires attention in later robustness analysis.

![Feature ablation comparison](overview.svg)

| Features | Count | Mean fold stability | Change from control | Worst fold | OOF ROC AUC |
|---|---:|---:|---:|---:|---:|
| All selected features | 700 | 0.585188 | 0.000000 | 0.393682 | 0.846894 |
| Without depth-2 blocks | 651 | 0.569426 | -0.015762 | 0.355247 | 0.846051 |
| Without previous applications | 516 | 0.553748 | -0.031440 | 0.292277 | 0.845742 |
| Without credit bureau A | 427 | 0.491375 | -0.093813 | 0.375419 | 0.821066 |

Four conditions used identical model settings and five expanding development folds.
The source list was frozen before these experiments; removed features were not
replaced. All comparisons use the same 727,187 OOF cases. This experiment supports
keeping the tested blocks in this configuration; it does not establish that every
individual feature helps, identify a causal effect, or measure final-test performance.

The run completed at **2026-09-07 00:15:20 UTC**. Full-study runtime was **8,824.609
seconds (2h 27m 5s)**; the launcher including checks/smoke took **8,971 seconds
(2h 29m 31s)**. All 20 fits and the three notebook code cells completed. S3 persisted
the models, predictions, checkpoint receipts, comparison, notebook and offline HTML.

- [Executed notebook](06_feature_ablation.ipynb): original saved tables and Plotly outputs.
- [Comparison JSON](comparison.json): fold-level and pooled metrics, deltas and provenance.
- [Control manifest](control_manifest.json): immutable hashes used to reuse control predictions.
- [Next tuning runbook](../../docs/model_tuning.md).

The notebook is stored beside its input JSON so it can be rerun from this directory.
The static overview above provides a readable GitHub view without executing JavaScript.
Its source formatting is normalized for repository linting; original executed outputs
are preserved. The original S3 notebook SHA-256 is
`61a42daf5c5a3b50a1a601b20aa4d75082bd772bda1942c4db5c9dc5996a9a26`.

The comparison is copied byte-for-byte from the completed S3 report with SHA-256
`d035dc6993bf83dc9f386c23c7f6612275a13c47053758325e7ad1efcdd853ef`.
Training commit: `e7b53ca1732a362f5ab4eb8fa6f06a53e44ffd09`.
Original control checkpoint manifest SHA-256:
`387d0a32b49ce04abcc8271969ed2ad28886d454bab684aa33b967208cf2fb5d`.

Generate the static overview without AWS or training:

```bash
uv run --locked python scripts/review_feature_ablation.py
```

These metrics use development folds for early stopping and selection. Holdout weeks
73-91 remain locked. Average precision, Brier score and log loss are included in
`comparison.json`; the primary selection metric is mean fold stability.
