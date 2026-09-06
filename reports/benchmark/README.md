# Home Credit temporal model benchmark

Development artifact acceptance passed. Final holdout evaluation is pending.

![Benchmark overview](overview.svg)

Open `report.html` in a browser for interactive charts. All chart assets are embedded.

| Model | Mean fold stability | Worst fold | OOF AUC | OOF AP | OOF Brier |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.585188 | 0.393682 | 0.846894 | 0.204525 | 0.032425 |
| XGBoost | 0.558892 | 0.343432 | 0.846576 | 0.204701 | 0.032441 |
| CatBoost | 0.546330 | 0.297149 | 0.842745 | 0.201164 | 0.032512 |
| Logistic SGD | -0.120484 | -0.586889 | 0.675897 | 0.102612 | 0.036354 |

Verified 70 files and 20 model folds; 727,187 OOF cases per model.

## Interpretation

- Development folds were used for early stopping and model selection; scores are not an unbiased final test.
- Pooled OOF stability spans predictions from five refitted models; mean fold stability remains the selection objective.
- Artifact acceptance verifies recorded predictions and provenance, not raw-data point-in-time correctness or every historical process.
- The adversarial screening AUC is a shift diagnostic from the screening sample; it is not the credit-risk model AUC.
- The constant Brier reference uses OOF prevalence descriptively and is not a trained baseline.
- Final holdout evaluation, model calibration, subgroup robustness, and deployment readiness remain pending.

## Next experiment

Inspect weak-fold behavior and temporal shift, then run LightGBM/XGBoost tuning and feature-block ablations. Keep weeks 73-91 locked.

Training commit: `414397e1f5731aa969b8599d5c11a58459a0e361`.

Summary SHA-256: `cb87d1ce0abd7419a115e3b47aa1d332aad27f4d476a36a0028c2c7b2af183f4`.
