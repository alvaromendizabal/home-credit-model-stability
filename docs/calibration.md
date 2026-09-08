# Probability calibration across development periods

The reference blend ranks applications well, but ranking does not establish
calibrated default probabilities. This completed experiment asks whether fitting
a calibration map on earlier out-of-fold predictions improves later periods.
The employer-facing evidence is in [notebook 12](../notebooks/12_calibration.ipynb).

## Protocol and interpretation

The study uses the saved 90% tuned / 10% original LightGBM predictions. It fits
sigmoid and isotonic maps on all strictly earlier development folds, then evaluates
folds 2–5, covering weeks 41–72 and 544,611 unique applications. Fold 1 supplies
the initial calibration population. Eight calibrator fits complete the fixed
budget; no base learner is retrained. All methods use identical evaluation cases.

Sigmoid fits logistic regression on clipped log-odds with a positive slope.
Isotonic fits a nondecreasing piecewise-linear mapping. The first preserves
within-fold ranks; isotonic can introduce ties. Because each fold has its own
mapping, pooled discrimination can change even when every within-fold rank is
preserved. The numerical convention clips only log-loss inputs at 1e-7.

This is **post-release exploratory research**, after notebook 09's holdout was
opened. Neither that holdout nor its predictions enter calibration. The base
hyperparameters and blend weights were selected using all development folds,
so this comparison is **not fully nested unbiased validation**. It does not
retroactively alter the release's no-calibration decision or its score.

## Observed results

| Method | Mean fold stability | Worst fold | Pooled Brier | Pooled log loss |
|---|---:|---:|---:|---:|
| Frozen blend | 0.634245 | 0.491566 | 0.033271 | 0.129072 |
| Earlier-fold sigmoid | 0.634245 | 0.491566 | 0.033349 | 0.129347 |
| Earlier-fold isotonic | 0.630754 | 0.486079 | 0.033314 | 0.129474 |

Neither tested calibrator improved pooled Brier score or log loss. Isotonic also
reduced mean stability by 0.003491. These observations argue against adopting
either tested mapping for this reference pipeline. They do not imply all
calibration methods fail, nor establish a production default-probability model.

The notebook includes ROC AUC, average precision, fold-level changes and reliability
curves with bin support. Its four-fold scores must not be compared directly with
the full five-fold model-selection scores or the separate final holdout.
Proper scoring rules such as Brier combine calibration and discrimination effects;
a lower Brier score alone would not isolate improved calibration. See the
[scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

## Durable execution and independent verification

The managed job `home-credit-calibration-research-20260908-0601` completed on
2026-09-08 at 06:03:46 UTC using source
`02a81aa7b29df1011f8c7aa2545fb8030b465a65`, the pinned Python 3.12.14 environment,
and `configs/calibration_research.json`. It published all twelve method/fold
prediction checkpoints and portable JSON parameters with complete S3 read-back
verification. The unchanged second invocation restored them with zero fits.

An independent local verifier then checked both original OOF file hashes, every
saved prediction hash and exact case/label/week alignment. It replayed all
1,633,833 predictions from the saved equations and independently recomputed all
eight fold and pooled metrics. Maximum absolute disagreement was 1.67e-16 for
predictions and 2.00e-15 for metrics. This verification used Python 3.12.13 and
did not fit any model; the exact runtime and verifier source hash are recorded in
`reports/calibration/verification.json`.

`configs/calibration_review.json` pins the public evidence. The canonical notebook
reads aggregate records only and needs no cloud credentials or private data.

```bash
uv run --locked python scripts/review_calibration.py --force
uv run --locked python scripts/review_calibration.py
```

The second review reuses its verified executed output. To independently recheck
downloaded OOF and checkpoint files, use `scripts/verify_calibration_research.py`
with `--inputs` and `--predictions`; it verifies the committed hashes before use.
