# Project status and completion gates

The authoritative evidence is the committed, hash-pinned experiment record, not an
unchecked task list or a subjective project rating.

| Stage | Verified state | Evidence |
|---|---|---|
| Raw data and feature cache | 34 persisted train/test blocks, 17 groups | `reports/feature_ablation/feature_manifest.json` |
| Feature screening | 2,508 candidates, 2,034 eligible, 700 retained | Notebook 02 and original feature screen |
| Temporal benchmark | Four families, five expanding folds, 727,187 OOF cases | Notebook 05 |
| Family ablation | Control plus three removals, five folds each | Notebook 06 |
| LightGBM tuning | Eight candidates, 40 new fits | Notebook 07 |
| Ensemble selection | 15 fixed candidates, no new fits | Notebook 08 |
| Frozen final evaluation | 203,345 cases, weeks 73-91 | Notebook 09 |
| All-label refit | Two native models, 1,526,659 cases | Immutable release state |
| Raw inference | Ten public example cases; durable batch reuse | Release verification record |
| Submission controls | Owner-enabled notebook; no automatic CSV/upload | Notebook 10 |
| Expanded feature research | Not exhaustive; untested families and budget comparisons remain | Notebook 02 research boundary |
| Hidden Kaggle execution | Not run; no leaderboard result | Explicit inference limitation |
| Production lending validation | Outside research-portfolio scope | README research scope |

The release fixes 700 features, 90/10 LightGBM weights, no calibration and 1,852/1,355
iterations before its holdout evaluation. Preserve that evidence. Further experiments
must not re-label the observed weeks as untouched or tune against those results.

Employer review starts with notebooks 02, 05 and 09; notebooks 06-08 contain the
supporting experimental detail. Notebook 10 leaves submission generation to the owner.
