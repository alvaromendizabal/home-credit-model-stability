# Completed calibration evidence

Start with [the executed notebook](../../notebooks/12_calibration.ipynb) and
[the protocol and interpretation](../../docs/calibration.md).

- `comparison.json`: all twelve method/fold records, portable parameters, pooled
  metrics and reliability-bin support; byte identity pinned in the review config.
- `study.json`: completed S3 ledger and content-addressed checkpoint references.
- `verification.json`: independent input, population, prediction and metric checks.
- `publication.json`: executed notebook identity and reproducibility receipt.

Only aggregate evidence is committed. Borrower-level predictions remain in the
verified private artifact store. This study uses development periods only and
does not change the released model or its already-observed holdout evaluation.
