# Selection behind the frozen release

**15 predeclared candidates · 75 candidate-fold records · no new model fits**

The fixed OOF comparison selected **90% tuned LightGBM and 10% original LightGBM**.
Mean fold stability was **0.6018988903**, compared with **0.6012378499** for the tuned
single model. Four folds improved, but worst-fold stability declined from
**0.479200 to 0.472514**. The small mean improvement is not proof of statistical
significance or robust future performance.

[Executed selection analysis](../notebooks/08_model_selection.ipynb) ·
[Original selection record](../reports/model_selection/selection.json) ·
[Frozen release](model_release.md)

## Selection and data contracts

The comparison used **727,187 development cases** from weeks **33–72**. Candidates
were fixed before scoring: four single-model controls and eleven specified blends.
Weights were not repeatedly optimized against the same outcomes. Selection used
the registered mean-fold objective, secondary metrics, and incumbent-preserving
tie-break. A calibration map was not added to this release.

Every input required unique and complete case IDs, matching targets, weeks and
folds, and finite probabilities in [0, 1]. An inner join could not silently drop
cases. Recomputed incumbent metrics had to match the original tuning study.
The earlier-fold weight diagnostic is not unbiased nested validation: base-model
hyperparameters had already been selected using all development folds.

## Reproduce from the public checkout

```bash
uv sync --locked --group dev
uv run --locked python scripts/review_model_selection.py
```

The command verifies aggregate digests, lineage, all candidate-fold records,
selected weights and population coverage. Add `--force` to execute the four code
cells in a fresh kernel. Unchanged results are reused only after validating the
execution receipt and dependency/output hashes. Failed execution does not replace
the canonical notebook. No raw loan records, cloud account or private model are
needed for this path.

## Retained implementation

[`selection.py`](../src/home_credit/modeling/selection.py) contains the fixed
comparison; [`experiment_store.py`](../src/home_credit/modeling/experiment_store.py)
binds recoverable state to source, configuration, lock and metric identity.
The native code preserves writer exclusion, verified inputs and committed candidate
state. The public tests exercise those contracts without restoring private OOF.
Historical S3 locations in original receipts are provenance, not a claim that the
retired project resources remain available.

## Completed downstream delivery

The selected specification was frozen before evaluating weeks 73–91. That
**development-trained model** and its **all-label inference refit** are separate
artifacts. The original package subsequently completed external evaluation, as
recorded in the [delivery evidence](kaggle_submission.md). The ten-row public-example
parity check alone does not establish hidden-test accuracy or a competition score.
The later externally evaluated ensemble is documented separately in the
[model card](../MODEL_CARD.md); it is not relabeled as this frozen release.
