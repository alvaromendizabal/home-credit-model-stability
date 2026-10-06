# Public reproducibility and research boundary

This repository is intentionally **semi-reproducible**: a reviewer can inspect the
validation design, reproduce the published review layer, verify aggregate results and
follow the research decisions without receiving borrower-level data, private
checkpoints or the active competition recipe.

That boundary is deliberate. It mirrors a real ML organization where model review,
scientific traceability and engineering quality must remain auditable even when raw
data and production or competitive assets cannot be published.

## Reproducibility matrix

| Layer | Public evidence | Reproducibility level |
|---|---|---|
| Frozen release | Hash-pinned reports, model-release notebook, metrics, manifests and review scripts | **High** — published evidence can be recomputed without retraining |
| Feature system | Candidate accounting, selected-feature counts, ablations and aggregate feature-family evidence | **High for review; partial for raw rebuild** |
| Temporal validation | Fold definitions, promotion discipline, aggregate fold metrics and rejection decisions | **High** |
| Model-family benchmarks | LightGBM, XGBoost, CatBoost, logistic and neural-family comparisons | **High for reported comparisons** |
| Portable inference | Schema checks, raw-feature parity logic, artifact lineage and offline notebook evidence | **High for the published release path** |
| Post-release frontier | Hypotheses, model-family descriptions, aggregate fold outcomes, gates and decisions | **Review-reproducible** |
| Active frontier recipes | Exact private feature identities, current competitive weights, private checkpoints and runtime bundles | **Intentionally private** |
| Borrower-level data | Competition/raw source records and derived private matrices | **Not redistributed** |

## What a reviewer can verify

The repository is designed so review does not require rerunning expensive training.
Committed scripts and notebooks validate the evidence already produced by the
AWS-canonical research system.

Useful entry points:

- [Employer review guide](employer_review.md)
- [Research engineering and reproducibility](research_engineering.md)
- [Model release notebook](../notebooks/09_model_release.ipynb)
- [Post-release research record](post_release_research.md)
- [Model card](../MODEL_CARD.md)

The review layer checks artifact identities, metric calculations, notebook outputs and
published experiment decisions. Expensive private training is not hidden behind a
"run everything" instruction.

## What remains private and why

Private AWS artifacts retain:

- borrower-level feature matrices and labels;
- exact active frontier feature identities;
- trained frontier checkpoints and optimizer state;
- current competitive blend recipes and runtime bundles;
- large intermediate caches and resumable private checkpoints.

Publishing those assets would add little employer-review value while exposing data or
competitive implementation details. Public Git instead exposes the architecture,
validation rules, aggregate evidence, failure/recovery behavior and engineering
contracts needed to assess the work.

## Research traceability

Every substantial private run is designed to bind:

1. source and configuration identity;
2. data and feature identities;
3. fold/model progress;
4. hardware and dependency state;
5. checkpoints and parent/resume lineage;
6. aggregate metrics and promotion decisions;
7. success, rejection, guarded-stop or failure state.

Completed work is reused after interruption. Negative experiments remain part of the
record rather than disappearing from the narrative.

## Current frontier publication policy

The frozen evaluated release is stable and reviewable. Later research is published only
after its evidence has been reconciled. The public record summarizes mechanisms and
aggregate decisions while withholding the exact active competition recipe.

Recent private research has expanded the frontier across efficient neural ensembles,
learned historical representations, nonlinear numerical encodings, retrieval-based
models, self-supervised objectives, source-robust training and categorical-information
recovery. The public record reports what each class of experiment established without
publishing private borrower-level artifacts or live competitive recipes.

## Employer-facing interpretation

This boundary demonstrates a practical engineering skill: making an ML system
reviewable under real constraints.

The project shows how to preserve:

- scientific accountability without publishing restricted data;
- reproducibility without forcing costly retraining;
- recoverability without committing private checkpoints;
- model lineage without exposing an active competitive recipe;
- negative evidence without weakening the clarity of the main portfolio story.

The goal is not to make every private experiment clonable from GitHub. The goal is to
make every published claim traceable, testable at the appropriate layer and explicit
about what remains outside the repository.
