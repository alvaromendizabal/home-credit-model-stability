# Research engineering and reproducibility

This repository is designed to show more than model fitting. The research system treats
temporal validation, artifact identity, recoverability, hardware utilization and negative
results as first-class engineering requirements.

## Evaluation architecture

All post-release experiments use a fixed temporal promotion lifecycle:

1. **Selection** on earlier development periods.
2. **Confirmation** on later development periods that are not used to choose the candidate.
3. **All-label refit** only after the frozen candidate clears the confirmation gate.
4. **Portable inference parity** before an external evaluation artifact is produced.

Weeks 73–91 belong to the already-observed historical release evaluation and are not reused
as a fresh model-selection population.

Promotion is intentionally stricter than "better average score." A challenger must improve
the official weekly-Gini stability metric consistently across the required folds while
preserving ranking quality. This has rejected several candidates that looked strong on
earlier periods but did not transfer to later ones.

## Experiment system

Long-running research is executed from AWS/SageMaker with bounded, resumable runners.
The private runtime layer records:

- immutable run identifiers and parent/child resume lineage;
- source, data and configuration hashes;
- fold/model progress and completed/remaining work;
- CPU, RAM, GPU, GPU-memory and disk telemetry;
- heartbeat-level compute-cost estimates;
- checkpoint and receipt identities;
- explicit promotion, rejection, timeout and failure states.

A completed model fold is reused rather than retrained after a later-stage interruption.
Avoidable failures become regression tests before the next expensive run.

## Hardware-aware execution

The active research instance is an NVIDIA L4-backed SageMaker environment. Performance
choices are measured rather than assumed. Examples include:

- DenseLight inference batch benchmarking and CPU/GPU probability parity;
- CatBoost feed-thread benchmarking;
- LightGBM thread-count benchmarking to avoid nested oversubscription;
- explicit effective-device checks for XGBoost, rather than trusting a requested CUDA flag;
- concurrent CPU fits only when host-memory and thread budgets make that safe.

This makes resource telemetry part of correctness: a job that silently falls back to a
different execution path is treated as a scientific and engineering issue, not merely a
speed problem.

## Runtime isolation and recovery hardening

Later GPU research made runtime identity an explicit part of the experiment contract.
Tree-model and neural workloads may use different pre-existing interpreters on the same
SageMaker instance, but the runner records and validates each interpreter, package stack,
CUDA capability and artifact handoff before scientific work begins.

The recovery system also distinguishes completed training from completed evaluation.
For example, a saved booster or neural checkpoint can be reused after a downstream
validation or packaging interruption without silently retraining the completed model.
Checkpoint receipts bind model state, optimizer or boosting progress, random-state
lineage and source identities. Smoke-test state is kept separate from substantive
model state.

These controls are private-runtime engineering details. Public Git records the system
contract and aggregate outcomes, not the active competition checkpoints or exact
recovery recipes.

## Model and representation diversity

The research program has exercised complementary model families and representations,
including:

- LightGBM with multiple feature systems and boosting modes;
- CatBoost with recovered categorical identity;
- reduced-sample XGBoost;
- DenseLight-style neural modeling;
- HistGradientBoosting;
- fixed and learned heterogeneous ensembles;
- relational, sparse, temporal, categorical and historical aggregation families.

The public repository records aggregate evidence and lifecycle decisions. Exact active
feature lists, borrower-level matrices, private checkpoints and portable competition
runtime bundles remain in private AWS artifacts.

## Temporal-shift diagnostics

The research repeatedly tests whether apparent development gains survive later periods.
An adversarial early-vs-late period classifier reached approximately **0.998 AUC**,
quantifying substantial covariate shift. That diagnostic motivated controlled experiments
with recency weighting, stability-aware training, adversarial feature pruning and
heterogeneous stacking.

Several of those experiments produced strong early-period gains and were still rejected
because the gains did not persist on confirmation periods. Retaining those negative
results is intentional: the repository demonstrates decision quality and reproducible
evidence, not only successful model runs.

## Publication boundary

Public:
- validation design;
- aggregate fold-level outcomes;
- model-family descriptions;
- promotion/rejection decisions;
- reproducibility contracts;
- engineering lessons.

Private:
- exact active feature identities;
- borrower-level matrices;
- trained checkpoints;
- private inference bundles;
- competition-specific implementation details.

This boundary keeps the project technically reviewable and semi-reproducible while
preserving private research assets.
