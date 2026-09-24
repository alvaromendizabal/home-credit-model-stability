# DenseLight frontier contract

DenseLight is the next registered post-release architecture study. The contract was
frozen **before any DenseLight result existed**. The evaluated September release
remains unchanged.

The study independently recreates the documented DenseLight tabular-neural mechanism
with weighted categorical embeddings, PLR continuous embeddings, a DenseLight-style
feed-forward network, early stopping and SWA. The goal is complementary error
structure versus the frozen tree champion, not another LightGBM tuning sweep.

Selection uses weeks 33–56 (three fits). Only a passing candidate proceeds to weeks
57–72 (two confirmation fits). A single all-label refit is permitted only after
confirmation. Weeks 73–91 are prohibited for selection. The maximum substantive fit
budget is six and there is no artificial wall-clock cutoff.

The committed status is `registered_not_executed`. The readiness check verified
the 700-feature snapshot and prior-probe identities while the inspected CPU
environment lacked CUDA/Torch/LightAutoML, so no DenseLight fit had executed at
publication time.

This separation is intentional: Git contains the scientific contract and compact
reproducibility evidence; GPU checkpoints and borrower-level matrices remain private
runtime artifacts.
