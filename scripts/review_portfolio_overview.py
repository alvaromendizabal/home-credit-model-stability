#!/usr/bin/env python3
"""Render the employer overview from accepted aggregates without fitting models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def review(root: Path) -> None:
    """Keep the benchmark and controlled feature removals in their development scope."""
    benchmark = json.loads((root / "reports/benchmark/review.json").read_text())
    ablation = json.loads((root / "reports/feature_ablation/comparison.json").read_text())
    models = sorted(benchmark["models"], key=lambda row: row["rank"])
    removals = sorted(
        (row for row in ablation["rows"] if row["experiment"] != "control"),
        key=lambda row: row["delta_vs_control"],
    )
    names = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "linear_logistic": "Logistic SGD",
        "without_credit_bureau_a": "Credit bureau A",
        "without_previous_applications": "Previous applications",
        "without_depth2": "Depth-two history",
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "home-credit-portfolio",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#d5dce5",
            "text.color": "#16283c",
            "axes.labelcolor": "#34465b",
            "xtick.color": "#34465b",
            "ytick.color": "#34465b",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
        fig.patch.set_facecolor("#f5f8fc")
        fig.subplots_adjust(left=0.12, right=0.96, top=0.72, bottom=0.23, wspace=0.70)
        scores = [row["mean_fold_stability"] for row in models]
        axes[0].barh(
            [names[row["model"]] for row in models],
            scores,
            color=["#087f8c", "#8299b1", "#8299b1", "#b5645d"],
            height=0.56,
        )
        axes[0].set(xlim=(-0.2, 0.7), xlabel="Mean fold stability (higher is better)")
        axes[0].set_title("Which model transfers best?", loc="left", pad=16, weight="bold")
        axes[0].axvline(0, color="#64748b", linewidth=0.8)
        for index, score in enumerate(scores):
            axes[0].text(score + 0.012, index, f"{score:.4f}", va="center", fontsize=10)
        losses = [-row["delta_vs_control"] for row in removals]
        axes[1].barh(
            [names[row["experiment"]] for row in removals],
            losses,
            color=["#087f8c", "#4d91a2", "#87b1bb"],
            height=0.48,
        )
        axes[1].set(xlim=(0, 0.12), xlabel="Stability lost when the block is removed")
        axes[1].set_title("Which features earn their place?", loc="left", pad=16, weight="bold")
        for index, loss in enumerate(losses):
            axes[1].text(loss + 0.002, index, f"{loss:.4f}", va="center", fontsize=10)
        for ax in axes:
            ax.invert_yaxis()
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e6ebf1", linewidth=0.8)
            ax.tick_params(axis="y", length=0, pad=8)
        fig.text(0.04, 0.91, "HOME CREDIT  /  RESEARCH EVIDENCE", fontsize=11, weight="bold")
        fig.text(0.04, 0.82, "Model choice and feature value are tested separately.", fontsize=19)
        fig.text(
            0.04,
            0.10,
            "Five expanding development folds · 727,187 aligned out-of-fold cases",
            fontsize=11,
        )
        fig.text(
            0.04,
            0.045,
            "Development comparisons; separate from the observed final holdout. "
            "Removal effects are conditional, not causal.",
            fontsize=10,
        )
        destination = root / "reports/portfolio"
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / "overview.svg", metadata={"Date": None})
        fig.savefig(destination / "overview.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    review(Path(__file__).resolve().parents[1])
