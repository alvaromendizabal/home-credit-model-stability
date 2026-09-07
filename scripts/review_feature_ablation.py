#!/usr/bin/env python3
"""Render the accepted aggregate ablation evidence without downloading or training."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from home_credit.modeling.acceptance import read_json, require
from home_credit.modeling.checkpoints import sha256_file
from home_credit.observability.logging import RunLogger
from home_credit.observability.runtime import StageTimer


def main() -> int:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports/feature_ablation"
    source = directory / "comparison.json"
    require(
        sha256_file(source) == "d035dc6993bf83dc9f386c23c7f6612275a13c47053758325e7ad1efcdd853ef",
        "accepted ablation comparison changed",
    )
    result = read_json(source)
    require(
        result["smoke"] is False and result["outer_holdout_touched"] is False,
        "review requires full development evidence",
    )
    logger = RunLogger("ablation-review", root / "logs")
    labels = {
        "control": "All 700 features",
        "without_depth2": "Without depth-2 features",
        "without_previous_applications": "Without previous applications",
        "without_credit_bureau_a": "Without credit bureau A",
    }
    colors = ["#087f8c", "#516dab", "#be873d", "#b45158"]
    with StageTimer(logger, "render_ablation_review", heartbeat_seconds=15):
        plt.rcParams.update(
            {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "home-credit-ablation"}
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]}
        )
        fig.patch.set_facecolor("#f5f8fc")
        rows = result["rows"]
        positions = np.arange(len(rows))
        scores = [r["mean_fold_stability"] for r in rows]
        axes[0].barh(positions, scores, color=colors, height=0.58)
        axes[0].set_yticks(positions, [labels[r["experiment"]] for r in rows])
        axes[0].invert_yaxis()
        axes[0].set(
            xlim=(0, 0.67),
            xlabel="Mean fold stability (higher is better)",
            title="Removing these blocks reduced stability",
        )
        for index, score in enumerate(scores):
            axes[0].text(score + 0.007, index, f"{score:.6f}", va="center", fontsize=10)
        for row, color in zip(rows, colors, strict=True):
            folds = sorted(
                [f for f in result["folds"] if f["experiment"] == row["experiment"]],
                key=lambda f: f["fold"],
            )
            axes[1].plot(
                [f["fold"] for f in folds],
                [f["stability_score"] for f in folds],
                "o-",
                color=color,
                label=labels[row["experiment"]],
                linewidth=2,
            )
        axes[1].set(
            xlabel="Expanding temporal fold",
            ylabel="Official stability score",
            title="The first time window remains the weak point",
        )
        axes[1].set_xticks(range(1, 6))
        axes[1].legend(fontsize=8, frameon=False, loc="lower right")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x" if ax is axes[0] else "y", alpha=0.15)
            ax.set_axisbelow(True)
        fig.suptitle("Home Credit | Retain the full feature set", fontsize=19, fontweight="bold")
        fig.supxlabel(
            "20 completed fits | 727,187 aligned OOF cases | Final holdout remains locked",
            fontsize=11,
            color="#50657a",
        )
        fig.savefig(directory / "overview.svg", metadata={"Date": None})
        fig.savefig(directory / "overview.png", dpi=140)
        plt.close(fig)
    logger.event(
        "ablation_review_completed",
        selected_features=700,
        total_elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
