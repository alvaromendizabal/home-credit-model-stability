#!/usr/bin/env python3
"""Publish the immutable feature and release reviews without fitting models."""

from __future__ import annotations

import argparse
from pathlib import Path

from home_credit.modeling.portfolio_report import review_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    review_portfolio(
        Path(__file__).resolve().parents[1], force=args.force, write_only=args.write_only
    )


if __name__ == "__main__":
    main()
