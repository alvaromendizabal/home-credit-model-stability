#!/usr/bin/env python3
"""Execute the canonical feature review from pinned aggregate research evidence."""

import argparse
from pathlib import Path

from home_credit.modeling.feature_research_report import review

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    review(Path(__file__).resolve().parents[1], force=args.force, write_only=args.write_only)
