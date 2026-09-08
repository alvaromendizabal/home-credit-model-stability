#!/usr/bin/env python3
"""Run the preregistered, post-release development feature comparisons."""

import argparse
from pathlib import Path

from home_credit.modeling.feature_research import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.bucket)
