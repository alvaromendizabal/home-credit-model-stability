#!/usr/bin/env python3
"""Verify saved feature-study models and recompute representative SHAP diagnostics."""

import argparse
from pathlib import Path

from home_credit.modeling.feature_interpretation import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.bucket)
