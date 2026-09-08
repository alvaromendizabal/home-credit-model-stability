#!/usr/bin/env python3
"""Execute the fixed, development-only probability calibration comparison."""

import argparse
from pathlib import Path

from home_credit.modeling.calibration_research import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.bucket)
