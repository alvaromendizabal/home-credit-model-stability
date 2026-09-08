#!/usr/bin/env python3
"""Run or resume the preregistered raw-history distribution study."""

import argparse
from pathlib import Path

from home_credit.modeling.history_research import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.bucket)
