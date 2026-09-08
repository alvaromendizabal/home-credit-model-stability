#!/usr/bin/env python3
"""Run the frozen model release; leave submission export to the notebook owner."""

from home_credit.modeling.release_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
