#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STARTED="$(date +%s)"
printf '[%s] model_selection_launch_started new_model_fits=0 candidates=15\n' "$(date -u +%FT%TZ)"
bash scripts/start_here.sh --require-persistent-storage
.venv/bin/python scripts/run_model_selection.py "$@"
printf '[%s] model_selection_launch_completed total_elapsed_seconds=%s\n' "$(date -u +%FT%TZ)" "$(( $(date +%s) - STARTED ))"
