#!/usr/bin/env bash
# One durable study: quality gates, isolated smoke run, then the bounded full study.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs artifacts/model_tuning
exec 9>artifacts/model_tuning/launch.lock
if ! flock -n 9; then
    echo 'A tuning launch is already active. Inspect logs/model-tuning-launch-*.log.' >&2
    exit 1
fi
START_EPOCH="$(date +%s)"
finish() {
    local rc=$?
    printf '[%s] tuning_launch_finished exit_code=%s total_elapsed_seconds=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$(( $(date +%s) - START_EPOCH ))"
}
trap finish EXIT
printf '[%s] tuning_launch_started new_candidates=8 full_model_fits=40\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash scripts/start_here.sh --require-persistent-storage
export PYTHONUNBUFFERED=1
.venv/bin/python -u scripts/run_model_tuning.py "$@" --smoke
.venv/bin/python -u scripts/run_model_tuning.py "$@"
