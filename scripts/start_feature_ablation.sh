#!/usr/bin/env bash
# Run in a child shell; closing a nohup viewer does not terminate this workflow.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs artifacts/feature_ablation
exec 9>artifacts/feature_ablation/launch.lock
if ! flock -n 9; then
    echo 'An ablation launch is already active. Inspect logs/feature-ablation-launch-*.log.' >&2
    exit 1
fi
START_EPOCH="$(date +%s)"
finish() {
    local rc=$?
    printf '[%s] ablation_launch_finished exit_code=%s total_elapsed_seconds=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$(( $(date +%s) - START_EPOCH ))"
}
trap finish EXIT
printf '[%s] ablation_launch_started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash scripts/start_here.sh --require-persistent-storage
export PYTHONUNBUFFERED=1
.venv/bin/python -u scripts/run_feature_ablation.py "$@" --smoke
.venv/bin/python -u scripts/run_feature_ablation.py "$@"
