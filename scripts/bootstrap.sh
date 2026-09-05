#!/usr/bin/env bash
# Deterministic local bootstrap. This script may exit non-zero, but because it is
# invoked with `bash scripts/bootstrap.sh`, it cannot terminate the parent terminal.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
START_EPOCH="$(date +%s)"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-20}"
POLL_SECONDS="${POLL_SECONDS:-1}"

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
log() { echo "$(stamp) $*"; }

run_stage() {
    local name="$1"
    shift
    local started rc elapsed
    started="$(date +%s)"
    log "stage_started stage=$name"
    "$@"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -ne 0 ]; then
        log "stage_failed stage=$name exit_code=$rc elapsed_seconds=$elapsed"
        return "$rc"
    fi
    log "stage_completed stage=$name elapsed_seconds=$elapsed"
    return 0
}

run_long_stage() {
    local name="$1"
    shift
    local started pid rc elapsed
    started="$(date +%s)"
    log "stage_started stage=$name"
    "$@" &
    pid=$!
    local next_heartbeat="$HEARTBEAT_SECONDS"
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_SECONDS"
        elapsed=$(( $(date +%s) - started ))
        if kill -0 "$pid" 2>/dev/null && [ "$elapsed" -ge "$next_heartbeat" ]; then
            log "heartbeat stage=$name elapsed_seconds=$elapsed pid=$pid"
            next_heartbeat=$(( next_heartbeat + HEARTBEAT_SECONDS ))
        fi
    done
    wait "$pid"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -ne 0 ]; then
        log "stage_failed stage=$name exit_code=$rc elapsed_seconds=$elapsed"
        return "$rc"
    fi
    log "stage_completed stage=$name elapsed_seconds=$elapsed"
    return 0
}

log "bootstrap_started"

if ! command -v uv >/dev/null 2>&1; then
    log "bootstrap_failed reason=uv_missing"
    return 1 2>/dev/null || exit 1
fi

if [ -f uv.lock ]; then
    run_stage dependency_lock_check uv lock --check || exit $?
else
    run_long_stage dependency_lock uv lock || exit $?
fi
run_long_stage dependency_sync uv sync --locked --group dev || exit $?
run_stage dependency_integrity uv pip check --python .venv/bin/python || exit $?
run_stage environment_doctor uv run --locked home-credit doctor || exit $?
run_stage dataframe_smoke uv run --locked home-credit dataframe-smoke || exit $?
run_stage model_smoke uv run --locked home-credit model-smoke || exit $?
run_stage metric_smoke uv run --locked home-credit metric-smoke || exit $?
run_stage heartbeat_smoke \
    uv run --locked home-credit heartbeat-smoke --seconds 3 --interval 1 || exit $?
run_long_stage quality_gates bash scripts/check.sh || exit $?

if [ -d .git ]; then
    run_stage pre_commit_install uv run --locked pre-commit install || exit $?
else
    log "stage_skipped stage=pre_commit_install reason=git_repository_not_initialized"
fi

TOTAL=$(( $(date +%s) - START_EPOCH ))
log "bootstrap_completed total_elapsed_seconds=$TOTAL"
exit 0
