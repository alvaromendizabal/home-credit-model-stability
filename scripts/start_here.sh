#!/usr/bin/env bash
# Safe interactive entry point. Run with: bash scripts/start_here.sh
# Do NOT source this script.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/bootstrap-${STAMP}.log"
START_EPOCH="$(date +%s)"
EXPECTED_UV_VERSION="0.12.10"

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
log() { echo "$(stamp) $*" | tee -a "$LOG_FILE"; }

log "start_here_started project_root=$PROJECT_ROOT"
log "log_file=$LOG_FILE"

log "source_validation_started"
python3 scripts/source_check.py 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    log "start_here_failed stage=source_validation exit_code=$RC"
    log "terminal_will_remain_open log_file=$LOG_FILE"
    exit "$RC"
fi
log "source_validation_passed"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/home-credit-uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
VENV_TARGET="${HOME_CREDIT_VENV_DIR:-/tmp/home-credit-model-stability-venv}"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$VENV_TARGET")"
CURRENT_UV_VERSION="$(uv --version 2>/dev/null | awk '{print $2}' || true)"
if [ "$CURRENT_UV_VERSION" != "$EXPECTED_UV_VERSION" ]; then
    log "installing_pinned_uv expected_version=$EXPECTED_UV_VERSION"
    log "installing_pinned_uv current_version=${CURRENT_UV_VERSION:-missing}"
    python3 -m pip install --user --upgrade "uv==$EXPECTED_UV_VERSION" 2>&1 | tee -a "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        log "start_here_failed stage=install_uv exit_code=$RC"
        exit "$RC"
    fi
fi

UV_VERSION="$(uv --version 2>/dev/null || true)"
log "uv_ready version=$UV_VERSION"

# The SageMaker home filesystem can remain small even when the Studio space UI
# reports a larger storage setting. Keep the reproducible virtual environment on
# ephemeral instance storage and expose it through the conventional .venv path.
if [ -e .venv ] && [ ! -L .venv ]; then
    log "project_venv_relocating source=$PROJECT_ROOT/.venv target=$VENV_TARGET"
    rm -rf .venv
fi
if [ -L .venv ]; then
    CURRENT_VENV_TARGET="$(readlink -f .venv 2>/dev/null || true)"
    if [ "$CURRENT_VENV_TARGET" != "$VENV_TARGET" ]; then
        rm -f .venv
    fi
fi
if [ ! -x "$VENV_TARGET/bin/python" ]; then
    rm -rf "$VENV_TARGET"
    uv venv "$VENV_TARGET" --python /opt/conda/bin/python3 2>&1 | tee -a "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        log "start_here_failed stage=create_project_venv exit_code=$RC"
        exit "$RC"
    fi
fi
if [ ! -L .venv ]; then
    ln -s "$VENV_TARGET" .venv
fi
log "project_venv_ready path=$VENV_TARGET link=$PROJECT_ROOT/.venv"

bash scripts/bootstrap.sh 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START_EPOCH ))

if [ "$RC" -ne 0 ]; then
    log "start_here_failed exit_code=$RC total_elapsed_seconds=$ELAPSED"
    log "terminal_will_remain_open rerun_after_reviewing=$LOG_FILE"
    exit "$RC"
fi

log "start_here_completed total_elapsed_seconds=$ELAPSED"
log "NEXT: run bash scripts/connectivity_check.sh"
exit 0
