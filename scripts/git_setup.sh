#!/usr/bin/env bash
# Safe Git bootstrap for an interactive SageMaker terminal.
# Invoke this script with `bash scripts/git_setup.sh ...`; do not source it.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

GIT_NAME="${1:-}"
GIT_EMAIL="${2:-}"
REMOTE_URL="${3:-}"
PUSH_MODE="${4:-no-push}"
START_EPOCH="$(date +%s)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/git-setup-${STAMP}.log"
mkdir -p logs

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
log() { echo "$(stamp) $*" | tee -a "$LOG_FILE"; }

fail() {
    local reason="$1"
    local code="${2:-1}"
    local elapsed=$(( $(date +%s) - START_EPOCH ))
    log "git_setup_failed reason=$reason exit_code=$code total_elapsed_seconds=$elapsed"
    log "terminal_will_remain_open log_file=$LOG_FILE"
    exit "$code"
}

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ] || [ -z "$REMOTE_URL" ]; then
    log "usage: bash scripts/git_setup.sh <name> <email> <remote_url> [no-push|push]"
    fail "missing_arguments" 2
fi

log "git_setup_started"

if [ ! -d .git ]; then
    git init -b main >>"$LOG_FILE" 2>&1 || fail "git_init"
    log "git_initialized branch=main"
else
    log "git_initialized already_present=true"
fi

git config user.name "$GIT_NAME" || fail "git_user_name"
git config user.email "$GIT_EMAIL" || fail "git_user_email"
log "git_identity_configured name=$GIT_NAME email=$GIT_EMAIL"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL" || fail "git_remote_update"
else
    git remote add origin "$REMOTE_URL" || fail "git_remote_add"
fi
log "git_remote_configured remote=$REMOTE_URL"

REMOTE_MAIN="$(git ls-remote --heads origin main 2>>"$LOG_FILE" || true)"
if [ -n "$REMOTE_MAIN" ] && ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    log "git_setup_stopped reason=remote_main_exists"
    log "remote_main=$REMOTE_MAIN"
    log "No local commit or remote history was modified."
    exit 3
fi

log "quality_gate_started"
bash scripts/check.sh 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    fail "quality_gate" "$RC"
fi
log "quality_gate_passed"

uv run --locked pre-commit install >>"$LOG_FILE" 2>&1 || fail "pre_commit_install"
log "pre_commit_installed"

git add . || fail "git_add"

STAGED_RUNTIME="$(git diff --cached --name-only | grep -E '(^|/)(data|logs|artifacts|\.venv|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)' || true)"
if [ -n "$STAGED_RUNTIME" ]; then
    log "git_setup_failed reason=runtime_content_staged"
    printf '%s\n' "$STAGED_RUNTIME" | tee -a "$LOG_FILE"
    git reset >/dev/null 2>&1 || true
    fail "runtime_content_staged"
fi
log "staged_content_validated"

uv run --locked pre-commit run --all-files 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    fail "pre_commit_gate" "$RC"
fi
log "pre_commit_passed"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    git commit -m "Initialize Home Credit model stability project" 2>&1 | tee -a "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        fail "initial_commit" "$RC"
    fi
    log "initial_commit_created"
elif ! git diff --cached --quiet; then
    log "git_setup_stopped reason=existing_history_with_staged_changes"
    log "Review and commit staged changes manually; no commit was created."
    exit 4
else
    log "commit_skipped reason=existing_clean_history"
fi

COMMIT="$(git rev-parse --short HEAD)"
if [ "$PUSH_MODE" = "push" ]; then
    log "git_push_started remote=origin branch=main"
    git push -u origin main 2>&1 | tee -a "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        fail "git_push" "$RC"
    fi
    log "git_push_completed remote=origin branch=main"
elif [ "$PUSH_MODE" = "no-push" ]; then
    log "git_push_skipped reason=no_push_mode"
    log "next_action=rerun_git_setup_with_push_mode"
else
    fail "invalid_push_mode" 2
fi

TOTAL=$(( $(date +%s) - START_EPOCH ))
log "git_setup_completed branch=main commit=$COMMIT total_elapsed_seconds=$TOTAL"
log "github_remote=$REMOTE_URL"
exit 0
