#!/usr/bin/env bash
# External-service gate. Project-managed tools run through the locked uv environment.
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
START_EPOCH="$(date +%s)"

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
check() {
    local name="$1"; shift
    local started rc elapsed
    started="$(date +%s)"
    echo "$(stamp) connectivity_started check=$name"
    "$@"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -ne 0 ]; then
        echo "$(stamp) connectivity_failed check=$name exit_code=$rc elapsed_seconds=$elapsed"
        return "$rc"
    fi
    echo "$(stamp) connectivity_passed check=$name elapsed_seconds=$elapsed"
}

check aws_identity aws sts get-caller-identity || exit $?
check kaggle_cli uv run --locked kaggle --version || exit $?
if ! check kaggle_competition \
    uv run --locked kaggle competitions files \
    home-credit-credit-risk-model-stability \
    --page-size 5; then
    echo "$(stamp) connectivity_action check=kaggle_competition"
    echo "$(stamp) next_action='uv run --locked kaggle auth login'"
    exit 1
fi
check git git --version || exit $?

if [ -d .git ]; then
    GIT_NAME="$(git config --get user.name || true)"
    GIT_EMAIL="$(git config --get user.email || true)"
    if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
        echo "$(stamp) connectivity_failed check=git_identity reason=missing_name_or_email"
        echo "$(stamp) next_action='git config user.name <name>; git config user.email <email>'"
        exit 1
    fi
    echo "$(stamp) connectivity_passed check=git_identity name=$GIT_NAME"
fi

if command -v gh >/dev/null 2>&1; then
    check github_auth gh auth status || exit $?
else
    echo "$(stamp) connectivity_skipped check=github_auth reason=gh_not_installed"
fi

TOTAL=$(( $(date +%s) - START_EPOCH ))
echo "$(stamp) connectivity_completed total_elapsed_seconds=$TOTAL"
exit 0
