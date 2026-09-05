#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
START_EPOCH="$(date +%s)"

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
run() {
    local name="$1"; shift
    local started rc elapsed
    started="$(date +%s)"
    echo "$(stamp) check_started check=$name"
    "$@"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -ne 0 ]; then
        echo "$(stamp) check_failed check=$name exit_code=$rc elapsed_seconds=$elapsed"
        return "$rc"
    fi
    echo "$(stamp) check_passed check=$name elapsed_seconds=$elapsed"
}

run ruff_lint uv run --locked ruff check . || exit $?
run ruff_format uv run --locked ruff format --check . || exit $?
run mypy uv run --locked mypy src || exit $?
run pytest uv run --locked pytest -q --cov=home_credit --cov-report=term-missing || exit $?

TOTAL=$(( $(date +%s) - START_EPOCH ))
echo "$(stamp) checks_completed total_elapsed_seconds=$TOTAL"
exit 0
