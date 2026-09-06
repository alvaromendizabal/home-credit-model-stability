#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
START_EPOCH="$(date +%s)"

# Recover a historical /tmp-backed link before uv tries to create .venv.
if [ -L .venv ] || [ ! -x .venv/bin/python ]; then
    if [ "${HOME_CREDIT_BOOTSTRAP_ACTIVE:-0}" = "1" ]; then
        echo "Environment preparation failed; inspect logs/bootstrap-*.log" >&2
        exit 1
    fi
    exec bash scripts/start_here.sh
fi
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$PROJECT_ROOT/artifacts/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/artifacts/runtime/python"
export UV_PYTHON="$(readlink -f .venv/bin/python)"

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
run mypy uv run --locked mypy src scripts/bootstrap_environment.py scripts/review_model_benchmark.py scripts/run_feature_ablation.py || exit $?
run pytest uv run --locked pytest -q --cov=home_credit --cov-report=term-missing || exit $?

TOTAL=$(( $(date +%s) - START_EPOCH ))
echo "$(stamp) checks_completed total_elapsed_seconds=$TOTAL"
exit 0
