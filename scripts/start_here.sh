#!/usr/bin/env bash
# Canonical entry point; run with bash, never source into the interactive shell.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec python3 scripts/bootstrap_environment.py "$@"
