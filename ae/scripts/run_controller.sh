#!/usr/bin/env bash
# Track B3 — online controller. CPU-only. Reviewer: ./ae/scripts/run_controller.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
if [[ -x /opt/kvs-venv/bin/python ]]; then
  PY=/opt/kvs-venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
exec "$PY" ae/scripts/run_controller.py "$@"
