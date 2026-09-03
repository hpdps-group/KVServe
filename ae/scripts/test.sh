#!/usr/bin/env bash
# Backward-compatible alias. Reviewer-facing entrypoint: run_compression.sh.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/run_compression.sh" "$@"
