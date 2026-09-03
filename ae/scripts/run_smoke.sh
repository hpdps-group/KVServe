#!/usr/bin/env bash
# Backward-compatible alias for the reviewer-facing Track B1 entrypoint.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/run_compression.sh" "$@"
