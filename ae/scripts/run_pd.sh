#!/usr/bin/env bash
# Track C1 — cross-machine PD. Same reviewer path as other tracks:
#   docker exec -it sigcomm-ae bash
#   cd /workspace && ./ae/scripts/run_pd.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

if ! command -v ssh >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-client >/dev/null
  else
    echo "[AE] ssh is required to start decode on SL3061." >&2
    exit 2
  fi
fi
if [[ -f /.dockerenv ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  if [[ ! -f /root/.ssh/id_rsa && -f /host-ssh/id_rsa ]]; then
    cp /host-ssh/id_rsa /root/.ssh/id_rsa
    chmod 600 /root/.ssh/id_rsa
  fi
fi

if [[ -x /opt/kvs-venv/bin/python ]]; then
  PY=/opt/kvs-venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
exec "$PY" ae/scripts/run_pd.py "$@"
