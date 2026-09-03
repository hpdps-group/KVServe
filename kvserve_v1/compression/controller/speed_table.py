"""Component throughput CSV → pipeline harmonic S (MB/s)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kvserve_v1.compression.controller.analytical_model import GB_TO_MB

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEED_DIR = _REPO_ROOT / "profiles" / "raw" / "main" / "speed_results"


def speed_csv_for(machine: str, model: str) -> Optional[Path]:
    names = [
        f"{machine}_{model}_speed.csv",
        f"{machine}-{model}_speed.csv",
        f"{machine}_{model}_speed(2).csv",
    ]
    for name in names:
        p = _SPEED_DIR / name
        if p.exists():
            return p
    matches = sorted(_SPEED_DIR.glob(f"{machine}*{model}*speed*.csv"))
    return matches[0] if matches else None


class SpeedTable:
    """Look up encode/decode GB/s by (component, input_length), then pipeline harmonic."""

    def __init__(self, machine: str, model: str):
        path = speed_csv_for(machine, model)
        if path is None:
            raise FileNotFoundError(f"no speed CSV for machine={machine} model={model}")
        self.machine = machine
        self.model = model
        self.path = path
        self._gbps: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self._pipe_cache: Dict[Tuple[Tuple[str, ...], int], Optional[float]] = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    length = int(float(row["input_length"]))
                    pre = float(row["prefill_throughput(GB/s)"])
                    dec = float(row["decode_throughput(GB/s)"])
                except (TypeError, ValueError):
                    continue
                ctype = row["type"].strip().lower()
                self._gbps.setdefault(ctype, {})[length] = (pre, dec)

    def _component_gbps(self, ctype: str, input_length: int) -> Optional[float]:
        data = self._gbps.get(ctype.lower())
        if not data:
            return None
        nearest = min(data, key=lambda L: (abs(L - input_length), L))
        pre, dec = data[nearest]
        if pre > 0 and dec > 0:
            return 2.0 / (1.0 / pre + 1.0 / dec)
        return pre or dec or None

    def pipeline_mbs(self, components: List[str], input_length: int) -> Optional[float]:
        key = (tuple(c.lower() for c in components), int(input_length))
        if key in self._pipe_cache:
            return self._pipe_cache[key]
        inv = 0.0
        for c in components:
            s = self._component_gbps(c, input_length)
            if s is None or s <= 0:
                self._pipe_cache[key] = None
                return None
            inv += 1.0 / s
        value = None if inv <= 0 else (1.0 / inv) * GB_TO_MB
        self._pipe_cache[key] = value
        return value
