#!/usr/bin/env python3
"""Track B3 AE: online controller. CPU-only. Reviewer: ./ae/scripts/run_controller.sh"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from kvserve_v1.compression.controller import DynamicOnlineController, gbps_to_mbps

ROOT = Path(__file__).resolve().parents[2]

_KV_SHAPE = {
    "Qwen2.5-7B-Instruct": (28, 4, 128, 2),
}


def _kv_bytes(model: str, input_length: int, kv_mib: float | None) -> int:
    if kv_mib is not None:
        return int(kv_mib * 1024 * 1024)
    shape = _KV_SHAPE.get(model)
    if shape is None:
        raise KeyError(f"no KV shape for {model}; set kv_mib in the config")
    n_layers, n_kv_heads, head_dim, dtype_bytes = shape
    return 2 * n_layers * n_kv_heads * head_dim * input_length * dtype_bytes


def _cfg() -> dict:
    p = ROOT / "configs" / "representative" / "controller.json"
    return json.loads(p.read_text())


def _library_path(model: str, dataset: str, machine: str) -> Path:
    p = ROOT / "profiles" / "libraries" / model / dataset / f"{machine}.json"
    if not p.exists():
        raise FileNotFoundError(f"no library at {p}")
    return p


def _short(pid: str) -> str:
    if pid in ("no_compression", "no_accuracy", "slo_infeasible"):
        return pid
    return pid.rsplit("_", 1)[-1] if "_" in pid else pid


def _select(selector, gbps, acc_req, slo_ms, V_bytes, T_model_ms, input_length):
    return selector.select(
        B_mbps=gbps_to_mbps(gbps),
        acc_req=acc_req,
        slo_ms=slo_ms,
        V_bytes=V_bytes,
        T_model_ms=T_model_ms,
        input_length=input_length,
    )


def _print_choice(d) -> None:
    name = _short(d.profile_id)
    if d.profile is None:
        print(f"    -> {name:<16}  T={d.T_0_ms:.2f} ms")
        return
    print(
        f"    -> {name:<16}  cr={d.cr:.2f}  "
        f"T={d.T_hat_ms:.2f} ms  uncompressed={d.T_0_ms:.2f} ms  ({d.speedup:.2f}x)"
    )


def main() -> int:
    cfg = _cfg()
    model = cfg.get("model_name", "Qwen2.5-7B-Instruct")
    dataset = cfg.get("dataset", "qasper")
    machine = cfg.get("machine", "5090")
    acc_req = float(cfg["constraints"]["acc_req"])
    slo_ms = float(cfg["constraints"]["slo_ms"])
    T_model_ms = float(cfg.get("t_model_ms", 5))
    input_length = int(cfg.get("input_length", 1024))
    bandwidths = [c["gbps"] for c in cfg["bandwidth_cases"]]
    kv_mib_cfg = cfg.get("kv_mib")
    V_bytes = _kv_bytes(model, input_length, float(kv_mib_cfg) if kv_mib_cfg is not None else None)
    V_mib = V_bytes / (1024 * 1024)
    kw = dict(
        acc_req=acc_req, slo_ms=slo_ms, V_bytes=V_bytes,
        T_model_ms=T_model_ms, input_length=input_length,
    )

    lib_path = _library_path(model, dataset, machine)
    selector = DynamicOnlineController.from_library_json(
        lib_path, machine=machine, model=model, eta=0.3, epsilon=0.0,
    )
    t0 = time.perf_counter()

    print("KVServe AE Track B3 — Online controller")
    print()
    print("INPUT")
    print(f"  model        {model}")
    print(f"  dataset      {dataset}")
    print(f"  machine      {machine}")
    print(f"  acc_req      {acc_req:.2f}")
    print(f"  SLO          {slo_ms:.0f} ms")
    print(f"  KV volume    {V_mib:.0f} MiB  (fp16 KV at {input_length} tokens)")
    print(f"  T_model      {T_model_ms:.0f} ms  (decode-side compute on the PD path)")
    print(f"  input_length {input_length} tokens")
    print()
    print("OUTPUT")

    checks = [("library", len(selector.profiles) > 0)]
    decisions = []
    for gbps in bandwidths:
        print(f"  B={gbps} Gbps  machine={machine}  acc_req={acc_req:.2f}  SLO={slo_ms:.0f}ms")
        d = _select(selector, gbps, **kw)
        _print_choice(d)
        decisions.append((gbps, d))

    low = decisions[0][1]
    high = decisions[-1][1]
    checks.append(("low-B compresses", low.profile is not None and low.speedup > 1.0))
    checks.append(("high-B off", high.profile is None))

    d0 = _select(selector, 25, **kw)
    if d0.profile is None:
        print("  B=25 Gbps  observe slower-than-predicted latency")
        print("    -> skipped (no compression at 25 Gbps)")
        checks.append(("residual switch", False))
    else:
        victim = d0.profile.profile_id
        for _ in range(6):
            d_inj = _select(selector, 25, **kw)
            if d_inj.profile is not None and d_inj.profile.profile_id == victim:
                selector.observe(d_inj, T_obs_ms=4.0 * d_inj.T_hat_ms, slo_ms=slo_ms)
        d1 = _select(selector, 25, **kw)
        print(f"  B=25 Gbps  observe 4x predicted latency on {_short(victim)}")
        _print_choice(d1)
        checks.append(("residual switch", d1.profile_id != victim))

    elapsed = time.perf_counter() - t0
    checks.append(("runtime", elapsed < 180.0))

    failed = [name for name, ok in checks if not ok]
    print()
    if failed:
        print(f"KVServe AE Track B3: FAILED ({', '.join(failed)})")
        return 1
    print(f"KVServe AE Track B3: PASSED  ({elapsed:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
