#!/usr/bin/env python3
"""Track C1 AE: cross-machine PD, none vs default.

Two PD pairs (GPU0 and GPU1) share the 50 Gb/s NIC. 10 HotpotQA
requests each, 20 total. Run from the SL3060 host.

Reviewer:
    ./ae/scripts/run_pd.sh
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    return json.loads((ROOT / "configs" / "representative" / "pd.json").read_text())


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=check)


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def _ssh(cfg: dict, remote: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = (
        "ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/tmp/kvserve_known_hosts "
        f"{shlex.quote(cfg['decode_ssh'])} {shlex.quote(remote)}"
    )
    return _run(cmd, check=check)


def _pairs(cfg: dict) -> int:
    return max(1, int(cfg.get("parallel_pairs", 1)))


def _pair_ports(cfg: dict, pair: int) -> tuple[int, int]:
    return int(cfg["kv_port"]) + 10 * pair, int(cfg["sync_port"]) + 10 * pair


def _inner_cmd(cfg: dict, role: str, mode: str, pair: int, log_rel: str) -> str:
    kv_port, sync_port = _pair_ports(cfg, pair)
    label = f"{mode}-p{pair}"
    gpu = str(cfg["gpus"][role]).split(",")[pair].strip()
    args = [
        "/opt/kvs-venv/bin/python", "tests/test_kvserve_remote.py",
        "--role", role,
        "--model", cfg["model"],
        "--gpus", gpu,
        "--kv-ip", cfg["decode_host"],
        "--kv-port", str(kv_port),
        "--sync-port", str(sync_port),
        "--sync-timeout-s", str(float(cfg["sync_timeout_s"])),
        "--ib-device", cfg["ib_device"],
        "--gpu-mem-util", str(float(cfg["gpu_mem_util"])),
        "--max-model-len", str(int(cfg["max_model_len"])),
        "--max-prompt-chars", str(int(cfg.get("max_prompt_chars", 0))),
        "--max-num-batched-tokens", str(int(cfg["max_num_batched_tokens"])),
        "--max-num-seqs", str(int(cfg["max_num_seqs"])),
        "--kv-buffer-gib", str(float(cfg["kv_buffer_gib"])),
        "--num-requests", str(int(cfg["num_requests"])),
        "--warmup-requests", str(int(cfg["warmup_requests"])),
        "--max-tokens", str(int(cfg["max_tokens"])),
        "--data-path", "/workspace/" + cfg["dataset"],
        "--mode", mode,
        "--run-label", label,
        "--output-dir", "/workspace/results/pd",
        "--transfer-prefix", f"pd-{label}",
    ]
    env = (
        "export PYTHONPATH=/workspace:/workspace/tests PYTHONUNBUFFERED=1 "
        "NCCL_IB_HCA=mlx5_0 NCCL_SOCKET_IFNAME=eno1np0; "
        "cd /workspace; "
    )
    quoted = " ".join(shlex.quote(str(a)) for a in args)
    return env + quoted + f" > /workspace/results/pd/{log_rel} 2>&1"


def _start_decode(cfg: dict, mode: str, pair: int) -> None:
    inner = _inner_cmd(cfg, "decode", mode, pair, f"decode_{mode}_p{pair}.log")
    ctn = cfg["decode_container"]
    _ssh(cfg, "mkdir -p /data/ubuntu/KVServe/results/pd")
    _ssh(
        cfg,
        f"docker exec -d {shlex.quote(ctn)} bash -lc {shlex.quote(inner)}",
    )


def _start_prefill(cfg: dict, mode: str, pair: int) -> subprocess.Popen:
    inner = _inner_cmd(cfg, "prefill", mode, pair, f"prefill_{mode}_p{pair}.log")
    Path(ROOT / "results" / "pd").mkdir(parents=True, exist_ok=True)
    if _in_container():
        return subprocess.Popen(
            ["bash", "-lc", inner],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    ctn = cfg["prefill_container"]
    return subprocess.Popen(
        ["docker", "exec", ctn, "bash", "-lc", inner],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_log(path_cmd: str, needle: str, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        proc = _run(path_cmd, check=False)
        if needle in (proc.stdout or ""):
            return
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {needle!r}")


def _wait_file_ssh(cfg: dict, path: str, needle: str, timeout_s: float) -> None:
    remote = "cat " + shlex.quote(path) + " 2>/dev/null || true"
    cmd = (
        "ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/tmp/kvserve_known_hosts "
        f"{shlex.quote(cfg['decode_ssh'])} {shlex.quote(remote)}"
    )
    _wait_log(cmd, needle, timeout_s)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _payload_mib(stats_path: Path) -> float:
    total = 0
    if not stats_path.exists():
        return 0.0
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("transfer_id", "")).startswith("warmup"):
            continue
        total += int(row.get("payload_bytes") or row.get("compressed_bytes") or 0)
    return total / (1024 ** 2)


def _compression_cr(stats_path: Path) -> float:
    raw = 0
    comp = 0
    if not stats_path.exists():
        return float("nan")
    by_tid: dict[str, dict] = {}
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row.get("transfer_id", ""))
        if tid.startswith("warmup"):
            continue
        cur = by_tid.setdefault(tid, {})
        for k, v in row.items():
            if v is None:
                continue
            if k in ("original_bytes", "compressed_bytes") and not v:
                continue
            cur[k] = v
    crs = []
    for row in by_tid.values():
        a = float(row.get("original_bytes") or 0)
        b = float(row.get("compressed_bytes") or 0)
        if a > 0 and b > 0:
            crs.append(a / b)
            raw += a
            comp += b
    return _mean(crs)


_KILL_INNER = (
    "ps -eo pid,cmd | awk '/tests\\/test_kvserve_remote\\.py/ {print $1}' "
    "| while read p; do kill -9 $p 2>/dev/null || true; done; "
    "ps -eo pid,cmd | awk '/VLLM::EngineCor/ {print $1}' "
    "| while read p; do kill -9 $p 2>/dev/null || true; done"
)


def _kill_role(cfg: dict, where: str) -> None:
    if where == "prefill":
        if _in_container():
            _run("bash -lc " + shlex.quote(_KILL_INNER), check=False)
        else:
            _run(
                "docker exec -u root "
                + shlex.quote(cfg["prefill_container"])
                + " bash -lc "
                + shlex.quote(_KILL_INNER),
                check=False,
            )
        return
    _ssh(
        cfg,
        "docker exec -u root "
        + shlex.quote(cfg["decode_container"])
        + " bash -lc "
        + shlex.quote(_KILL_INNER),
        check=False,
    )


def _collect_pair(cfg: dict, mode: str, pair: int) -> dict:
    label = f"{mode}-p{pair}"
    bench = {}
    pulled = _ssh(
        cfg,
        f"cat /data/ubuntu/KVServe/results/pd/benchmark_remote_{label}_decode.json 2>/dev/null || true",
        check=False,
    ).stdout.strip()
    if pulled:
        bench = json.loads(pulled)
        (ROOT / "results" / "pd" / f"benchmark_remote_{label}_decode.json").write_text(
            pulled + "\n"
        )
    if not bench:
        local_prefill = ROOT / "results" / "pd" / f"benchmark_remote_{label}_prefill.json"
        if local_prefill.exists():
            bench = json.loads(local_prefill.read_text())
    wall = float(bench.get("measured_job_time_s") or 0.0)
    transport = bench.get("transport") or {}
    payload = float(transport.get("payload_bytes") or 0) / (1024 ** 2)
    if payload <= 0:
        payload = _payload_mib(
            ROOT / "results" / "pd" / f"transport_stats_remote_{label}_prefill.jsonl"
        )
    cr = _compression_cr(
        ROOT / "results" / "pd" / f"compression_stats_remote_{label}.jsonl"
    ) if mode != "none" else 1.0
    return {"pair": pair, "measured_s": wall, "payload_mib": payload, "compression_ratio": cr}


def _run_mode(cfg: dict, mode: str) -> dict:
    n = _pairs(cfg)
    print(f"  starting {mode}: {n} decode(s) on 3061, then {n} prefill(s) together...", flush=True)
    _kill_role(cfg, "prefill")
    _kill_role(cfg, "decode")
    time.sleep(2)
    for pair in range(n):
        for p in (
            ROOT / "results" / "pd" / f"prefill_{mode}_p{pair}.log",
            ROOT / "results" / "pd" / f"decode_{mode}_p{pair}.log",
        ):
            if p.exists():
                p.unlink()
        _ssh(
            cfg,
            f"rm -f /data/ubuntu/KVServe/results/pd/decode_{mode}_p{pair}.log "
            f"/data/ubuntu/KVServe/results/pd/prefill_{mode}_p{pair}.log",
            check=False,
        )

    for pair in range(n):
        _start_decode(cfg, mode, pair)
    for pair in range(n):
        _wait_file_ssh(
            cfg,
            f"/data/ubuntu/KVServe/results/pd/decode_{mode}_p{pair}.log",
            "[Remote] engine_ready role=decode",
            timeout_s=300,
        )
    print(f"  decode {mode} ready (pairs={n})", flush=True)

    prefills = [_start_prefill(cfg, mode, pair) for pair in range(n)]
    for pair in range(n):
        _wait_log(
            f"cat {shlex.quote(str(ROOT / 'results' / 'pd' / f'prefill_{mode}_p{pair}.log'))} 2>/dev/null || true",
            "[Remote] engine_ready role=prefill",
            timeout_s=300,
        )
    print(f"  prefill {mode} ready; waiting for {n * int(cfg['num_requests'])} requests...", flush=True)

    rcs = [p.wait(timeout=int(cfg["sync_timeout_s"])) for p in prefills]
    t0 = time.time()
    pending = set(range(n))
    while pending and time.time() - t0 < 180:
        done = set()
        for pair in pending:
            tail = _ssh(
                cfg,
                f"tail -n 30 /data/ubuntu/KVServe/results/pd/decode_{mode}_p{pair}.log",
                check=False,
            ).stdout
            if "[Benchmark] role=decode" in tail or "Traceback" in tail:
                done.add(pair)
        pending -= done
        if pending:
            time.sleep(3)
    time.sleep(2)

    pair_rows = [_collect_pair(cfg, mode, pair) for pair in range(n)]
    walls = [r["measured_s"] for r in pair_rows]
    payloads = [r["payload_mib"] for r in pair_rows]
    crs = [r["compression_ratio"] for r in pair_rows]
    return {
        "mode": mode,
        "prefill_rc": rcs,
        "measured_s": max(walls) if walls else 0.0,
        "mean_s": _mean(walls),
        "payload_mib": sum(payloads),
        "compression_ratio": _mean(crs),
        "pairs": pair_rows,
    }


def main() -> int:
    cfg = _cfg()
    (ROOT / "results" / "pd").mkdir(parents=True, exist_ok=True)

    print("KVServe AE Track C1 — Cross-machine PD")
    print()
    print("INPUT")
    n = _pairs(cfg)
    total = n * int(cfg["num_requests"])
    print(f"  prefill        SL3060  gpu={cfg['gpus']['prefill']}  container={cfg['prefill_container']}")
    print(f"  decode         SL3061  gpu={cfg['gpus']['decode']}  host={cfg['decode_host']}")
    print(f"  pairs          {n} concurrent PD streams, start together")
    print(f"  dataset        HotpotQA  ({cfg['dataset']}, {cfg['num_requests']} req/pair, total={total})")
    print(f"  modes          {', '.join(cfg['modes'])}")
    print(f"  max_model_len  {cfg['max_model_len']}  (char_cap={cfg.get('max_prompt_chars', 0)})")
    print(f"  interconnect   NCCL over mlx5_0 (50 Gb/s), shared by {n} stream(s)")
    print()
    print("OUTPUT", flush=True)

    results = []
    for mode in cfg["modes"]:
        results.append(_run_mode(cfg, mode))
        print(
            f"  {mode:<8}  makespan={results[-1]['measured_s']:.2f}s  "
            f"mean={results[-1]['mean_s']:.2f}s  "
            f"payload={results[-1]['payload_mib']:.1f} MiB  "
            f"cr={results[-1]['compression_ratio']:.2f}x",
            flush=True,
        )
        _kill_role(cfg, "prefill")
        _kill_role(cfg, "decode")
        time.sleep(3)

    by = {r["mode"]: r for r in results}
    none, default = by.get("none"), by.get("default")
    print()

    def _complete(row):
        if not row:
            return False
        rcs = row.get("prefill_rc") or []
        return (
            float(row.get("measured_s") or 0) > 0
            and float(row.get("payload_mib") or 0) > 0
            and (not rcs or all(int(c) == 0 for c in rcs))
        )

    ok = False
    if none and default:
        wall_sp = none["measured_s"] / default["measured_s"] if default["measured_s"] else float("nan")
        byte_sp = none["payload_mib"] / default["payload_mib"] if default["payload_mib"] else float("nan")
        print(f"  speedup vs uncompressed (wall)     {wall_sp:.2f}x")
        print(f"  speedup vs uncompressed (payload)  {byte_sp:.2f}x")
        print("  expected trend: compressed mode normally reduces wall time on 50 Gb/s")
        ok = (
            _complete(none)
            and _complete(default)
            and default["payload_mib"] < none["payload_mib"]
            and default["compression_ratio"] > 1.0
        )
    print()
    (ROOT / "results" / "pd" / "summary.json").write_text(
        json.dumps({"results": results, "ok": ok}, indent=2, default=str) + "\n"
    )
    if not ok:
        print("KVServe AE Track C1: FAILED")
        return 1
    print("KVServe AE Track C1: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
