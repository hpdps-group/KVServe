#!/usr/bin/env python3
"""Track C2 AE: prefix-cache hit vs recompute vs remote SSD @ 5 Gbps.

The HotpotQA prompt (passages + question) is the cached prefix.
Compression uses KVServe's built-in default profile.
SSD bandwidth is simulated at 5 Gbps (NVMe is faster, so transfer dominates).

Reviewer:
    ./ae/scripts/run_kv_reuse.sh
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _py() -> str:
    for c in ("/opt/kvs-venv/bin/python", "python3", "python"):
        if c.startswith("/") and Path(c).exists():
            return c
        if not c.startswith("/"):
            return c
    return "python3"


def _cfg() -> dict:
    p = ROOT / "configs" / "representative" / "kv_reuse.json"
    return json.loads(p.read_text())


def _xfer_ms(nbytes: float, gbps: float) -> float:
    bps = gbps * 1e9 / 8.0
    return 0.0 if bps <= 0 else (nbytes / bps) * 1000.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def measure_recompute(cfg: dict, prompts: list[str], out_dir: Path) -> list[float]:
    """Run recompute in a child process so this process never holds a CUDA context."""
    payload = out_dir / "recompute_prompts.json"
    result = out_dir / "recompute_times.json"
    payload.write_text(json.dumps(prompts))
    proc = subprocess.run(
        [_py(), str(Path(__file__).resolve()), "--recompute", str(payload), str(result)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
    )
    if proc.returncode != 0 or not result.exists():
        raise RuntimeError("recompute worker failed")
    return json.loads(result.read_text())


def _recompute_worker(prompt_path: Path, result_path: Path) -> int:
    cfg = _cfg()
    prompts = json.loads(prompt_path.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"]["recompute"])
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model"],
        gpu_memory_utilization=float(cfg.get("gpu_mem_util", 0.6)),
        max_model_len=int(cfg["max_model_len"]),
        max_num_batched_tokens=int(cfg.get("max_num_batched_tokens", 8192)),
        max_num_seqs=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    params = SamplingParams(max_tokens=int(cfg["max_tokens"]), temperature=0)
    llm.generate(prompts[:1], sampling_params=params)
    times_ms: list[float] = []
    for prompt in prompts:
        t0 = time.perf_counter()
        llm.generate([prompt], sampling_params=params)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    result_path.write_text(json.dumps(times_ms) + "\n")
    return 0


def run_kvserve_default(cfg: dict, data_path: Path, out_dir: Path) -> Path:
    py = _py()
    cmd = [
        py, str(ROOT / "tests" / "test_kvserve.py"),
        "--mode", "default",
        "--model", cfg["model"],
        "--data-path", str(data_path),
        "--num-requests", str(int(cfg["num_requests"])),
        "--warmup-requests", str(int(cfg.get("warmup_requests", 1))),
        "--max-tokens", str(int(cfg["max_tokens"])),
        "--max-model-len", str(int(cfg["max_model_len"])),
        "--max-prompt-chars", str(int(cfg.get("max_prompt_chars", 0))),
        "--max-num-batched-tokens", str(int(cfg.get("max_num_batched_tokens", cfg["max_model_len"]))),
        "--max-num-seqs", str(int(cfg.get("max_num_seqs", 1))),
        "--prefill-gpu", str(int(cfg["gpus"]["prefill"])),
        "--decode-gpu", str(int(cfg["gpus"]["decode"])),
        "--gpu-mem-util", str(float(cfg.get("gpu_mem_util", 0.6))),
        "--kv-port", str(int(cfg.get("kv_port", 25110))),
        "--barrier-timeout-s", str(float(cfg.get("barrier_timeout_s", 1200))),
        "--output-dir", str(out_dir),
    ]
    log = out_dir / "kvserve_default.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"KVServe default PD failed, see {log}")
    return out_dir / "compression_stats_default.jsonl"


def load_stats(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    by_tid: dict[str, dict] = {}
    for line in path.read_text().splitlines():
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
    rows = list(by_tid.values())
    rows.sort(key=lambda r: str(r.get("transfer_id", "")))
    return rows


def main() -> int:
    sys.path.insert(0, str(ROOT))
    cfg = _cfg()
    out_dir = ROOT / "results" / "kv_reuse"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / cfg["dataset"]
    if not data_path.exists():
        raise FileNotFoundError(f"missing dataset {data_path}")

    from tests.test_kvserve import build_prompts

    n = int(cfg["num_requests"])
    prompts = build_prompts(
        None, n, data_path=str(data_path),
        max_prompt_chars=int(cfg.get("max_prompt_chars", 0)),
        max_prompt_tokens=int(cfg["max_model_len"]) - 128,
    )
    gbps = float(cfg["ssd_gbps"])

    print("KVServe AE Track C2 — Prefix cache (reduced functional test)")
    print()
    print("INPUT")
    print(f"  model          {cfg['model']}")
    print(f"  dataset        HotpotQA  ({data_path.name}, n={len(prompts)})")
    print(f"  prefix         LongBench prompt (passages + question)")
    print(f"  compression    default")
    print(f"  SSD bandwidth  {gbps:.0f} Gbps (simulated)")
    print(f"  max_model_len  {cfg['max_model_len']}  (char_cap={cfg.get('max_prompt_chars', 0)})")
    print(f"  max_tokens     {cfg['max_tokens']}")
    print()
    print("OUTPUT", flush=True)

    print("  measuring recompute (single GPU, no KV reuse)...", flush=True)
    t_recompute = measure_recompute(cfg, prompts, out_dir)
    print(f"  recompute done  mean={_mean(t_recompute):.1f} ms", flush=True)

    print("  running KVServe default compression (PD)...", flush=True)
    stats_path = run_kvserve_default(cfg, data_path, out_dir)
    stats = load_stats(stats_path)
    if not stats:
        print("  FAIL: no compression stats (default path did not compress)")
        return 1

    t_ssd = []
    t_kvs = []
    crs = []
    for row in stats:
        raw = float(row["original_bytes"])
        comp = float(row["compressed_bytes"])
        if raw <= 0 or comp <= 0:
            continue
        decode_ms = float(row.get("decode_ms") or 0.0)
        t_ssd.append(_xfer_ms(raw, gbps))
        t_kvs.append(_xfer_ms(comp, gbps) + decode_ms)
        crs.append(raw / comp)

    rec = _mean(t_recompute)
    ssd = _mean(t_ssd)
    kvs = _mean(t_kvs)
    cr = _mean(crs)
    decode_ms = _mean([float(r.get("decode_ms") or 0.0) for r in stats])
    raw_mib = _mean([float(r["original_bytes"]) / (1024 ** 2) for r in stats if r.get("original_bytes")])
    vs_rec = rec / kvs if kvs > 0 else float("nan")
    vs_ssd = ssd / kvs if kvs > 0 else float("nan")

    print()
    print(f"  recompute (cold prefill)     {rec:8.1f} ms")
    print(f"  remote SSD uncompressed      {ssd:8.1f} ms   @ {gbps:.0f} Gbps")
    print(f"  KVServe default (cache hit)  {kvs:8.1f} ms   cr={cr:.2f}x  decode={decode_ms:.1f} ms")
    print(f"  mean KV size                 {raw_mib:8.1f} MiB  (n={len(t_kvs)})")
    print()
    print(f"  speedup vs recompute         {vs_rec:.2f}x")
    print(f"  speedup vs remote SSD        {vs_ssd:.2f}x")
    print("  expected trend: cache-hit is normally faster than recompute and uncompressed SSD")
    print()

    summary = {
        "n": len(prompts),
        "n_compressed": len(t_kvs),
        "recompute_ms": rec,
        "ssd_ms": ssd,
        "kvserve_ms": kvs,
        "compression_ratio": cr,
        "speedup_vs_recompute": vs_rec,
        "speedup_vs_ssd": vs_ssd,
        "ssd_gbps": gbps,
        "mean_decode_ms": decode_ms,
        "mean_kv_mib": raw_mib,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ok = (
        rec > 0
        and len(t_kvs) == len(prompts)
        and cr > 1.0
        and decode_ms > 0
    )
    if not ok:
        print("KVServe AE Track C2: FAILED")
        return 1
    print("KVServe AE Track C2: PASSED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--recompute":
        sys.exit(_recompute_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(main())
