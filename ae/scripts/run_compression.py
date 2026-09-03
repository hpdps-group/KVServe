#!/usr/bin/env python3
"""Track B1 AE driver: run the real compression path and validate artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KVServe Track B1 compression AE")
    parser.add_argument("--model", default=_env("MODEL", "/data/models/Qwen2.5-7B-Instruct"))
    parser.add_argument(
        "--config", type=Path,
        default=Path(_env("CONFIG", str(ROOT / "configs/smoke/compression.json"))),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(_env("OUT_DIR", str(ROOT / "results/smoke"))),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path(_env("CACHE_ROOT", "/root/.cache/kvserve")),
    )
    parser.add_argument("--prefill-gpu", default=_env("PREFILL_GPU", "0"))
    parser.add_argument("--decode-gpu", default=_env("DECODE_GPU", "1"))
    parser.add_argument("--num-requests", type=int, default=int(_env("NUM_REQUESTS", "1")))
    parser.add_argument("--warmup-requests", type=int, default=int(_env("WARMUP_REQUESTS", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(_env("MAX_TOKENS", "8")))
    parser.add_argument("--kv-port", type=int, default=int(_env("KV_PORT", "25010")))
    parser.add_argument("--gpu-mem-util", type=float, default=0.6)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _measured_stats(path: Path, prefix: str = "sim-") -> list[dict[str, Any]]:
    by_transfer: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        transfer_id = str(row.get("transfer_id", ""))
        if not transfer_id.startswith(prefix):
            continue
        merged = by_transfer.setdefault(transfer_id, {"transfer_id": transfer_id})
        merged.update({key: value for key, value in row.items() if value is not None})
    return [by_transfer[key] for key in sorted(by_transfer)]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else float("nan")


def _pipeline_label(config: dict[str, Any]) -> str:
    labels: list[str] = []
    for component in config.get("pipeline") or []:
        if component == "transformer":
            labels.append(str((config.get("transformer_config") or {}).get(
                "transform_type", "transformer")).capitalize())
        elif component == "quantizer":
            labels.append("quantizer")
        elif component == "codec":
            codec = config.get("codec_config") or {}
            codec_type = str(codec.get("codec_type", "codec"))
            if codec_type.lower() == "nvcomp":
                codec_type = "nvCOMP"
            algorithm = codec.get("nvcomp_algorithm") or codec.get("lc_algorithm")
            labels.append(f"{codec_type}/{algorithm}" if algorithm else codec_type)
        else:
            labels.append(str(component))
    return " -> ".join(labels)


def _prepare_environment(args: argparse.Namespace) -> dict[str, str]:
    cache_dirs = {
        "CUPY_CACHE_DIR": args.cache_root / "cupy",
        "XDG_CACHE_HOME": args.cache_root / "xdg",
        "TRITON_CACHE_DIR": args.cache_root / "triton",
        "FLASHINFER_WORKSPACE_BASE": args.cache_root / "flashinfer",
        "VLLM_CACHE_ROOT": args.cache_root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": args.cache_root / "torchinductor",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in cache_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env.update({name: str(path) for name, path in cache_dirs.items()})
    return env


def _run_benchmark(args: argparse.Namespace, log_path: Path) -> int:
    command = [
        sys.executable, str(ROOT / "tests/test_kvserve.py"),
        "--mode", "custom",
        "--compression-config", str(args.config),
        "--model", args.model,
        "--num-requests", str(args.num_requests),
        "--warmup-requests", str(args.warmup_requests),
        "--max-tokens", str(args.max_tokens),
        "--prefill-gpu", str(args.prefill_gpu),
        "--decode-gpu", str(args.decode_gpu),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--kv-port", str(args.kv_port),
        "--output-dir", str(args.output_dir),
    ]
    print(f"[AE] Running Track B1 compression pipeline; detailed vLLM output -> {log_path}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command, cwd=ROOT, env=_prepare_environment(args),
            stdout=log_handle, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        print(f"KVServe AE Track B1: FAILED (benchmark exited {completed.returncode})", file=sys.stderr)
        print("[AE] Last 30 log lines:", file=sys.stderr)
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]:
            print(line, file=sys.stderr)
    return completed.returncode


def _validate_and_print(
    args: argparse.Namespace, benchmark_path: Path, stats_path: Path,
    summary_path: Path, log_path: Path,
) -> int:
    config = _read_json(args.config)
    benchmark = _read_json(benchmark_path)
    rows = _measured_stats(stats_path)
    expected_pipeline = list(config.get("pipeline") or [])
    raw_bytes = sum(int(row.get("original_bytes") or 0) for row in rows)
    compressed_bytes = sum(int(row.get("compressed_bytes") or 0) for row in rows)
    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else float("nan")
    checks = {
        "requests_completed": int(benchmark.get("requests", 0)) == args.num_requests,
        "compression_records_complete": len(rows) == args.num_requests,
        "pipeline_match": bool(rows) and all(
            row.get("compression_pipeline") == expected_pipeline for row in rows),
        "shape_match": bool(rows) and all(row.get("shape_match") is True for row in rows),
        "compression_ratio_gt_1": math.isfinite(ratio) and ratio > 1.0,
        "encode_succeeds": bool(rows) and all(row.get("encode_ms") is not None for row in rows),
        "decode_succeeds": bool(rows) and all(row.get("decode_ms") is not None for row in rows),
    }
    if "transformer" in expected_pipeline:
        transform_type = str((config.get("transformer_config") or {}).get("transform_type", ""))
        checks["transformer_forward"] = bool(rows) and all(
            row.get("transformer_applied") is True and row.get("transform_type") == transform_type
            for row in rows)
        checks["transformer_inverse"] = bool(rows) and all(
            row.get("inverse_transformer_applied") is True for row in rows)
    if "quantizer" in expected_pipeline:
        checks["quantizer_applied"] = bool(rows) and all(
            row.get("quantizer_applied") is True for row in rows)
    if "codec" in expected_pipeline:
        codec_type = (config.get("codec_config") or {}).get("codec_type")
        checks["codec_applied"] = bool(rows) and all(
            row.get("codec_type") == codec_type for row in rows)

    passed = all(checks.values())
    summary = {
        "track": "B1", "status": "passed" if passed else "failed",
        "model": benchmark.get("model"), "pipeline": expected_pipeline,
        "pipeline_label": _pipeline_label(config),
        "requests": int(benchmark.get("requests", 0)), "measured_transfers": len(rows),
        "original_bytes": raw_bytes, "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio, "mean_encode_ms": _mean(rows, "encode_ms"),
        "mean_decode_ms": _mean(rows, "decode_ms"),
        "request_wall_ms": float(benchmark.get("measured_job_time_s", 0.0)) * 1000.0,
        "checks": checks,
        "artifacts": {"benchmark": str(benchmark_path), "compression_stats": str(stats_path),
                      "full_log": str(log_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("KVServe AE Track B1 — Compression Pipeline\n\nINPUT")
    print(f"  model       {Path(str(benchmark.get('model', 'unknown'))).name}")
    print(f"  GPUs        prefill={args.prefill_gpu}  decode={args.decode_gpu}")
    print(f"  requests    {args.num_requests} measured (+{args.warmup_requests} warm-up)")
    print(f"  pipeline    {summary['pipeline_label']}\n\nOUTPUT")
    print(f"  completed   {benchmark.get('requests', 0)}/{args.num_requests} requests")
    print(f"  KV payload  {raw_bytes / 2**20:.2f} -> {compressed_bytes / 2**20:.2f} MiB  ({ratio:.2f}x)")
    print(f"  path time   encode={summary['mean_encode_ms']:.1f} ms  decode={summary['mean_decode_ms']:.1f} ms")
    print(f"  PD wall     {summary['request_wall_ms']:.1f} ms  (functional smoke only)")
    print("  checks      Hadamard fwd+inv / shape / encode+decode " + ("OK" if passed else "FAILED"))
    print()
    if passed:
        print("KVServe AE Track B1: PASSED")
    else:
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        print(f"KVServe AE Track B1: FAILED ({failed})")
    print(f"  summary     {summary_path}\n  full log    {log_path}")
    return 0 if passed else 1


def main() -> int:
    args = _args()
    args.config, args.output_dir, args.cache_root = (
        args.config.resolve(), args.output_dir.resolve(), args.cache_root.resolve())
    mode_label = args.config.stem
    log_path = args.output_dir / "run.log"
    benchmark_path = args.output_dir / f"benchmark_single_{mode_label}_builtin.json"
    stats_path = args.output_dir / f"compression_stats_{mode_label}.jsonl"
    summary_path = args.output_dir / "summary.json"
    return_code = _run_benchmark(args, log_path)
    if return_code:
        return return_code
    try:
        return _validate_and_print(args, benchmark_path, stats_path, summary_path, log_path)
    except Exception as error:
        print(f"KVServe AE Track B1: FAILED (artifact validation: {error})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
