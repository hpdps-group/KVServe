#!/usr/bin/env python3
"""Validate and print the compact Track B1 compression summary."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _measured_stats(path: Path, prefix: str) -> list[dict[str, Any]]:
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
            labels.append(
                str((config.get("transformer_config") or {}).get(
                    "transform_type", "transformer")).capitalize()
            )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--prefill-gpu", required=True)
    parser.add_argument("--decode-gpu", required=True)
    args = parser.parse_args()

    config = _read_json(args.config)
    benchmark = _read_json(args.benchmark)
    rows = _measured_stats(args.stats, "sim-")

    expected_pipeline = list(config.get("pipeline") or [])
    raw_bytes = sum(int(row.get("original_bytes") or 0) for row in rows)
    compressed_bytes = sum(int(row.get("compressed_bytes") or 0) for row in rows)
    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else float("nan")

    checks = {
        "requests_completed": int(benchmark.get("requests", 0)) == args.expected_requests,
        "compression_records_complete": len(rows) == args.expected_requests,
        "pipeline_match": bool(rows) and all(
            row.get("compression_pipeline") == expected_pipeline for row in rows),
        "shape_match": bool(rows) and all(row.get("shape_match") is True for row in rows),
        "compression_ratio_gt_1": math.isfinite(ratio) and ratio > 1.0,
        "encode_succeeds": bool(rows) and all(row.get("encode_ms") is not None for row in rows),
        "decode_succeeds": bool(rows) and all(row.get("decode_ms") is not None for row in rows),
    }
    if "transformer" in expected_pipeline:
        transform_type = str(
            (config.get("transformer_config") or {}).get("transform_type", ""))
        checks["transformer_forward"] = bool(rows) and all(
            row.get("transformer_applied") is True
            and row.get("transform_type") == transform_type
            for row in rows
        )
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
        "track": "B1",
        "status": "passed" if passed else "failed",
        "model": benchmark.get("model"),
        "pipeline": expected_pipeline,
        "pipeline_label": _pipeline_label(config),
        "requests": int(benchmark.get("requests", 0)),
        "measured_transfers": len(rows),
        "original_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "mean_encode_ms": _mean(rows, "encode_ms"),
        "mean_decode_ms": _mean(rows, "decode_ms"),
        "request_wall_ms": float(benchmark.get("measured_job_time_s", 0.0)) * 1000.0,
        "checks": checks,
        "artifacts": {
            "benchmark": str(args.benchmark),
            "compression_stats": str(args.stats),
            "full_log": str(args.log),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("KVServe AE Track B1 — Compression Pipeline")
    print()
    print("INPUT")
    print(f"  model       {Path(str(benchmark.get('model', 'unknown'))).name}")
    print(f"  GPUs        prefill={args.prefill_gpu}  decode={args.decode_gpu}")
    print(
        f"  requests    {args.expected_requests} measured "
        f"(+{benchmark.get('warmup_requests', 0)} warm-up)"
    )
    print(f"  pipeline    {summary['pipeline_label']}")
    print()
    print("OUTPUT")
    print(f"  completed   {benchmark.get('requests', 0)}/{args.expected_requests} requests")
    print(
        f"  KV payload  {raw_bytes / 2**20:.2f} -> "
        f"{compressed_bytes / 2**20:.2f} MiB  ({ratio:.2f}x)"
    )
    print(
        f"  path time   encode={summary['mean_encode_ms']:.1f} ms  "
        f"decode={summary['mean_decode_ms']:.1f} ms"
    )
    print(
        f"  PD wall     {summary['request_wall_ms']:.1f} ms  "
        "(functional smoke only)"
    )
    print(
        "  checks      Hadamard fwd+inv / shape / encode+decode "
        + ("OK" if passed else "FAILED")
    )
    print()
    if passed:
        print("KVServe AE Track B1: PASSED")
    else:
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        print(f"KVServe AE Track B1: FAILED ({failed})")
    print(f"  summary     {args.summary}")
    print(f"  full log    {args.log}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
