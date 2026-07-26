#!/usr/bin/env python3
"""Benchmark TP2 KV compression pipelines and standalone fused/LC stages."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvserve_v1.compression.codec.lc_codec import LCCodec
from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.quantizer.tilelang_quantizer import (
    TileLangFusedQuantizer,
)
from kvserve_v1.compression.wire import build_wire


PIPELINES = [
    ("original_nvcomp", "configs/compression/original_top_nvcomp.json"),
    ("original_lc", "configs/compression/original_top_lc.json"),
    ("fused_nvcomp", "configs/compression/fused_top_nvcomp.json"),
    ("fused_lc", "configs/compression/fused_top_lc.json"),
]


def sync(device: str | torch.device) -> None:
    torch.cuda.synchronize(device)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_kv(args: argparse.Namespace) -> torch.Tensor:
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    return (
        torch.randn(
            args.num_layers,
            2,
            args.num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.head_size,
            generator=generator,
            device=args.device,
            dtype=torch.bfloat16,
        )
        * args.input_scale
    ).contiguous()


def benchmark(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    device: str,
) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmup):
        result = fn()
    sync(device)
    samples: list[float] = []
    for _ in range(iterations):
        sync(device)
        started = time.perf_counter()
        result = fn()
        sync(device)
        samples.append(time.perf_counter() - started)
    return result, samples


def timing_fields(samples: list[float], input_bytes: int) -> dict[str, float]:
    mean_s = statistics.mean(samples)
    median_s = statistics.median(samples)
    return {
        "mean_ms": mean_s * 1e3,
        "median_ms": median_s * 1e3,
        "min_ms": min(samples) * 1e3,
        "max_ms": max(samples) * 1e3,
        "mean_GBps": input_bytes / mean_s / 1e9,
        "median_GBps": input_bytes / median_s / 1e9,
    }


def benchmark_pipeline(
    name: str,
    config_path: Path,
    kv: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = load_config(config_path)
    model_name = config.get("quantizer_config", {}).get("model_name")
    adapter = KVCompressionAdapter(
        config,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        model_name=model_name,
        tp_rank=args.tp_rank,
        tp_size=args.tp_size,
    )

    # First touch and all TileLang JIT compilation are outside measurements.
    compressed = adapter.compress(kv, request_id=f"jit-{name}")
    if compressed is None:
        raise RuntimeError(f"{name}: compression returned None")
    restored = adapter.decompress(compressed)
    if restored is None:
        raise RuntimeError(f"{name}: decompression returned None")
    sync(args.device)

    sequence = 0

    def encode():
        nonlocal sequence
        sequence += 1
        result = adapter.compress(kv, request_id=f"bench-{name}-{sequence}")
        if result is None:
            raise RuntimeError(f"{name}: timed compression returned None")
        return result

    compressed, encode_samples = benchmark(
        encode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )

    def decode():
        result = adapter.decompress(compressed)
        if result is None:
            raise RuntimeError(f"{name}: timed decompression returned None")
        return result

    restored, decode_samples = benchmark(
        decode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )
    if tuple(restored.shape) != tuple(kv.shape):
        raise RuntimeError(
            f"{name}: restored shape {tuple(restored.shape)} != {tuple(kv.shape)}"
        )

    wire = build_wire(compressed, max_chunk_bytes=1 << 30)
    original_bytes = kv.numel() * kv.element_size()
    return {
        "category": "pipeline",
        "name": name,
        "throughput_basis": "original_bf16_bytes",
        "input_bytes": original_bytes,
        "output_bytes": wire.nbytes,
        "compression_ratio": original_bytes / wire.nbytes,
        "compress": timing_fields(encode_samples, original_bytes),
        "decompress": timing_fields(decode_samples, original_bytes),
    }


def make_fused_quantizer(args: argparse.Namespace) -> TileLangFusedQuantizer:
    config = load_config(ROOT / "configs/compression/fused_top_lc.json")
    quantizer_config = dict(config["quantizer_config"])
    quantizer_config.update({
        "tp_rank": args.tp_rank,
        "tensor_parallel_size": args.tp_size,
    })
    return TileLangFusedQuantizer(**quantizer_config)


def benchmark_fused_stage(
    kv: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]]]:
    quantizer = make_fused_quantizer(args)
    quantized = torch.empty_like(kv, dtype=torch.uint8)

    def encode():
        metadata: list[dict[str, Any]] = []
        for layer_id in range(args.num_layers):
            _, layer_meta = quantizer.quantize(
                layer_id,
                kv[layer_id],
                out=quantized[layer_id],
            )
            metadata.append(layer_meta)
        return metadata

    # Compile both encode and decode kernels before collecting samples.
    metadata = encode()
    for layer_id in range(args.num_layers):
        quantizer.dequantize(layer_id, quantized[layer_id], metadata[layer_id])
    sync(args.device)

    metadata, encode_samples = benchmark(
        encode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )

    def decode():
        restored = None
        for layer_id in range(args.num_layers):
            restored = quantizer.dequantize(
                layer_id,
                quantized[layer_id],
                metadata[layer_id],
            )
        return restored

    restored, decode_samples = benchmark(
        decode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )
    if restored is None or tuple(restored.shape) != tuple(kv[-1].shape):
        raise RuntimeError("fused stage returned an invalid tensor")

    original_bytes = kv.numel() * kv.element_size()
    quantized_bytes = quantized.numel() * quantized.element_size()
    row = {
        "category": "component",
        "name": "fused_transform_quant",
        "throughput_basis": "original_bf16_bytes",
        "input_bytes": original_bytes,
        "output_bytes": quantized_bytes,
        "compression_ratio": original_bytes / quantized_bytes,
        "compress": timing_fields(encode_samples, original_bytes),
        "decompress": timing_fields(decode_samples, original_bytes),
    }
    return row, quantized, metadata


def benchmark_lc_stage(
    quantized_native: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Match CompressionManager's default wire layout exactly.
    quantized = quantized_native.permute(0, 1, 4, 2, 3, 5).contiguous()
    codec = LCCodec(lc_algorithm="TUPL8_1 BIT_8 RZE_2")

    def encode():
        return codec.encode(quantized)

    compressed = encode()
    restored = codec.decode(
        compressed,
        "uint8",
        list(quantized.shape),
        args.device,
    )
    sync(args.device)
    if not torch.equal(restored, quantized):
        raise RuntimeError("LC standalone round trip mismatch")

    compressed, encode_samples = benchmark(
        encode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )

    def decode():
        return codec.decode(
            compressed,
            "uint8",
            list(quantized.shape),
            args.device,
        )

    restored, decode_samples = benchmark(
        decode,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )
    if not torch.equal(restored, quantized):
        raise RuntimeError("LC timed round trip mismatch")

    input_bytes = quantized.numel() * quantized.element_size()
    output_bytes = compressed.numel() * compressed.element_size()
    return {
        "category": "component",
        "name": "lc_only",
        "throughput_basis": "quantized_uint8_bytes",
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "compression_ratio": input_bytes / output_bytes,
        "effective_ratio_vs_original_bf16": (input_bytes * 2) / output_bytes,
        "compress": timing_fields(encode_samples, input_bytes),
        "decompress": timing_fields(decode_samples, input_bytes),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "category",
        "name",
        "throughput_basis",
        "input_bytes",
        "output_bytes",
        "compression_ratio",
        "effective_ratio_vs_original_bf16",
        "compress_mean_ms",
        "compress_median_ms",
        "compress_mean_GBps",
        "compress_median_GBps",
        "decompress_mean_ms",
        "decompress_median_ms",
        "decompress_mean_GBps",
        "decompress_median_GBps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "category": row["category"],
                "name": row["name"],
                "throughput_basis": row["throughput_basis"],
                "input_bytes": row["input_bytes"],
                "output_bytes": row["output_bytes"],
                "compression_ratio": row["compression_ratio"],
                "effective_ratio_vs_original_bf16": row.get(
                    "effective_ratio_vs_original_bf16", ""
                ),
                "compress_mean_ms": row["compress"]["mean_ms"],
                "compress_median_ms": row["compress"]["median_ms"],
                "compress_mean_GBps": row["compress"]["mean_GBps"],
                "compress_median_GBps": row["compress"]["median_GBps"],
                "decompress_mean_ms": row["decompress"]["mean_ms"],
                "decompress_median_ms": row["decompress"]["median_ms"],
                "decompress_mean_GBps": row["decompress"]["mean_GBps"],
                "decompress_median_GBps": row["decompress"]["median_GBps"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=243)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-scale", type=float, default=0.35)
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=[name for name, _ in PIPELINES],
        default=[name for name, _ in PIPELINES],
        help="Pipeline rows to benchmark (default: all).",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kv = make_kv(args)
    sync(args.device)
    original_bytes = kv.numel() * kv.element_size()
    print(
        f"device={torch.cuda.get_device_name(args.device)} shape={list(kv.shape)} "
        f"original={original_bytes / 1e6:.1f} MB "
        f"TP={args.tp_size} rank={args.tp_rank} "
        f"warmup={args.warmup} iterations={args.iterations}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for name, relative_path in PIPELINES:
        if name not in args.pipelines:
            continue
        print(f"benchmarking {name}", flush=True)
        rows.append(
            benchmark_pipeline(name, ROOT / relative_path, kv, args)
        )

    print("benchmarking fused transform+quantize stage", flush=True)
    fused_row, quantized, _ = benchmark_fused_stage(kv, args)
    rows.append(fused_row)

    print("benchmarking LC-only stage", flush=True)
    rows.append(benchmark_lc_stage(quantized, args))

    result = {
        "device_name": torch.cuda.get_device_name(args.device),
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": list(kv.shape),
        "dtype": str(kv.dtype),
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "pipelines": args.pipelines,
        "rows": rows,
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "throughput.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    for row in rows:
        print(
            f"{row['name']}: ratio={row['compression_ratio']:.3f}x "
            f"compress={row['compress']['mean_GBps']:.3f} GB/s "
            f"decompress={row['decompress']['mean_GBps']:.3f} GB/s",
            flush=True,
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
