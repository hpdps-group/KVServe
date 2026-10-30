#!/usr/bin/env python3
"""Check benchmark symbols against a real CUDA CompressionRuntime encode.

Use the BF16 fixture and --save-symbols output from
bench_accelerated_stages.py with the same token count. The check uses a full
layer-split runtime with 16-token blocks and verifies every K/V symbol.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbols-input", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("cpp", "python"), default="cpp")
    parser.add_argument(
        "--worker", action="store_true", help="Encode from a worker thread"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tokens < 16 or args.tokens % 16 or not torch.cuda.is_available():
        parser.error("A CUDA GPU and token count divisible by 16 are required")
    os.environ["KVSERVE_COMPRESSION_BACKEND"] = args.backend
    from kvserve_v1.compression.runtime import (
        CacheSpec,
        CompressionConfig,
        CompressionRuntime,
        build_memory_plan,
    )

    fixture = torch.load(args.input, map_location="cpu", weights_only=True)
    expected = torch.load(args.symbols_input, map_location="cpu", weights_only=True)
    key, value = fixture["key"], fixture["value"]
    if (
        key.ndim != 4
        or key.shape != value.shape
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or key.shape[1] < args.tokens
    ):
        raise ValueError("Fixture must contain matching 4D BF16 K/V tensors")
    layers, _, heads, dim = key.shape
    shape = (layers, args.tokens, heads, dim)
    for name in ("key", "value"):
        if expected[name].dtype != torch.uint8 or tuple(expected[name].shape) != shape:
            raise ValueError(f"{name} symbols must have shape {shape} and dtype uint8")

    device = torch.device(args.device)
    blocks = args.tokens // 16
    names = [f"layer.{i}" for i in range(layers)]
    specs = {
        name: CacheSpec(i, 16, heads, dim, torch.bfloat16)
        for i, name in enumerate(names)
    }
    config = CompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "logical_blocks_per_chunk": blocks,
                "split_type": "layer",
                "slot_count": 2,
            }
        }
    )
    with (
        torch.cuda.device(device),
        torch.inference_mode(),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        plan = build_memory_plan(config, specs, num_layers=layers, device=device)
        runtime = CompressionRuntime(plan, "HND", 1, device)
        try:
            caches = {}
            for i, name in enumerate(names):
                k = (
                    key[i, : args.tokens]
                    .reshape(blocks, 16, heads, dim)
                    .permute(0, 2, 1, 3)
                )
                v = (
                    value[i, : args.tokens]
                    .reshape(blocks, 16, heads, dim)
                    .permute(0, 2, 1, 3)
                )
                caches[name] = torch.cat((k, v), dim=-1).to(device).contiguous()
            torch.cuda.synchronize(device)
            runtime.bind_kv_caches(caches, dict.fromkeys(names, 0))
            result = {
                "shape": list(shape),
                "device": str(device),
                "parts": {},
                "backend": args.backend,
                "worker_thread": args.worker,
            }

            def encode(part):
                with torch.cuda.device(device), torch.inference_mode():
                    return runtime.encode(
                        0,
                        [list(range(blocks))],
                        part,
                        valid_tokens_by_group={0: args.tokens},
                    )

            for part, name in enumerate(("key", "value")):
                encoded = (
                    executor.submit(encode, part).result()
                    if args.worker
                    else encode(part)
                )
                encoded.event.synchronize()
                base = runtime.layout.u8_base
                actual = runtime.slots[0].arena_b[base : base + encoded.u8_bytes]
                want = expected[name].reshape(-1).to(device)
                mismatches = int((actual != want).sum())
                result["parts"][name] = {
                    "symbol_bytes": encoded.u8_bytes,
                    "payload_bytes": encoded.codec_bytes,
                    "mismatches": mismatches,
                }
                if mismatches:
                    raise AssertionError(f"{name}: {mismatches} symbol mismatches")
        finally:
            runtime.shutdown()
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
