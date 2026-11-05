#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Compare fixed-profile Runtime backends on identical synthetic KV pages.

Timings include Python submission, GPU execution, ANS and completion waits, but
exclude initialization, model forward, transport and buffer allocation. Use a
dedicated GPU and record competing workloads when interpreting measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

from kvserve_v1.compression.runtime import (
    CacheSpec,
    CompressionConfig,
    CompressionRuntime,
    build_memory_plan,
)
from kvserve_v1.compression.runtime.fused_runtime import create_runtime


def digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--backends", nargs="+", default=["python", "cpp"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires an NVIDIA CUDA device")
    if min(args.block_size, args.blocks, args.layers, args.heads, args.head_size,
           args.rounds, args.repeats) < 1 or args.warmups < 0:
        raise ValueError("Invalid benchmark geometry or repetition count")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(3817)
    config = CompressionConfig.from_extra_config({"compression": {
        "enable-compression-pipeline": True,
        "logical_blocks_per_chunk": args.blocks,
        "slot_count": 2,
    }})
    specs = {
        f"layer.{i}": CacheSpec(i, args.block_size, args.heads, args.head_size,
                                torch.bfloat16)
        for i in range(args.layers)
    }
    caches = {
        name: torch.randn((2 * args.blocks, args.heads, args.block_size,
                           2 * args.head_size), device=device, dtype=torch.bfloat16)
        for name in specs
    }
    plan = build_memory_plan(config, specs, blocks_per_logical=1,
                             num_layers=args.layers, device=device)
    tokens = args.blocks * args.block_size
    source, destination = list(range(args.blocks)), list(range(args.blocks, 2 * args.blocks))
    raw_bytes = args.layers * tokens * args.heads * args.head_size * 2 * 2
    rows = []
    expected = None
    try:
        for round_index in range(args.rounds):
            order = args.backends if round_index % 2 == 0 else list(reversed(args.backends))
            for backend in order:
                if backend not in ("python", "cpp", "fused"):
                    raise ValueError(f"Unknown benchmark backend: {backend}")
                os.environ["KVSERVE_COMPRESSION_BACKEND"] = (
                    "cpp" if backend == "fused" else backend
                )
                if backend == "fused":
                    runtime = create_runtime(plan, "HND", 1, device, backend="fused")
                else:
                    runtime = CompressionRuntime(plan, "HND", 1, device)
                try:
                    runtime.bind_kv_caches(caches, dict.fromkeys(specs, 0))

                    def encode():
                        chunks = []
                        for part in (0, 1):
                            chunk = runtime.encode(part, [source], part,
                                valid_tokens_by_group={0: tokens})
                            chunk.event.synchronize()
                            if chunk.raw_tail_bytes:
                                raise AssertionError("Full frames must have no raw tail")
                            chunks.append(chunk)
                        return chunks

                    def decode(chunks):
                        for part, chunk in enumerate(chunks):
                            runtime.decode(part, [destination], part, chunk.codec_bytes,
                                chunk.u8_bytes, valid_tokens_by_group={0: tokens}).synchronize()

                    for _ in range(args.warmups):
                        chunks = encode()
                        decode(chunks)
                    encode_seconds, decode_seconds = [], []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        chunks = encode()
                        encode_seconds.append(time.perf_counter() - start)
                        start = time.perf_counter()
                        decode(chunks)
                        decode_seconds.append(time.perf_counter() - start)

                    symbols, auxiliary = [], []
                    for part, chunk in enumerate(chunks):
                        slot = runtime.slots[part]
                        start = runtime.layout.u8_base
                        symbols.append(digest(slot.arena_b[start:start + chunk.u8_bytes]))
                        auxiliary.append(digest(slot.aux[:runtime.aux_bytes(part, {0: tokens})]))
                    restored = hashlib.sha256()
                    for name in specs:
                        restored.update(caches[name][args.blocks:].contiguous().view(torch.uint8)
                                        .cpu().numpy().tobytes())
                    identity = {"symbols": symbols, "auxiliary": auxiliary,
                                "restored": restored.hexdigest()}
                    if expected is None:
                        expected = identity
                    if identity != expected:
                        raise AssertionError(f"{backend}: codec output differs from reference")
                    row = {
                        "round": round_index, "backend": backend,
                        "runtime": type(runtime).__name__, "raw_bytes": raw_bytes,
                        "encoded_bytes": [chunk.codec_bytes for chunk in chunks],
                        "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                        "encode_GBps_median": raw_bytes / statistics.median(encode_seconds) / 1e9,
                        "decode_GBps_median": raw_bytes / statistics.median(decode_seconds) / 1e9,
                        "same_symbols_aux_restored": True,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    runtime.shutdown()
    finally:
        result = {
            "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "geometry": vars(args) | {"output": str(args.output)},
            "synthetic_input": "seeded BF16 normal", "rows": rows,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
