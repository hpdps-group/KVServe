#!/usr/bin/env python3
"""Compare the runtime CUDA FWHT with Humming on the same real KV tensors.

Requires the optional humming-kernels package in addition to the native CUDA
benchmark dependencies. Humming is measured directly, on the main thread; the
batched case is a microbenchmark, not a claim about the current runtime path.
Both cases exclude sign generation, gather, scatter and network transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from bench_accelerated_stages import (
    CSRC,
    add_rates,
    gpu_timing,
    load_kernels,
    sha256_tensor,
    signs_for,
)
from humming.ops import process_input


def run_part(data, part, kernels, stream, args):
    layers, tokens, heads, dim = data.shape
    signed = data * signs_for(layers, heads, dim, data.device)
    native = torch.empty_like(data)
    scratch = torch.empty_like(data)
    humming = torch.empty_like(data)

    def current():
        kernels.fwht(
            signed.data_ptr(),
            native.data_ptr(),
            scratch.data_ptr(),
            layers,
            tokens,
            heads,
            dim,
            stream.cuda_stream,
        )

    def humming_layer():
        for i in range(layers):
            process_input(
                signed[i].view(-1, dim),
                hadamard_block_size=dim,
                outputs=humming[i].view(-1, dim),
            )

    def humming_batch():
        process_input(
            signed.view(-1, dim), hadamard_block_size=dim, outputs=humming.view(-1, dim)
        )

    stream.wait_stream(torch.cuda.current_stream(data.device))
    with torch.cuda.stream(stream):
        current()
        humming_layer()
    stream.synchronize()
    layered = humming.clone()
    stream.wait_stream(torch.cuda.current_stream(data.device))
    with torch.cuda.stream(stream):
        humming_batch()
    stream.synchronize()
    if not torch.equal(layered, humming):
        raise AssertionError("Humming per-layer and batched results differ")
    if not torch.isfinite(native).all() or not torch.isfinite(humming).all():
        raise ValueError("Hadamard produced non-finite output")
    difference = (native.float() - humming.float()).abs()
    accuracy = {
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "exact_fraction": float((native == humming).float().mean()),
    }

    nbytes = data.numel() * data.element_size()
    timing = {}
    for name, fn in (
        ("current_cuda", current),
        ("humming_per_layer", humming_layer),
        ("humming_batched", humming_batch),
    ):
        timing[name] = add_rates(
            gpu_timing(
                fn,
                stream,
                args.warmup,
                args.repeats,
                args.batch,
                use_graph=not args.no_graph,
            ),
            nbytes,
            nbytes,
        )
        row = timing[name]
        print(
            f"{part:5} {name:20} {row['event_median_ms']:.4f} ms "
            f"{row['input_gb_s']:.2f} GB/s wall {row['wall_median_ms']:.4f} ms",
            flush=True,
        )
    return {
        "part": part,
        "shape": list(data.shape),
        "input_bytes": nbytes,
        "accuracy_vs_current": accuracy,
        "timings": timing,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--layers", type=int, default=0, help="0 means all layers")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if min(args.tokens, args.warmup, args.repeats, args.batch) <= 0 or args.layers < 0:
        p.error("tokens, warmup, repeats and batch must be positive; layers >= 0")
    if not torch.cuda.is_available() or torch.version.hip:
        p.error("This comparison requires an NVIDIA CUDA PyTorch environment")
    device = torch.device(args.device)
    fixture = torch.load(args.input, map_location="cpu", weights_only=True)
    key, value = fixture["key"], fixture["value"]
    if (
        key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or key.ndim != 4
        or key.shape != value.shape
    ):
        raise ValueError("key and value must have identical 4D BF16 shapes")
    layers = args.layers or key.shape[0]
    if layers > key.shape[0] or args.tokens > key.shape[1]:
        raise ValueError("Requested geometry exceeds the fixture")
    if key.shape[-1] < 2 or key.shape[-1] & (key.shape[-1] - 1):
        raise ValueError("FWHT requires power-of-two head_dim >= 2")

    with torch.cuda.device(device), torch.inference_mode():
        kernels = load_kernels(device)
        stream = torch.cuda.Stream(device=device)
        results = []
        hashes = {}
        for name, cpu in (("key", key), ("value", value)):
            chosen = cpu[:layers, : args.tokens].contiguous()
            hashes[name] = sha256_tensor(chosen)
            data = chosen.to(device)
            results.append(run_part(data, name, kernels, stream, args))
            del data
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "humming_version": importlib.metadata.version("humming-kernels"),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_cuda_sha256": hashlib.sha256(CSRC.read_bytes()).hexdigest(),
        "input_metadata": {
            k: v for k, v in fixture.items() if k not in ("key", "value")
        },
        "input_sha256": hashes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch": args.batch,
        "scope": "signed input, FWHT+scale only; excludes gather/sign/scatter and network",
        "results": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
