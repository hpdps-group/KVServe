#!/usr/bin/env python3
"""Measure the default per-layer BF16 FWHT with a captured real KV fixture.

Both the Python pipeline and native C++ layer loop use the production fused
kernel. Event timing can use graph replay; synchronized wall timing does not.
Input gathering, sign generation, allocation and network transfer are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
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

from kvserve_v1.compression.runtime import cpp_backend
from kvserve_v1.compression.runtime.pipeline import CompressionPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--minimum-gb-s", type=float, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.tokens, args.warmup, args.repeats, args.batch) <= 0:
        parser.error("tokens and timing counts must be positive")
    if not torch.cuda.is_available() or torch.version.hip:
        parser.error("Requires an NVIDIA CUDA environment")
    fixture = torch.load(args.input, map_location="cpu", weights_only=True)
    key, value = fixture["key"], fixture["value"]
    if (
        key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or key.ndim != 4
        or value.shape != key.shape
        or args.tokens > key.shape[1]
    ):
        parser.error("Requires matching [layers, tokens, heads, dim] BF16 tensors")
    dim = key.shape[-1]
    if dim < 1 or dim > 1024 or dim & (dim - 1):
        parser.error("This fused-kernel benchmark requires power-of-two dim <= 1024")
    device = torch.device(args.device)
    results = []
    with torch.cuda.device(device), torch.inference_mode():
        kernels = load_kernels(device)
        cpp_backend.load_kvs_compress(device)
        stream = torch.cuda.Stream(device=device)
        for name in ("key", "value"):
            source = fixture[name][:, : args.tokens].contiguous()
            data = source.to(device)
            layers, tokens, heads, dim = data.shape
            signed = data * signs_for(layers, heads, dim, device)
            native, python, scratch = [torch.empty_like(data) for _ in range(3)]

            def cpp_loop(
                signed=signed,
                native=native,
                scratch=scratch,
                shape=(layers, tokens, heads, dim),
            ):
                kernels.fwht(
                    signed.data_ptr(),
                    native.data_ptr(),
                    scratch.data_ptr(),
                    *shape,
                    stream.cuda_stream,
                )

            def python_loop(signed=signed, output=python, layers=layers, dim=dim):
                for i in range(layers):
                    CompressionPipeline._hadamard_transform(signed[i], output[i], dim)

            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                cpp_loop()
                python_loop()
            stream.synchronize()
            torch.testing.assert_close(native, python, rtol=0, atol=0)
            timings = {}
            for label, fn in (
                ("cpp_per_layer", cpp_loop),
                ("python_default", python_loop),
            ):
                timings[label] = add_rates(
                    gpu_timing(
                        fn,
                        stream,
                        args.warmup,
                        args.repeats,
                        args.batch,
                        use_graph=not args.no_graph,
                    ),
                    data.numel() * data.element_size(),
                    data.numel() * data.element_size(),
                )
                row = timings[label]
                print(
                    f"{name} {label}: {row['input_gb_s']:.2f} GB/s "
                    f"{row['event_median_ms']:.4f} ms; "
                    f"wall {row['wall_median_ms']:.4f} ms",
                    flush=True,
                )
            results.append(
                {
                    "part": name,
                    "shape": list(data.shape),
                    "input_sha256": sha256_tensor(source),
                    "timings": timings,
                }
            )
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "production_cuda_sha256": hashlib.sha256(CSRC.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch": args.batch,
        "scope": "per-layer normalized FWHT only; signed contiguous input; no gather or transport",
        "results": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    if any(
        t["input_gb_s"] < args.minimum_gb_s
        for row in results
        for t in row["timings"].values()
    ):
        raise SystemExit(f"Throughput below {args.minimum_gb_s} GB/s; see {target}")


if __name__ == "__main__":
    main()
