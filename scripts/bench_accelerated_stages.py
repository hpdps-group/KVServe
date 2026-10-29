#!/usr/bin/env python3
"""Benchmark current CUDA runtime kernels and preallocated nvCOMP ANS.

Input is a torch.save dictionary with BF16 `key` and `value` tensors in
[layers, tokens, KV heads, head_dim] order. It can contain real model KV or
synthetic input, but the source must be stated when interpreting results.
The same file can be used by another platform's equivalent benchmark.
Use --save-symbols on one system and --symbols-input on the other to guarantee
the lossless codecs receive identical quantized bytes. The JSON records hashes
for both each backend's quantizer output and the actual codec input.

This microbenchmark invokes the actual production FWHT and quantization CUDA
kernels through bench_runtime_kernels.cu, once per layer, and nvCOMP with
caller-owned buffers. It does not measure gather/sign, scatter, allocations,
network transfer or complete Mooncake requests. CUDA graph timing amortizes CPU
launch overhead for the kernels. nvCOMP uses batched stream events without graph
capture; synchronized interface timing is reported separately.

An optional --backend adapter module defines CODEC_NAME and exposes
load_kernels(device), returning an
object with fwht/quant/dequant pointer-based methods matching
bench_runtime_bind.cpp, and create_codec(q, stream), returning an object with
encode(), decode(), payload_bytes and a decoded uint8 tensor. All work must use
or join the supplied PyTorch stream. Codec buffers/configuration must be created
before timing; the codec must restore q exactly. --no-graph uses batched stream
events if an external backend cannot be captured by PyTorch's GPU graph API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "kvserve_v1/compression/runtime/csrc/kvs_compress.cu"


def sha256_tensor(x):
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load_kernels(device):
    if torch.version.hip or not torch.version.cuda:
        raise RuntimeError("This kernel wrapper targets NVIDIA CUDA, not ROCm.")
    major, minor = torch.cuda.get_device_capability(device)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return load(
        name="kvserve_bench_runtime",
        sources=[
            str(ROOT / "scripts/bench_runtime_kernels.cu"),
            str(ROOT / "scripts/bench_runtime_bind.cpp"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


class NvcompCodec:
    """Caller-owned buffers and configs, following the runtime ANS path."""

    def __init__(self, q, stream):
        from nvidia import nvcomp

        self.stream = stream
        self.codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
        self.cc = self.codec.compression_config(q.numel())
        self.dc = self.codec.decompression_config(self.cc)
        self.q_flat = q.view(-1)
        self.q_array = nvcomp.as_array(self.q_flat, cuda_stream=stream.cuda_stream)
        self.encoded = torch.empty(
            self.codec.get_max_comp_buffer_size(self.q_array),
            device=q.device,
            dtype=torch.uint8,
        )
        self.decoded = torch.empty_like(q)
        self.encoded_array = nvcomp.as_array(
            self.encoded, cuda_stream=stream.cuda_stream
        )
        self.decoded_flat = self.decoded.view(-1)
        self.decoded_array = nvcomp.as_array(
            self.decoded_flat, cuda_stream=stream.cuda_stream
        )
        with torch.cuda.stream(stream):
            self.first = self.encode()
        stream.synchronize()
        self.payload_bytes = int(self.first.buffer_size)
        self.payload_array = nvcomp.as_array(
            self.encoded[: self.payload_bytes], cuda_stream=stream.cuda_stream
        )

    def encode(self):
        return self.codec.encode(
            self.q_array, out=self.encoded_array, compression_config=self.cc
        )

    def decode(self):
        return self.codec.decode(
            self.payload_array, out=self.decoded_array, decompression_config=self.dc
        )


def load_backend(path, device):
    if path is None:
        return load_kernels(device), NvcompCodec, "native_cuda_nvcomp_ans", "nvCOMP ANS"
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("kvserve_bench_backend", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load backend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    codec_name = getattr(module, "CODEC_NAME", None)
    if not isinstance(codec_name, str) or not codec_name.strip():
        raise ValueError("External adapter must declare a nonempty CODEC_NAME")
    return module.load_kernels(device), module.create_codec, str(path), codec_name


def signs_for(layers, heads, dim, device):
    generator = torch.Generator(device="cpu")
    signs = torch.empty((layers, 1, heads, dim), dtype=torch.bfloat16)
    for layer in range(layers):
        for head in range(heads):
            generator.manual_seed(0xC0FEBABE ^ (layer << 16) ^ head)
            signs[layer, 0, head] = (
                torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int8)
                .float()
                .mul_(2)
                .sub_(1)
            )
    return signs.to(device)


def gpu_timing(fn, stream, warmup, repeats, batch, use_graph):
    """Stream-event timing plus separately measured synchronized wall time."""
    with torch.cuda.stream(stream):
        warm_results = []
        for _ in range(warmup):
            warm_results.append(fn())
    stream.synchronize()
    del warm_results
    graph, keepalive = None, []
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(batch):
                keepalive.append(fn())
        stream.synchronize()
    event_ms = []
    wall_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        results = []
        with torch.cuda.stream(stream):
            start.record()
            if use_graph:
                graph.replay()
            else:
                for _ in range(batch):
                    results.append(fn())
            end.record()
        end.synchronize()
        event_ms.append(start.elapsed_time(end) / batch)
        del results
    for _ in range(repeats):
        stream.synchronize()
        t0 = time.perf_counter()
        results = []
        with torch.cuda.stream(stream):
            for _ in range(batch):
                results.append(fn())
        stream.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000 / batch)
        del results
    del keepalive
    return {
        "event_median_ms": statistics.median(event_ms),
        "event_samples_ms": event_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "wall_samples_ms": wall_ms,
        "event_mode": "cuda_graph" if use_graph else "batched_stream_events",
    }


def add_rates(timing, input_bytes, output_bytes):
    ms = timing["event_median_ms"]
    if ms <= 0:
        raise RuntimeError("GPU timing resolution insufficient")
    timing.update(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        input_gb_s=input_bytes / ms / 1e6,
        output_gb_s=output_bytes / ms / 1e6,
    )
    return timing


def check_reference(
    source, transformed, quantized, dequantized, mins, scales, axis, levels
):
    """Check first, middle and last layers against PyTorch BF16 operations."""
    for layer in sorted({0, source.shape[0] // 2, source.shape[0] - 1}):
        x = source[layer]
        dim = x.shape[-1]
        # Production rounds each butterfly to BF16, then normalizes.
        ref = x.clone()
        step = 1
        while step < dim:
            pair = ref.view(-1, dim // (2 * step), 2, step)
            a, b = pair[:, :, 0, :], pair[:, :, 1, :]
            ref = torch.stack((a + b, a - b), dim=2).reshape_as(ref)
            step <<= 1
        ref.mul_(dim**-0.5)
        if not torch.equal(ref, transformed[layer]):
            raise AssertionError(f"FWHT differs from BF16 reference at layer {layer}")

        y = transformed[layer]
        dims = (0,) if axis == 0 else (1, 2)
        low = torch.amin(y, dim=dims, keepdim=True)
        high = torch.amax(y, dim=dims, keepdim=True)
        scale = (high - low).clamp_(min=1e-5).div_(levels[layer] - 1)
        inv = scale.reciprocal()
        q = (y - low).mul_(inv).round_().clamp_(0, levels[layer] - 1).to(torch.uint8)
        if not torch.equal(q, quantized[layer]):
            count = int((q != quantized[layer]).sum())
            raise AssertionError(
                f"Quantization differs at layer {layer}: {count} symbols"
            )
        if not torch.equal(low.flatten(), mins[layer]) or not torch.equal(
            scale.flatten(), scales[layer]
        ):
            raise AssertionError(f"Quantization metadata differs at layer {layer}")
        deq = q.to(torch.bfloat16).mul_(scale).add_(low)
        if not torch.equal(deq, dequantized[layer]):
            raise AssertionError(f"Dequantization differs at layer {layer}")


def benchmark_part(data, part, kernels, codec_factory, stream, args, shared_symbols):
    layers, tokens, heads, dim = data.shape
    axis = 0 if part == "key" else 1
    high, low = (12, 6) if part == "key" else (8, 4)
    low_layers = round(layers * 0.33)
    levels = [high if i < layers - low_layers else low for i in range(layers)]
    signed = data * signs_for(layers, heads, dim, data.device)
    transformed = torch.empty_like(data)
    scratch = torch.empty_like(data)
    inverse_out = torch.empty_like(data)
    q = torch.empty(data.shape, device=data.device, dtype=torch.uint8)
    aux_count = heads * dim if axis == 0 else tokens
    mins, scales, maxs, inverses = [
        torch.empty((layers, aux_count), device=data.device, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    dequantized = torch.empty_like(data)
    ptr = lambda x: x.data_ptr()

    def fwht():
        kernels.fwht(
            ptr(signed),
            ptr(transformed),
            ptr(scratch),
            layers,
            tokens,
            heads,
            dim,
            stream.cuda_stream,
        )

    def inverse_fwht():
        kernels.fwht(
            ptr(transformed),
            ptr(inverse_out),
            ptr(scratch),
            layers,
            tokens,
            heads,
            dim,
            stream.cuda_stream,
        )

    def quantize():
        kernels.quant(
            ptr(transformed),
            ptr(q),
            ptr(mins),
            ptr(scales),
            ptr(maxs),
            ptr(inverses),
            layers,
            tokens,
            heads,
            dim,
            high,
            low,
            low_layers,
            axis,
            stream.cuda_stream,
        )

    def dequantize():
        kernels.dequant(
            ptr(q),
            ptr(dequantized),
            ptr(mins),
            ptr(scales),
            layers,
            tokens,
            heads,
            dim,
            axis,
            stream.cuda_stream,
        )

    # Source copies and sign multiplication ran on the current stream.
    # The benchmark stream must wait before reading their output.
    stream.wait_stream(torch.cuda.current_stream(data.device))
    with torch.cuda.stream(stream):
        fwht()
        quantize()
        dequantize()
    stream.synchronize()
    check_reference(signed, transformed, q, dequantized, mins, scales, axis, levels)
    quantized_cpu = q.cpu()
    if shared_symbols is None:
        codec_cpu = quantized_cpu
        codec_input = q
    else:
        if shared_symbols.shape != q.shape or shared_symbols.dtype != torch.uint8:
            raise ValueError(
                f"Shared {part} codec input must match quantized shape/dtype"
            )
        codec_cpu = shared_symbols.contiguous()
        codec_input = codec_cpu.to(data.device)
        stream.wait_stream(torch.cuda.current_stream(data.device))

    # Codec objects/configs and caller-owned output buffers are kept alive and
    # initialized outside timing. External adapters follow the same contract.
    codec = codec_factory(codec_input, stream)
    payload_bytes = int(codec.payload_bytes)
    if (
        payload_bytes <= 0
        or codec.decoded.shape != q.shape
        or codec.decoded.dtype != q.dtype
    ):
        raise ValueError("Invalid codec payload size or decoded buffer")
    with torch.cuda.stream(stream):
        codec.decode()
    stream.synchronize()
    if not torch.equal(codec_input, codec.decoded):
        raise AssertionError("Codec did not exactly restore quantized symbols")

    raw_bytes = data.numel() * data.element_size()
    symbols = q.numel()
    operations = {
        "fwht_scale": (fwht, raw_bytes, raw_bytes),
        "inverse_fwht_scale": (inverse_fwht, raw_bytes, raw_bytes),
        "quantize": (quantize, raw_bytes, symbols),
        "dequantize": (dequantize, symbols, raw_bytes),
        "codec_encode": (codec.encode, symbols, payload_bytes),
        "codec_decode": (codec.decode, payload_bytes, symbols),
    }
    timings = {}
    for name, (fn, in_bytes, out_bytes) in operations.items():
        timings[name] = add_rates(
            gpu_timing(
                fn,
                stream,
                args.warmup,
                args.repeats,
                args.batch,
                use_graph=not args.no_graph and not name.startswith("codec_"),
            ),
            in_bytes,
            out_bytes,
        )
        row = timings[name]
        print(
            f"{part:5} {name:20} {row['event_median_ms']:.4f} ms "
            f"in {row['input_gb_s']:.2f} GB/s out {row['output_gb_s']:.2f} GB/s "
            f"wall {row['wall_median_ms']:.4f} ms",
            flush=True,
        )
    return {
        "part": part,
        "axis": "channel" if axis == 0 else "token",
        "high_levels": high,
        "low_levels": low,
        "low_layers": low_layers,
        "shape": list(data.shape),
        "original_bf16_bytes": raw_bytes,
        "symbol_bytes": symbols,
        "payload_bytes": payload_bytes,
        "quantized_sha256": sha256_tensor(quantized_cpu),
        "codec_input_sha256": sha256_tensor(codec_cpu),
        "codec_ratio": symbols / payload_bytes,
        "timings": timings,
    }, quantized_cpu if args.save_symbols else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--layers", type=int, default=0, help="0 means all layers")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--backend", help="External adapter .py (default: native CUDA + nvCOMP ANS)"
    )
    p.add_argument(
        "--no-graph", action="store_true", help="Use batched events for kernel timings"
    )
    p.add_argument(
        "--save-symbols", help="Save this backend's quantized K/V for sharing"
    )
    p.add_argument(
        "--symbols-input", help="Use shared uint8 K/V symbols for codec timing"
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--batch", type=int, default=5, help="operations per graph replay")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if min(args.tokens, args.warmup, args.repeats, args.batch) <= 0 or args.layers < 0:
        p.error("tokens, warmup, repeats and batch must be positive; layers >= 0")
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required")
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
        raise ValueError("The production FWHT requires power-of-two head_dim >= 2")
    shared = (
        torch.load(args.symbols_input, map_location="cpu", weights_only=True)
        if args.symbols_input
        else {}
    )
    if args.symbols_input and (
        not isinstance(shared, dict) or set(shared) != {"key", "value"}
    ):
        raise ValueError("Shared symbol fixture must contain key and value tensors")
    if args.symbols_input and any(
        not isinstance(t, torch.Tensor) for t in shared.values()
    ):
        raise TypeError("Shared symbol fixture entries must be tensors")
    print(
        f"GPU={torch.cuda.get_device_name(device)} shape={[layers, args.tokens, key.shape[2], key.shape[3]]}",
        flush=True,
    )
    with torch.cuda.device(device), torch.inference_mode():
        kernels, codec_factory, backend_name, codec_name = load_backend(
            args.backend, device
        )
        stream = torch.cuda.Stream(device=device)
        results = []
        hashes = {}
        saved_symbols = {}
        for name, cpu in (("key", key), ("value", value)):
            chosen = cpu[:layers, : args.tokens].contiguous()
            hashes[name] = sha256_tensor(chosen)
            data = chosen.to(device)
            row, symbols = benchmark_part(
                data, name, kernels, codec_factory, stream, args, shared.get(name)
            )
            results.append(row)
            if symbols is not None:
                saved_symbols[name] = symbols
            del data
    if args.save_symbols:
        symbol_path = Path(args.save_symbols)
        symbol_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_symbols, symbol_path)
        print(f"Wrote shared symbols {symbol_path}")
    try:
        nvcomp_version = (
            importlib.metadata.version(
                f"nvidia-nvcomp-cu{torch.version.cuda.split('.')[0]}"
            )
            if args.backend is None
            else None
        )
    except importlib.metadata.PackageNotFoundError:
        nvcomp_version = None
    output = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "backend": backend_name,
        "codec": codec_name,
        "nvcomp_version": nvcomp_version,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            (ROOT / "scripts/bench_runtime_kernels.cu").read_bytes()
        ).hexdigest()
        if args.backend is None
        else None,
        "production_cuda_sha256": hashlib.sha256(CSRC.read_bytes()).hexdigest()
        if args.backend is None
        else None,
        "input_metadata": {
            k: v for k, v in fixture.items() if k not in ("key", "value")
        },
        "input_sha256": hashes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch": args.batch,
        "timing": (
            "kernel batched stream events" if args.no_graph else "kernel graph events"
        )
        + "; codec batched stream events; separate synchronized Python wall time",
        "scope": "per-layer FWHT+scale, quant/dequant and preallocated codec; excludes gather/sign/scatter/network",
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
