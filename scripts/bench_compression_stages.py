#!/usr/bin/env python3
"""Compare KVServe compression stages on NVIDIA and ported backends.

Run with a CUDA or ROCm PyTorch environment. The default backend measures the
legacy KVServe HadamardTransform, quantizer_func and nvCOMP/bitpack codec; it
does not measure the Mooncake C++/CUDA runtime. Both systems must use the same
input file and the same benchmark settings.

    python bench_compression_stages.py --make-input input.pt --axis channel
    python bench_compression_stages.py --input input.pt --axis channel \
        --backend kvserve-legacy --codec nvcomp --output nvidia.json
    python bench_compression_stages.py --input input.pt --axis channel \
        --backend /path/to/amd_backend.py --codec mycodec --output amd.json

For the other axis, generate another input with --axis token. The input file
contains the same BF16 source and quantized-symbol fixture for both platforms.
The symbol fixture is generated independently of the tested quantizer; it keeps
codec comparisons on identical bytes rather than platform-dependent outputs.
Measurements are stage-level interface timings, not end-to-end serving or
isolated kernel durations. Quantizer metadata is excluded from byte counts.

An external backend module exports create_backend(device, axis, levels, codec).
The returned object provides:
  transform(x) -> tensor, inverse(y) -> tensor
  quantize(x) -> (uint8 tensor, metadata), dequantize(q, metadata) -> tensor
  encode(uint8 tensor) -> one contiguous uint8 payload tensor
  decode(payload, original_shape) -> uint8 tensor of original_shape
The source shape is [blocks, block_size, heads, head_size]. Transform is the
normalized, per-head signed Hadamard used by KVServe's legacy layer 0 with
base seed 0xC0FEBABE. Quantization is per-channel or per-token min/max, with
the number of levels specified by --levels. The built-in backend requires
fast_hadamard_transform and, for nvcomp, NVIDIA's nvCOMP Python bindings.
All work must be ordered on PyTorch's current stream for device; methods must
not mutate their inputs. Synchronize or join any internal streams back to the
current stream before returning. Codec payloads need not be cross-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch


class KVServeLegacyBackend:
    """Adapters to the original KVServe components, not the fused runtime."""

    def __init__(self, device: torch.device, axis: str, levels: int, codec: str):
        if torch.version.hip:
            raise RuntimeError("Use --backend with the AMD port's adapter on ROCm.")
        if codec not in ("nvcomp", "bitpack"):
            raise ValueError("The KVServe legacy backend supports nvcomp or bitpack.")
        from kvserve_v1.compression.codec.kvserve_codec import KVServeCodec
        from kvserve_v1.compression.quantizer.quantizer_func import dequantize, quantize
        from kvserve_v1.compression.transformer.hadamard_func import HadamardTransform

        self.device = device
        self.axis = axis
        self.levels = levels
        self.transformer = HadamardTransform(base_seed=0xC0FEBABE)
        self._quantize = quantize
        self._dequantize = dequantize
        options = {"codec_type": codec}
        if codec == "nvcomp":
            options["nvcomp_algorithm"] = "ANS"
        else:
            options["bitpack_bits"] = (levels - 1).bit_length()
        self.codec = KVServeCodec(**options)

    def transform(self, x):
        return self.transformer.transform(0, x)

    def inverse(self, x):
        return self.transformer.inverse(0, x)

    def quantize(self, x):
        return self._quantize(self.levels, x, self.axis)

    def dequantize(self, q, metadata):
        return self._dequantize(q, metadata)

    def encode(self, q):
        return self.codec.encode(0, q)

    def decode(self, payload, shape):
        return self.codec.decode(0, payload, "uint8", list(shape), str(self.device))


def load_backend(name: str, device: torch.device, axis: str, levels: int, codec: str):
    if name == "kvserve-legacy":
        return KVServeLegacyBackend(device, axis, levels, codec)
    path = Path(name).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("benchmark_external_backend", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load backend adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_backend(device=device, axis=axis, levels=levels, codec=codec)


def fingerprint(tensor: torch.Tensor) -> str:
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def reference_transform(source: torch.Tensor) -> torch.Tensor:
    """Independent CPU FP32 reference for KVServe's signed Hadamard transform."""
    generator = torch.Generator(device="cpu")
    signs = []
    for head in range(source.shape[2]):
        generator.manual_seed(0xC0FEBABE ^ head)  # layer 0
        signs.append(
            torch.randint(
                0, 2, (source.shape[3],), generator=generator, dtype=torch.int8
            )
            .float()
            .mul_(2)
            .sub_(1)
        )
    signed = source.float() * torch.stack(signs).view(1, 1, source.shape[2], -1)
    output = torch.empty_like(signed)
    offset, remaining = 0, source.shape[-1]
    while remaining:
        size = 1 << (remaining.bit_length() - 1)
        chunk = signed[..., offset : offset + size].contiguous()
        step = 1
        while step < size:
            pairs = chunk.reshape(-1, size // (2 * step), 2, step)
            left, right = pairs[:, :, 0, :], pairs[:, :, 1, :]
            chunk = torch.stack((left + right, left - right), dim=2).reshape(
                chunk.shape
            )
            step *= 2
        output[..., offset : offset + size] = chunk / math.sqrt(size)
        offset += size
        remaining -= size
    return output


def make_input(args) -> None:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.blocks, args.block_size, args.heads, args.head_size)
    source = (torch.randn(shape, generator=generator) * args.input_scale).to(
        torch.bfloat16
    )
    # Fixed codec fixture with a bounded quantized alphabet. This is not a
    # timing of the quantizer and is not claimed to be its byte-exact output.
    axes = (0, 1) if args.axis == "channel" else (2, 3)
    x = source.float()
    lo = torch.amin(x, dim=axes, keepdim=True)
    hi = torch.amax(x, dim=axes, keepdim=True)
    scale = (hi - lo).clamp(min=1e-5) / (args.levels - 1)
    symbols = ((x - lo) / scale).round().clamp(0, args.levels - 1).to(torch.uint8)
    fixture = {
        "source": source.contiguous(),
        "codec_input": symbols.contiguous(),
        "axis": args.axis,
        "levels": args.levels,
        "seed": args.seed,
    }
    path = Path(args.make_input)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, path)
    print(f"Wrote {path}: shape={shape}, axis={args.axis}, levels={args.levels}")
    print(f"source sha256={fingerprint(source)}, codec sha256={fingerprint(symbols)}")


def check_tensor(value, *, label, shape, dtype, device):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if (
        tuple(value.shape) != tuple(shape)
        or value.dtype != dtype
        or value.device != device
    ):
        raise ValueError(
            f"{label}: expected shape={tuple(shape)} dtype={dtype} device={device}; "
            f"got shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        )


def measure(fn, input_bytes: int, output_bytes: int, device, warmup, iterations):
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize(device)

    # One event pair per sample; event timing is a stream interval, which can
    # include launch gaps. It is not a profiler's sum of kernel durations.
    gpu_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        gpu_ms.append(start.elapsed_time(end))

    wall_ms = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(device)
        wall_ms.append((time.perf_counter() - start) * 1000)
    del result

    event_median = statistics.median(gpu_ms)
    wall_median = statistics.median(wall_ms)
    if event_median <= 0 or wall_median <= 0:
        raise RuntimeError("Timing resolution was insufficient; use a larger input.")
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "gpu_event_median_ms": event_median,
        "gpu_event_min_ms": min(gpu_ms),
        "gpu_event_max_ms": max(gpu_ms),
        "input_gb_per_s": input_bytes / event_median / 1e6,
        "output_gb_per_s": output_bytes / event_median / 1e6,
        "wall_median_ms": wall_median,
        "wall_input_gb_per_s": input_bytes / wall_median / 1e6,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--make-input", help="Create a CPU fixture file, then exit")
    inputs.add_argument("--input", help="Use an existing shared fixture file")
    p.add_argument(
        "--backend",
        default="kvserve-legacy",
        help="kvserve-legacy or an adapter .py path",
    )
    p.add_argument("--codec", default="nvcomp", help="Codec name passed to the adapter")
    p.add_argument("--axis", choices=("channel", "token"), default="channel")
    p.add_argument("--levels", type=int, default=16)
    p.add_argument("--blocks", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-scale", type=float, default=0.35)
    p.add_argument(
        "--device", default="cuda:0", help="PyTorch device (also cuda:N on ROCm)"
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--output", help="Optional JSON output path")
    args = p.parse_args()
    if not 2 <= args.levels <= 16:
        p.error("--levels must be between 2 and 16 for this BF16 reference check")
    if min(args.blocks, args.block_size, args.heads, args.head_size) < 1:
        p.error("input dimensions must be positive")
    if args.warmup < 1 or args.iterations < 1 or args.input_scale <= 0:
        p.error("warmup, iterations, and input scale must be positive")
    if args.make_input:
        make_input(args)
        return
    if not torch.cuda.is_available():
        p.error("A CUDA or ROCm PyTorch GPU is required for benchmarking")

    fixture = torch.load(args.input, map_location="cpu", weights_only=True)
    if fixture["axis"] != args.axis or fixture["levels"] != args.levels:
        p.error("--axis and --levels must match those used to create the input")
    source_cpu, codec_cpu = fixture["source"], fixture["codec_input"]
    if source_cpu.dtype != torch.bfloat16 or codec_cpu.dtype != torch.uint8:
        raise ValueError("Expected BF16 source and uint8 codec input")
    if (
        source_cpu.ndim != 4
        or source_cpu.shape != codec_cpu.shape
        or source_cpu.numel() == 0
    ):
        raise ValueError(
            "Fixture must contain equal nonempty 4D source and codec inputs"
        )
    if not source_cpu.is_contiguous() or not codec_cpu.is_contiguous():
        raise ValueError("Fixture tensors must be contiguous")
    if not torch.isfinite(source_cpu).all() or int(codec_cpu.max()) >= args.levels:
        raise ValueError("Fixture contains nonfinite data or out-of-range symbols")
    device = torch.device(args.device)
    if device.type != "cuda":
        p.error("--device must be a CUDA/ROCm device such as cuda:0")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    source = source_cpu.to(device)
    symbols = codec_cpu.to(device)

    with torch.cuda.device(device), torch.inference_mode():
        backend = load_backend(args.backend, device, args.axis, args.levels, args.codec)
        transformed = backend.transform(source)
        restored = backend.inverse(transformed)
        check_tensor(
            transformed,
            label="transform",
            shape=source.shape,
            dtype=source.dtype,
            device=device,
        )
        check_tensor(
            restored,
            label="inverse",
            shape=source.shape,
            dtype=source.dtype,
            device=device,
        )
        torch.testing.assert_close(
            transformed.float().cpu(),
            reference_transform(source_cpu),
            atol=0.03,
            rtol=0.03,
        )
        torch.testing.assert_close(restored, source, atol=0.03, rtol=0.03)

        quantized, metadata = backend.quantize(source)
        dequantized = backend.dequantize(quantized, metadata)
        check_tensor(
            quantized,
            label="quantize",
            shape=source.shape,
            dtype=torch.uint8,
            device=device,
        )
        check_tensor(
            dequantized,
            label="dequantize",
            shape=source.shape,
            dtype=source.dtype,
            device=device,
        )
        if not bool(torch.isfinite(dequantized).all()):
            raise ValueError("Dequantization produced nonfinite values")
        axes = (0, 1) if args.axis == "channel" else (2, 3)
        source_f32 = source.float()
        low = torch.amin(source_f32, dim=axes, keepdim=True)
        high = torch.amax(source_f32, dim=axes, keepdim=True)
        scale = (high - low).clamp(min=1e-5) / (args.levels - 1)
        reference_q = ((source_f32 - low) / scale).round().clamp(0, args.levels)
        max_symbol_difference = float((quantized.float() - reference_q).abs().max())
        if max_symbol_difference > 1:
            raise ValueError(
                f"Quantized symbols differ from the min/max reference by up to {max_symbol_difference}"
            )
        torch.testing.assert_close(
            dequantized.float(), quantized.float() * scale + low, atol=0.03, rtol=0.03
        )
        error = (dequantized.float() - source.float()).abs()
        quantization_error = {
            "mean_abs": float(error.mean()),
            "max_abs": float(error.max()),
            "max_symbol_difference_from_fp32_reference": max_symbol_difference,
        }

        payload = backend.encode(symbols)
        if (
            not isinstance(payload, torch.Tensor)
            or payload.dtype != torch.uint8
            or payload.device != device
        ):
            raise ValueError(
                "Codec payload must be a uint8 tensor on the selected device"
            )
        if payload.numel() == 0 or not payload.is_contiguous():
            raise ValueError("Codec payload must be nonempty and contiguous")
        decoded = backend.decode(payload, symbols.shape)
        check_tensor(
            decoded,
            label="decode",
            shape=symbols.shape,
            dtype=torch.uint8,
            device=device,
        )
        if not torch.equal(decoded, symbols):
            raise ValueError(
                "Lossless codec round trip did not restore the input exactly"
            )
        torch.cuda.synchronize(device)

        raw_bytes = source.numel() * source.element_size()
        symbol_bytes = symbols.numel() * symbols.element_size()
        payload_bytes = payload.numel() * payload.element_size()
        operations = {
            "transform": (lambda: backend.transform(source), raw_bytes, raw_bytes),
            "inverse_transform": (
                lambda: backend.inverse(transformed),
                raw_bytes,
                raw_bytes,
            ),
            "quantize": (lambda: backend.quantize(source), raw_bytes, symbol_bytes),
            "dequantize": (
                lambda: backend.dequantize(quantized, metadata),
                symbol_bytes,
                raw_bytes,
            ),
            "lossless_encode": (
                lambda: backend.encode(symbols),
                symbol_bytes,
                payload_bytes,
            ),
            "lossless_decode": (
                lambda: backend.decode(payload, symbols.shape),
                payload_bytes,
                symbol_bytes,
            ),
        }
        timings = {}
        for name, (fn, in_bytes, out_bytes) in operations.items():
            timings[name] = measure(
                fn, in_bytes, out_bytes, device, args.warmup, args.iterations
            )
            item = timings[name]
            print(
                f"{name:18s} event={item['gpu_event_median_ms']:.4f} ms  "
                f"input={item['input_gb_per_s']:.3f} GB/s  "
                f"output={item['output_gb_per_s']:.3f} GB/s  "
                f"wall={item['wall_median_ms']:.4f} ms",
                flush=True,
            )

    result = {
        "backend": args.backend,
        "codec": args.codec,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "shape": list(source.shape),
        "dtype": "bfloat16",
        "axis": args.axis,
        "levels": args.levels,
        "source_sha256": fingerprint(source_cpu),
        "codec_input_sha256": fingerprint(codec_cpu),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "codec_ratio": symbol_bytes / payload_bytes,
        "quantization_error": quantization_error,
        "timings": timings,
        "measurement": "GPU event stream interval and separate synchronized wall time; GB=10^9 bytes",
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output}")
    print(
        f"codec ratio={result['codec_ratio']:.3f}x, quantization error={quantization_error}"
    )


if __name__ == "__main__":
    main()
