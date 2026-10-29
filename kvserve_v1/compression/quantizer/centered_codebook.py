"""Calibration-free affine Lloyd–Max quantization, with packet-owned scales."""

from functools import lru_cache
import math
import threading

import torch
import triton
import triton.language as tl

from kvserve_v1.compression.components import Quantizer


@lru_cache(maxsize=4)
def _centroids(bits):
    levels = 1 << bits
    centers = [(i - (levels - 1) / 2) * 4 / levels for i in range(levels)]
    for _ in range(300):
        bounds = [-12.0] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [12.0]
        pdf = [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in bounds]
        cdf = [0.5 * (1 + math.erf(x / math.sqrt(2))) for x in bounds]
        centers = [
            (pdf[i] - pdf[i + 1]) / max(cdf[i + 1] - cdf[i], 1.0e-15)
            for i in range(levels)
        ]
    return tuple(centers)


@triton.jit
def _bin2(x, bounds, BASE: tl.constexpr):
    return BASE + (x > tl.load(bounds + BASE)).to(tl.int32)


@triton.jit
def _bin4(x, bounds, BASE: tl.constexpr):
    return tl.where(
        x > tl.load(bounds + BASE + 1),
        _bin2(x, bounds, BASE + 2),
        _bin2(x, bounds, BASE),
    )


@triton.jit
def _bin8(x, bounds, BASE: tl.constexpr):
    return tl.where(
        x > tl.load(bounds + BASE + 3),
        _bin4(x, bounds, BASE + 4),
        _bin4(x, bounds, BASE),
    )


@triton.jit
def _bucketize(x, bounds, BITS: tl.constexpr):
    # Direct, compile-time threshold addresses avoid dependent gather loads.
    if BITS == 1:
        return _bin2(x, bounds, 0)
    elif BITS == 2:
        return _bin4(x, bounds, 0)
    elif BITS == 3:
        return _bin8(x, bounds, 0)
    else:
        return tl.where(
            x > tl.load(bounds + 7), _bin8(x, bounds, 8), _bin8(x, bounds, 0)
        )


@triton.jit
def _quantize(
    X,
    Q,
    M,
    S,
    C,
    B,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NG: tl.constexpr,
    BITS: tl.constexpr,
    CHANNEL: tl.constexpr,
    STEPS: tl.constexpr,
    ALPHA: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    d = tl.program_id(2) * BD + tl.arange(0, BD)
    offset = (t[:, None] * H + head) * D + d[None, :]
    mask = (t[:, None] < T) & (d[None, :] < D)
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    if CHANNEL:
        count = tl.minimum(BT, T - tl.program_id(0) * BT)
        mean = (tl.sum(x, 0) / count).to(M.dtype.element_ty).to(tl.float32)
        residual = tl.where(mask, x - mean[None, :], 0.0)
        scale = tl.sqrt(tl.sum(residual * residual, 0) / count) * ALPHA
    else:
        mean = (tl.sum(x, 1) / D).to(M.dtype.element_ty).to(tl.float32)
        residual = tl.where(mask, x - mean[:, None], 0.0)
        scale = tl.sqrt(tl.sum(residual * residual, 1) / D) * ALPHA
    scale = scale.to(S.dtype.element_ty).to(tl.float32)
    for _ in tl.static_range(STEPS):
        denominator = tl.maximum(scale, 1.0e-20)
        normalized = tl.div_rn(
            residual, denominator[None, :] if CHANNEL else denominator[:, None]
        )
        q = _bucketize(normalized, B, BITS)
        centers = tl.where(mask, tl.load(C + q), 0.0)
        if CHANNEL:
            scale = tl.sum(residual * centers, 0) / tl.maximum(
                tl.sum(centers * centers, 0), 1.0e-20
            )
        else:
            scale = tl.sum(residual * centers, 1) / tl.maximum(
                tl.sum(centers * centers, 1), 1.0e-20
            )
        scale = tl.maximum(scale, 0.0).to(S.dtype.element_ty).to(tl.float32)
    denominator = tl.maximum(scale, 1.0e-20)
    normalized = tl.div_rn(
        residual, denominator[None, :] if CHANNEL else denominator[:, None]
    )
    q = _bucketize(normalized, B, BITS)
    tl.store(Q + offset, q.to(tl.uint8), mask)
    if CHANNEL:
        meta = (head * NG + tl.program_id(0)) * D + d
        tl.store(M + meta, mean, d < D)
        tl.store(S + meta, scale, d < D)
    else:
        meta = head * T + t
        tl.store(M + meta, mean, t < T)
        tl.store(S + meta, scale, t < T)


@triton.jit
def _dequantize(
    Q,
    X,
    M,
    S,
    C,
    N: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NG: tl.constexpr,
    GROUP: tl.constexpr,
    CHANNEL: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    head = i // D % H
    token = i // (H * D)
    meta = (head * NG + token // GROUP) * D + d if CHANNEL else head * T + token
    q = tl.load(Q + i, i < N, 0).to(tl.int32)
    tl.device_assert(q < (1 << BITS), "invalid centered-codebook symbol")
    center = tl.load(C + q, (i < N) & (q < (1 << BITS)), 0)
    mean = tl.load(M + meta, i < N, 0).to(tl.float32)
    scale = tl.load(S + meta, i < N, 0).to(tl.float32)
    tl.store(X + i, center * scale + mean, i < N)


class CenteredCodebookQuantizer(Quantizer):
    """Per-channel K / per-head-token V quantizer for ``[2,B,S,H,D]``.

    Shared Gaussian Lloyd–Max tables need no model or calibration data. K
    statistics cover bounded token groups. Means/scales belong to the returned
    packet, so another request or configuration update cannot overwrite them.
    """

    _defaults = dict(
        key_bits=3,
        value_bits=3,
        group_size=128,
        scale_refinements=1,
        scale_alpha=1.0,
        mean_dtype="float32",
    )
    supports_out_buffer = True

    def __init__(self, **kwargs):
        self._config = dict(self._defaults)
        self._tables = {}
        self._lock = threading.Lock()
        self.update_params(**kwargs)

    def _settings(self, kwargs):
        cfg = self._config | {k: v for k, v in kwargs.items() if k in self._defaults}
        if any(
            type(cfg[k]) is not int or not 1 <= cfg[k] <= 4
            for k in ("key_bits", "value_bits")
        ):
            raise ValueError("key_bits/value_bits must be integers in [1, 4]")
        group = cfg["group_size"]
        if type(group) is not int or group not in (16, 32, 64, 128, 256):
            raise ValueError("group_size must be 16, 32, 64, 128 or 256")
        if type(cfg["scale_refinements"]) is not int or cfg[
            "scale_refinements"
        ] not in (0, 1, 2):
            raise ValueError("scale_refinements must be 0, 1 or 2")
        if (
            not isinstance(cfg["scale_alpha"], (int, float))
            or not math.isfinite(cfg["scale_alpha"])
            or cfg["scale_alpha"] <= 0
        ):
            raise ValueError("scale_alpha must be finite and positive")
        if cfg["mean_dtype"] not in ("bfloat16", "float32"):
            raise ValueError("mean_dtype must be bfloat16 or float32")
        return cfg

    def update_params(self, **kwargs):
        self._config = self._settings(kwargs)

    def _table(self, bits, device):
        stream = torch.cuda.current_stream(device)
        key = (bits, device)
        with self._lock:
            entry = self._tables.get(key)
            if entry is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm centered-codebook tables before CUDA capture"
                    )
                centers = torch.tensor(
                    _centroids(bits), device=device, dtype=torch.float32
                )
                bounds = (centers[:-1] + centers[1:]) / 2
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = centers, bounds, ready, stream
                self._tables[key] = entry
        centers, bounds, ready, owner = entry
        if stream.cuda_stream != owner.cuda_stream:
            if ready is not None:
                if ready.query():
                    with self._lock:
                        if self._tables.get(key) is entry:
                            self._tables[key] = centers, bounds, None, owner
                elif torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Complete codebook initialization before CUDA capture"
                    )
                else:
                    stream.wait_event(ready)
            centers.record_stream(stream)
            bounds.record_stream(stream)
        return centers, bounds

    @staticmethod
    def _check(tensor):
        if not tensor.is_cuda or tensor.ndim != 5 or tensor.shape[0] != 2:
            raise ValueError("Expected CUDA KV [2, blocks, tokens, heads, dim]")
        if tensor.shape[-1] < 1 or tensor.shape[-1] > 512:
            raise ValueError("Supported head dimensions are 1–512")

    def quantize(self, layer_id, tensor, **kwargs):
        self._check(tensor)
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("Expected BF16, FP16 or FP32 KV")
        cfg = self._settings(kwargs)
        _, blocks, tokens, heads, dim = tensor.shape
        total = blocks * tokens
        group = cfg["group_size"]
        source = tensor.contiguous()
        output = kwargs.get("out")
        if output is None:
            output = torch.empty_like(source, dtype=torch.uint8)
        elif (
            output.shape != source.shape
            or output.device != source.device
            or output.dtype != torch.uint8
            or not output.is_contiguous()
            or output.untyped_storage().data_ptr()
            == tensor.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "out must be an independent contiguous CUDA uint8 buffer of the same shape"
            )
        params = []
        with torch.cuda.device(tensor.device):
            source.record_stream(torch.cuda.current_stream())
            for part, bits in enumerate((cfg["key_bits"], cfg["value_bits"])):
                shape = (
                    (heads, triton.cdiv(total, group), dim)
                    if part == 0
                    else (heads, total)
                )
                mean = torch.empty(
                    shape, device=tensor.device, dtype=getattr(torch, cfg["mean_dtype"])
                )
                scale = torch.empty(shape, device=tensor.device, dtype=torch.bfloat16)
                params.append(dict(mean=mean, scale=scale))
                if total * heads:
                    centers, bounds = self._table(bits, tensor.device)
                    bt, bd = (
                        (group, 32) if part == 0 else (16, triton.next_power_of_2(dim))
                    )
                    _quantize[(triton.cdiv(total, bt), heads, triton.cdiv(dim, bd))](
                        source[part],
                        output[part],
                        mean,
                        scale,
                        centers,
                        bounds,
                        total,
                        heads,
                        dim,
                        triton.cdiv(total, group),
                        bits,
                        part == 0,
                        cfg["scale_refinements"],
                        cfg["scale_alpha"],
                        bt,
                        bd,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        return output, dict(
            format="centered_codebook_v1",
            config=cfg,
            original_dtype=str(tensor.dtype).removeprefix("torch."),
            parts=params,
        )

    def dequantize(self, layer_id, tensor, meta_data, **kwargs):
        self._check(tensor)
        if (
            tensor.dtype != torch.uint8
            or meta_data.get("format") != "centered_codebook_v1"
        ):
            raise ValueError("Invalid centered-codebook payload or format")
        if (
            set(meta_data.get("config", {})) != set(self._defaults)
            or len(meta_data.get("parts", [])) != 2
        ):
            raise ValueError("Incomplete centered-codebook metadata")
        cfg = self._settings(meta_data["config"])
        dtype = meta_data["original_dtype"]
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("Invalid centered-codebook output dtype")
        _, blocks, tokens, heads, dim = tensor.shape
        total, group = blocks * tokens, cfg["group_size"]
        source = tensor.contiguous()
        output = torch.empty_like(source, dtype=getattr(torch, dtype))
        with torch.cuda.device(tensor.device):
            stream = torch.cuda.current_stream()
            source.record_stream(stream)
            for part, bits in enumerate((cfg["key_bits"], cfg["value_bits"])):
                mean, scale = (meta_data["parts"][part][k] for k in ("mean", "scale"))
                shape = (
                    (heads, triton.cdiv(total, group), dim)
                    if part == 0
                    else (heads, total)
                )
                for value, expected_dtype in (
                    (mean, getattr(torch, cfg["mean_dtype"])),
                    (scale, torch.bfloat16),
                ):
                    if (
                        value.shape != shape
                        or value.device != tensor.device
                        or value.dtype != expected_dtype
                        or not value.is_contiguous()
                    ):
                        raise ValueError("Invalid centered-codebook auxiliary tensor")
                    value.record_stream(stream)
                if total * heads:
                    centers, _ = self._table(bits, tensor.device)
                    _dequantize[(triton.cdiv(total * heads * dim, 1024),)](
                        source[part],
                        output[part],
                        mean,
                        scale,
                        centers,
                        total * heads * dim,
                        total,
                        heads,
                        dim,
                        triton.cdiv(total, group),
                        group,
                        part == 0,
                        bits,
                        1024,
                        debug=True,
                        enable_fp_fusion=False,
                    )
        return output
