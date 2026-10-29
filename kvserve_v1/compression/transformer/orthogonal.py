"""Cached orthogonal bases with GPU-only, layout-preserving application."""

from collections import OrderedDict
from functools import lru_cache
import math
import threading

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=128)
def _base(kind, dim, seed, layer_id):
    # CPU constants make the basis independent of the peer GPU's RNG geometry.
    generator = torch.Generator(device="cpu")
    if kind == "dct":
        index = torch.arange(dim, dtype=torch.float64, device="cpu")
        base = torch.cos(math.pi / dim * (index[:, None] + 0.5) * index[None, :])
        base *= math.sqrt(2 / dim)
        base[:, 0] /= math.sqrt(2)
        return base.float().contiguous()
    generator.manual_seed((seed ^ (layer_id << 16)) & ((1 << 63) - 1))
    matrix, r = torch.linalg.qr(
        torch.randn(dim, dim, generator=generator, dtype=torch.float64, device="cpu")
    )
    matrix *= torch.where(r.diagonal() < 0, -1.0, 1.0)[None, :]
    return matrix.float().contiguous()


def _signs(kind, dim, heads, seed, layer_id):
    if kind == "haar":
        return torch.ones((heads, dim), dtype=torch.bfloat16, device="cpu")
    generator = torch.Generator(device="cpu")
    signs = []
    for head in range(heads):
        generator.manual_seed((seed ^ (layer_id << 16) ^ head) & ((1 << 63) - 1))
        signs.append(
            torch.randint(2, (dim,), generator=generator, device="cpu") * 2 - 1
        )
    return torch.stack(signs).to(torch.bfloat16)


@triton.jit
def _rotate(
    X,
    HI,
    LO,
    SIGNS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    INVERSE: tl.constexpr,
    SPLIT_INPUT: tl.constexpr,
    SIGNED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2) // H
    head = tl.program_id(2) % H
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(D, BK)):
        k = start * BK + kk
        x = tl.load(
            X + part * T * H * D + (row[:, None] * H + head) * D + k[None, :],
            (row[:, None] < T) & (k[None, :] < D),
            0,
        ).to(tl.float32)
        if SIGNED and not INVERSE:
            signs = tl.load(SIGNS + head * D + k, k < D, 0).to(tl.float32)
            x *= signs[None, :]
        if INVERSE:
            offset = col[None, :] * D + k[:, None]
        else:
            offset = k[:, None] * D + col[None, :]
        mask = (k[:, None] < D) & (col[None, :] < D)
        high = tl.load(HI + offset, mask, 0)
        low = tl.load(LO + offset, mask, 0)
        xhi = x.to(tl.bfloat16)
        acc = tl.dot(xhi, high, acc)
        acc = tl.dot(xhi, low, acc)
        if SPLIT_INPUT:
            xlo = (x - xhi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(xlo, high, acc)
            acc = tl.dot(xlo, low, acc)
    if SIGNED and INVERSE:
        signs = tl.load(SIGNS + head * D + col, col < D, 0).to(tl.float32)
        acc *= signs[None, :]
    tl.store(
        Y + part * T * H * D + (row[:, None] * H + head) * D + col[None, :],
        acc,
        (row[:, None] < T) & (col[None, :] < D),
    )


class OrthogonalTransform:
    """DCT-II or Haar rotation of ``[2, blocks, tokens, heads, dim]`` KV.

    Both peers need the same kind, seed, layer id and local head ordering.
    Bases are immutable setup constants, not per-request metadata. Warming a
    geometry initializes its CPU basis, CUDA copies and Triton kernels; the
    warmed transform and inverse perform no host/device data transfer.
    """

    def __init__(self, kind="dct", seed=0x3333):
        if kind not in ("dct", "haar") or type(seed) is not int:
            raise ValueError("Expected dct/haar and an integer seed")
        self.kind, self.seed = kind, seed
        self._matrices = OrderedDict()
        self._lock = threading.Lock()

    def _weights(self, layer_id, heads, dim, device):
        key = (layer_id, heads, dim, device)
        stream = torch.cuda.current_stream(device)
        with self._lock:
            entry = self._matrices.get(key)
            if entry is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm the orthogonal transform before CUDA capture"
                    )
                matrix = _base(self.kind, dim, self.seed, layer_id)
                high = matrix.to(torch.bfloat16)
                low = (matrix - high.float()).to(torch.bfloat16)
                high, low = high.to(device), low.to(device)
                signs = _signs(self.kind, dim, heads, self.seed, layer_id).to(device)
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = (high, low, signs, ready, stream)
                self._matrices[key] = entry
                if len(self._matrices) > 128:
                    self._matrices.popitem(last=False)
            self._matrices.move_to_end(key)
        high, low, signs, ready, owner = entry
        if stream.cuda_stream != owner.cuda_stream:
            if ready is not None:
                if ready.query():
                    with self._lock:
                        if self._matrices.get(key) is entry:
                            self._matrices[key] = high, low, signs, None, owner
                elif torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Complete transform initialization before CUDA capture"
                    )
                else:
                    stream.wait_event(ready)
            high.record_stream(stream)
            low.record_stream(stream)
            signs.record_stream(stream)
        return high, low, signs

    def _apply(self, layer_id, tensor, inverse):
        if not tensor.is_cuda or tensor.ndim != 5 or tensor.shape[0] != 2:
            raise ValueError("Expected CUDA KV [2, blocks, tokens, heads, dim]")
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("Orthogonal transforms require BF16, FP16 or FP32")
        _, blocks, tokens, heads, dim = tensor.shape
        if not 1 <= dim <= 512:
            raise ValueError("Orthogonal transforms support head dimensions 1–512")
        source = tensor.contiguous()
        out = torch.empty_like(source)
        if not source.numel():
            return out
        with torch.cuda.device(source.device):
            high, low, signs = self._weights(layer_id, heads, dim, source.device)
            source.record_stream(torch.cuda.current_stream())
            _rotate[
                (triton.cdiv(blocks * tokens, 32), triton.cdiv(dim, 64), 2 * heads)
            ](
                source,
                high,
                low,
                signs,
                out,
                blocks * tokens,
                heads,
                dim,
                inverse,
                tensor.dtype != torch.bfloat16,
                self.kind == "dct",
                32,
                64,
                32,
                num_warps=4,
            )
        return out

    def transform(self, layer_id, tensor, **kwargs):
        return self._apply(layer_id, tensor, False)

    def inverse(self, layer_id, tensor, **kwargs):
        return self._apply(layer_id, tensor, True)
