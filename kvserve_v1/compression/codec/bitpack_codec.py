"""Lossless, fixed-width coding of bounded CUDA uint8 symbols.

The 16-byte GPU header identifies the format, bit width and symbol count.
Symbols are packed least-significant-bit first in contiguous tensor order.
CUDA assertions reject out-of-range symbols and mismatched decode contracts;
they are asynchronous and invalidate the CUDA context on invalid input.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl

from kvserve_v1.compression.components import Codec

_HEADER_BYTES = 16
_MAGIC = tl.constexpr(0x0150424B)  # "KBP", format version 1


@triton.jit
def _pack(source, output, N: tl.constexpr, BITS: tl.constexpr,
          BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size = (N * BITS + 7) // 8
    first = byte * 8 // BITS
    offset = byte * 8 % BITS
    value = tl.load(source + first, (byte < size) & (first < N), 0).to(tl.uint32)
    tl.device_assert(value < (1 << BITS), "symbol exceeds bitpack_bits")
    packed = value >> offset
    for j in tl.static_range(1, (7 + BITS - 1) // BITS + 1):
        shift = j * BITS - offset
        value = tl.load(source + first + j,
                        (byte < size) & (first + j < N) & (shift < 8), 0).to(tl.uint32)
        tl.device_assert(value < (1 << BITS), "symbol exceeds bitpack_bits")
        packed = packed | (value << shift)
    tl.store(output + 16 + byte, packed.to(tl.uint8), byte < size)
    if tl.program_id(0) == 0:
        h = tl.arange(0, 16)
        count = tl.full((16,), N, tl.uint64)
        header = tl.where(h < 4, (_MAGIC >> ((h % 4) * 8)) & 255,
                          tl.where(h == 4, BITS,
                                   tl.where(h >= 8, (count >> ((h % 8) * 8)) & 255, 0)))
        tl.store(output + h, header.to(tl.uint8))


@triton.jit
def _unpack(source, output, N: tl.constexpr, BITS: tl.constexpr,
            BLOCK: tl.constexpr):
    if tl.program_id(0) == 0:
        h = tl.arange(0, 16)
        actual = tl.load(source + h).to(tl.uint32)
        count = tl.full((16,), N, tl.uint64)
        expected = tl.where(h < 4, (_MAGIC >> ((h % 4) * 8)) & 255,
                            tl.where(h == 4, BITS,
                                     tl.where(h >= 8, (count >> ((h % 8) * 8)) & 255, 0)))
        tl.device_assert(actual == expected, "invalid bitpack header or decode configuration")
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size = (N * BITS + 7) // 8
    byte = i * BITS // 8
    offset = i * BITS % 8
    lo = tl.load(source + 16 + byte, (i < N) & (byte < size), 0).to(tl.uint32)
    hi = tl.load(source + 16 + byte + 1,
                 (i < N) & (byte + 1 < size) & (offset + BITS > 8), 0).to(tl.uint32)
    value = ((lo | (hi << 8)) >> offset) & ((1 << BITS) - 1)
    tl.store(output + i, value.to(tl.uint8), i < N)


class BitPackCodec(Codec):
    """A GPU-only codec for uint8 symbols in ``[0, 2**bitpack_bits)``.

    The default width 8 accepts every byte. Set a smaller width only when the
    upstream quantizer guarantees the corresponding alphabet. Quantization
    remains lossy; this codec preserves its integer symbols exactly.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.bitpack_bits = 8
        self.update_params(**kwargs)

    def update_params(self, **kwargs: Any) -> None:
        self.bitpack_bits = self._width(kwargs)

    def _width(self, kwargs: dict[str, Any]) -> int:
        bits = kwargs.get("bitpack_bits", self.bitpack_bits)
        if type(bits) is not int or not 1 <= bits <= 8:
            raise ValueError("bitpack_bits must be an integer in [1, 8]")
        return bits

    @staticmethod
    def _check_tensor(tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
            raise TypeError("BitPackCodec requires a torch.uint8 tensor")
        if not tensor.is_cuda:
            raise ValueError("BitPackCodec requires a CUDA tensor")

    def encode(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        bits = self._width(kwargs)
        self._check_tensor(tensor)
        source = tensor.contiguous()
        count = source.numel()
        size = _HEADER_BYTES + (count * bits + 7) // 8
        output = torch.empty(size, device=source.device, dtype=torch.uint8)
        with torch.cuda.device(source.device):
            _pack[(max(1, triton.cdiv(size - _HEADER_BYTES, 1024)),)](
                source, output, count, bits, 1024, debug=True)
        return output

    def decode(self, compressed_data: torch.Tensor, original_dtype: str,
               original_shape: list, device: str, **kwargs: Any) -> torch.Tensor:
        bits = self._width(kwargs)
        self._check_tensor(compressed_data)
        if str(original_dtype).removeprefix("torch.") != "uint8":
            raise TypeError("BitPackCodec restores uint8 symbols only")
        if any(type(d) is not int or d < 0 for d in original_shape):
            raise ValueError("original_shape must contain nonnegative integers")
        target_device = torch.device(device)
        with torch.cuda.device(compressed_data.device):
            if target_device.type != "cuda" or (
                target_device.index is not None
                and target_device.index != compressed_data.device.index
            ):
                raise ValueError("BitPackCodec decodes on the payload's CUDA device")
            count = math.prod(original_shape)
            size = _HEADER_BYTES + (count * bits + 7) // 8
            if compressed_data.numel() != size:
                raise ValueError("BitPackCodec payload length does not match shape and bit width")
            source = compressed_data.contiguous()
            output = torch.empty(original_shape, device=source.device, dtype=torch.uint8)
            _unpack[(max(1, triton.cdiv(count, 1024)),)](
                source, output, count, bits, 1024, debug=True)
        return output
