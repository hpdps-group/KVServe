"""
Quantizer components for KV cache compression
"""

from kvserve_v1.compression.quantizer.quantizer_func import quantize, dequantize
from kvserve_v1.compression.quantizer.split_func import (
    layer_split,
    layer_restore,
    layer_reconstruct,
    head_split,
    head_restore,
    head_reconstruct,
)

__all__ = [
    "quantize",
    "dequantize",
    "layer_split",
    "head_split",
    "head_restore",
    "layer_restore",
    "head_reconstruct",
    "layer_reconstruct",
    "KVServeQuantizer",
    "CenteredCodebookQuantizer",
]


def __getattr__(name):
    if name == "KVServeQuantizer":
        from .kvserve_quantizer import KVServeQuantizer

        return KVServeQuantizer
    if name == "CenteredCodebookQuantizer":
        from .centered_codebook import CenteredCodebookQuantizer

        return CenteredCodebookQuantizer
    raise AttributeError(name)
