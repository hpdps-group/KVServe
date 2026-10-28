"""
Codec compression components for KV cache compression
"""

from kvserve_v1.compression.codec.kvserve_codec import KVServeCodec
from kvserve_v1.compression.codec.lc_codec import LCCodec

__all__ = [
    "nvCOMPCodec",
    "KVServeCodec",
    "LCCodec",
    "BitPackCodec",
]


def __getattr__(name):
    if name == "BitPackCodec":
        from kvserve_v1.compression.codec.bitpack_codec import BitPackCodec
        return BitPackCodec
    if name == "nvCOMPCodec":
        from kvserve_v1.compression.codec.nvcomp_func import nvCOMPCodec
        return nvCOMPCodec
    raise AttributeError(name)
