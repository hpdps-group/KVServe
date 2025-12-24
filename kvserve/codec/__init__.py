"""
Codec compression components for KV cache compression
"""

# Placeholder for future codec compression implementations
# Components should inherit from kvserve.manager.components.Codec
from kvserve.codec.nvcomp_func import nvCOMPCodec
from kvserve.codec.kvserve_codec import KVServeCodec

__all__ = [
    "nvCOMPCodec",
    "KVServeCodec",
]


