"""
Codec compression components for KV cache compression
"""

# Placeholder for future codec compression implementations
# Components should inherit from kvserve.manager.components.Codec
from kvserve.codec.nvcomp_func import nvCOMPCodec
from kvserve.codec.kvserve_codec import KVServeCodec
from kvserve.codec.cachegen_codec import CachegenCodec
from kvserve.codec.kivi_codec import KIVICodec

__all__ = [
    "KVServeCodec",
    "CachegenCodec",
    "KIVICodec",
]
