"""
Codec compression components for KV cache compression
"""

import importlib

__all__ = [
    "KVServeCodec",
    "CachegenCodec",
    "KIVICodec",
    "nvCOMPCodec",
]


_LAZY_IMPORTS = {
    "KVServeCodec": "kvserve.codec.kvserve_codec",
    "CachegenCodec": "kvserve.codec.cachegen_codec",
    "KIVICodec": "kvserve.codec.kivi_codec",
    "nvCOMPCodec": "kvserve.codec.nvcomp_func",
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if not module_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
