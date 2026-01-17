"""
Lightweight helpers to reuse kvserve's CompressionManager in the simulator.

Key points:
- Does NOT perform any networking or NCCL setup.
- Works with CPU tensors by default to avoid GPU pressure inside the simulator.
- Provides thin wrappers for all-layer compression / decompression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from kvserve.manager.compression_manager import (
    CompressionConfig,
    CompressionManager,
    CompressedKVData,
    get_default_compression_config,
)
from kvserve.transformer import KVServeTransformer
from kvserve.quantizer import KVServeQuantizer
from kvserve.codec import KVServeCodec


def build_compression_manager(config_dict: Optional[Dict[str, Any]]) -> Optional[CompressionManager]:
    """
    Create a CompressionManager from a plain dict configuration.
    Returns None when compression is disabled or config is missing.
    """
    if not config_dict:
        config_dict = get_default_compression_config()

    config = CompressionConfig(**config_dict)
    if not config.enabled or not config.pipeline:
        return None

    # Components are optional; only instantiate those present in the pipeline
    transformer = KVServeTransformer if "transformer" in config.pipeline else None
    quantizer = KVServeQuantizer if "quantizer" in config.pipeline else None
    codec = KVServeCodec if "codec" in config.pipeline else None

    return CompressionManager(
        config=config,
        transformer=transformer,
        quantizer=quantizer,
        codec=codec,
    )


def compress_all_layers(
    manager: CompressionManager,
    tensor: torch.Tensor,
    request_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[CompressedKVData]:
    """
    Compress a 6D KV tensor using the fast-path API.
    """
    if manager is None:
        return None
    return manager.compress_all_layers(
        all_layers_data=tensor,
        request_id=request_id,
        metadata=metadata or {},
    )


def decompress_all_layers(
    manager: CompressionManager,
    compressed: CompressedKVData,
) -> Optional[torch.Tensor]:
    """
    Decompress data produced by compress_all_layers().
    """
    if manager is None:
        return None
    return manager.decompress_all_layers(compressed)

