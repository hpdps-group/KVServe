"""
KVServe codec factory/adapter for CacheGen integration

Provides a unified Codec implementation that dispatches to concrete codecs
current supported: TorchAC (entropy coding). Designed to mirror the style of
nvcomp_func.py, exposing clear initialization, validation, encode, and decode
APIs for KV cache compression/decompression.
"""

import torch
from kvserve.manager.components import Codec
from kvserve.codec.torchac_func import TorchACCodec

class CachegenCodec(Codec):
    """
    KVServe Codec implementation for CacheGen

    Acts as a thin wrapper that instantiates the chosen backend codec and
    forwards encode/decode calls. Currently supports TorchAC for entropy coding.
    """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize KVServe codec wrapper

        Args:
            **kwargs: Configuration parameters, notably:
                codec_type: Codec backend to use, currently "torchac" (default)
        """
        # Update parameters from kwargs
        self.codec_type = kwargs.get("codec_type", "torchac")

        self.codec = None

        # Validate codec parameters
        self.validate()

        # Initialize the codec based on codec_type
        match self.codec_type:
            case "torchac":
                self.codec = TorchACCodec(**kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['torchac']")

    def validate(
        self,
    ) -> None:
        """
        Validate codec parameters before instantiating backend

        Raises:
            AssertionError: If codec_type is unsupported
        """
        assert self.codec_type in ["torchac"], \
            f"Invalid codec type: {self.codec_type}, expected one of: ['torchac']"

    def update_params(
        self,
        **kwargs
    ) -> None:
        """
        Update codec parameters dynamically

        Can be used to swap codec backend or adjust settings between requests.

        Args:
            **kwargs: Parameters to update on the wrapper
        """
        # Update the parameters with the new values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Validate codec parameters
        self.validate()

        # Reinitialize codec with updated parameters
        match self.codec_type:
            case "torchac":
                self.codec = TorchACCodec(**kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['torchac']")

    def encode(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode (compress) KV cache tensor to bytes

        Forwards the tensor to the configured backend codec.

        Args:
            layer_id: Layer ID for encoding (unused, reserved for future routing)
            tensor: KV cache tensor to compress, shape: [32, 2, num_blocks, block_size, num_heads, head_size]
            **kwargs: Additional backend-specific encoding parameters

        Returns:
            Compressed tensor produced by backend codec
        """
        # Don't need to update parameters here because compression manager will handle it
        # if layer_id == 0:
        #     self.update_params(**kwargs)
    
        # Compress tensor to bytes
        comp_tensor = self.codec.encode(tensor, **kwargs)
        
        return comp_tensor

    def decode(
        self,
        layer_id: int,
        compressed_data: bytes,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode (decompress) compressed bytes back to KV cache tensor

        Forwards compressed data to the configured backend codec.

        Args:
            layer_id: Layer ID for decoding (unused, reserved for future routing)
            compressed_data: Compressed bytes from encode()
            original_dtype: Original tensor dtype name (e.g., "bfloat16", "float32")
            original_shape: Original tensor shape list (e.g., [32, 2, 128, 32, 8, 128])
            device: Target device for decompressed tensor (e.g., "cuda:0", "cpu")
            **kwargs: Additional backend-specific decoding parameters

        Returns:
            Tensor restored to original dtype and shape
        """
        # Decompress bytes to tensor
        decomp_tensor = self.codec.decode(compressed_data, original_dtype, original_shape, device, **kwargs)
        
        return decomp_tensor
        
        