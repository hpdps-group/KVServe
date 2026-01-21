"""
KVServe codec adapter for KIVI bit-packing backend

Provides a thin wrapper around BitpackingCodec with a consistent interface
mirroring nvcomp_func.py style: initialization, parameter validation, encode,
and decode for KV cache compression/decompression.
"""

import torch
from kvserve.manager.components import Codec
from kvserve.codec.bitpacking_func import BitpackingCodec

class KIVICodec(Codec):
    """
    KVServe Codec implementation for KIVI

    Dispatches encode/decode to the BitpackingCodec backend.
    """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize KIVI codec wrapper

        Args:
            **kwargs: Configuration parameters, notably:
                codec_type: Backend codec type, currently "bitpacking" (default)
        """
        # Update parameters from kwargs
        self.codec_type = kwargs.get("codec_type", "bitpacking")

        self.codec = None

        # Validate codec parameters
        self.validate()

        # Initialize the codec based on codec_type
        match self.codec_type:
            case "bitpacking":
                self.codec = BitpackingCodec(**kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['bitpacking']")

    def validate(
        self,
    ) -> None:
        """
        Validate codec parameters before backend instantiation

        Raises:
            AssertionError: If codec_type is unsupported
        """
        assert self.codec_type in ["bitpacking"], \
            f"Invalid codec type: {self.codec_type}, expected one of: ['bitpacking']"

    def update_params(
        self,
        **kwargs
    ) -> None:
        """
        Update codec parameters dynamically

        Useful for switching backend or tuning settings between requests.

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
            case "bitpacking":
                self.codec = BitpackingCodec(**kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['bitpacking']")

    def encode(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode (compress) KV cache tensor to bytes using bit-packing backend

        Args:
            layer_id: Layer ID for encoding (unused, reserved for routing)
            tensor: KV cache tensor to compress, shape: [2, num_blocks, block_size, num_heads, head_size]
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

        Args:
            layer_id: Layer ID for decoding (unused, reserved for routing)
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
        
        