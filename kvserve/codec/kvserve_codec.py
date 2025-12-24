"""
KVServe Codec for KV cache compression
Implements codec compression using libraries like nvCOMP
"""

import torch
from kvserve.manager.components import Codec
from kvserve.codec import nvCOMPCodec

class KVServeCodec(Codec):
    """
    KVServe Codec implementation
    Supports nvCOMP compression library for lossless KV cache compression and other codecs
    """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize KVServe Codec
        
        Args:
            **kwargs: Configuration parameters including:
                codec_type: Type of codec, "nvcomp" (default: "nvcomp")
                nvcomp_algorithm: Compression algorithm, one of:
                    "ANS", "Bitcomp", "Cascaded", "Deflate", "GDeflate", "LZ4", "Zstd"
                    (default: "ANS")
        """
        # Update parameters from kwargs
        self.codec_type = kwargs.get("codec_type", "nvcomp")
        self.nvcomp_algorithm = kwargs.get("nvcomp_algorithm", "ANS")

        self.codec = None

        # Validate codec parameters
        self.validate()

        # Initialize the codec based on codec_type
        match self.codec_type:
            case "nvcomp":
                self.codec = nvCOMPCodec(algorithm=self.nvcomp_algorithm, **kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp']")

    def validate(
        self,
    ) -> None:
        """
        Validate codec parameters
        
        Raises:
            AssertionError: If codec_type or nvcomp_algorithm is invalid
        """
        assert self.codec_type in ["nvcomp"], \
            f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp']"
        assert self.nvcomp_algorithm in ["ANS", "Bitcomp", "Cascaded", "Deflate", "GDeflate", "LZ4", "Zstd"], \
            f"Invalid nvcomp's algorithm: {self.nvcomp_algorithm}"

    def update_params(
        self,
        **kwargs
    ) -> None:
        """
        Update codec parameters dynamically, used in encode() to update the codec parameters for every request
        
        Args:
            **kwargs: Parameters to update
        """
        # Update the parameters with the new values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Validate codec parameters
        self.validate()

        # Reinitialize codec with updated parameters
        match self.codec_type:
            case "nvcomp":
                self.codec = nvCOMPCodec(algorithm=self.nvcomp_algorithm, **kwargs)
            case _:
                raise ValueError(f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp']")

    def encode(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        **kwargs
    ) -> bytes:
        """
        Encode (compress) KV cache tensor to bytes
        
        Args:
            layer_id: Layer ID for encoding
            tensor: KV cache tensor to compress, shape: [2, num_blocks, block_size, num_heads, head_size]
            **kwargs: Additional encoding parameters
            
        Returns:
            Compressed bytes representation of the tensor
        """
        # Update codec parameters for every request
        # Only update parameters for the first layer
        if layer_id == 0:
            self.update_params(**kwargs)
    
        # Compress tensor to bytes
        comp_bytes = self.codec.encode(tensor, **kwargs)
        
        return comp_bytes

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
            layer_id: Layer ID for decoding
            compressed_data: Compressed bytes from encode()
            original_dtype: Original tensor dtype as string (e.g., "bfloat16", "float32")
            original_shape: Original tensor shape as list (e.g., [2, 128, 32, 8, 128])
            device: Target device for decompressed tensor (e.g., "cuda:0", "cpu")
            **kwargs: Additional decoding parameters
            
        Returns:
            Decompressed tensor with original dtype and shape restored
        """
        # Decompress bytes to tensor
        decomp_tensor = self.codec.decode(compressed_data, original_dtype, original_shape, device, **kwargs)
        
        return decomp_tensor
        
        