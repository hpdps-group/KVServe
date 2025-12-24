"""
nvCOMP compression functions for KV cache compression
Implements GPU-accelerated lossless compression using NVIDIA nvCOMP library
"""

import torch
import cupy as cp
from nvidia import nvcomp

class nvCOMPCodec:
    """
    nvCOMP Codec wrapper for GPU-accelerated compression
    
    Provides interface to NVIDIA nvCOMP compression library for lossless
    compression/decompression of PyTorch tensors.
    """
    def __init__(
        self, 
        algorithm: str = "ANS", 
        **kwargs
    ) -> None:
        """
        Initialize nvCOMP Codec
        
        Args:
            algorithm: Compression algorithm to use, one of:
                "ANS", "Bitcomp", "Cascaded", "Deflate", "GDeflate", "LZ4", "Zstd"
                (default: "ANS")
            **kwargs: Additional nvCOMP codec configuration parameters
        """
        self.codec = nvcomp.Codec(algorithm=algorithm, **kwargs)

    def encode(
        self, 
        tensor: torch.Tensor, 
        **kwargs
    ) -> bytes:
        """
        Encode (compress) tensor to bytes using nvCOMP
        
        Converts tensor to uint8 view, compresses using nvCOMP, and returns
        compressed bytes. The tensor is flattened and viewed as uint8 for compression.
        
        Args:
            tensor: Input tensor to compress, any shape and dtype
            **kwargs: Additional encoding parameters
            
        Returns:
            Compressed bytes representation of the tensor
        """
        # Step 1: Flatten tensor and convert to uint8 view
        tensor = tensor.flatten().view(torch.uint8).contiguous()
        
        # Step 2: Convert PyTorch tensor to nvCOMP array format
        nv_array = nvcomp.as_array(tensor)
        
        # Step 3: Compress using nvCOMP
        comp_buffer = self.codec.encode(nv_array)
        
        # Step 4: Convert compressed buffer to Python bytes
        comp_bytes = cp.asarray(comp_buffer).tobytes()

        return comp_bytes

    def decode(
        self, 
        compressed_data: bytes,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode (decompress) compressed bytes back to tensor using nvCOMP
        
        Decompresses bytes using nvCOMP, converts to PyTorch tensor, and restores
        original dtype and shape.
        
        Args:
            compressed_data: Compressed bytes from encode()
            original_dtype: Original tensor dtype as string (e.g., "bfloat16", "float32")
            original_shape: Original tensor shape as list (e.g., [128, 32, 8, 128])
            device: Target device for decompressed tensor (e.g., "cuda:0", "cpu")
            **kwargs: Additional decoding parameters
            
        Returns:
            Decompressed tensor with original dtype and shape restored
        """
        # Step 1: Convert compressed bytes to CuPy array
        comp_cupy = cp.frombuffer(compressed_data, dtype=cp.uint8)
        
        # Step 2: Convert CuPy array to nvCOMP array format
        comp_buffer = nvcomp.as_array(comp_cupy)

        # Step 3: Decompress using nvCOMP
        decomp_buffer = self.codec.decode(comp_buffer)
        
        # Step 4: Convert decompressed buffer to PyTorch tensor
        decomp_cupy = cp.asarray(decomp_buffer)
        decomp_tensor = torch.as_tensor(decomp_cupy, device=device)
        
        # Step 5: Restore original dtype
        reconstructed = decomp_tensor.view(getattr(torch, original_dtype))
        
        # Step 6: Restore original shape
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed