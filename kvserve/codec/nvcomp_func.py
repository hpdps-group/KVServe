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
    ) -> torch.Tensor:
        """
        Encode (compress) tensor to bytes using nvCOMP
        
        Converts tensor to uint8 view, compresses using nvCOMP, and returns
        compressed bytes. The tensor is flattened and viewed as uint8 for compression.
        
        Args:
            tensor: Input tensor to compress, any shape and dtype
            **kwargs: Additional encoding parameters
            
        Returns:
            Compressed tensor
        """
        # Step 1: Reshape tensor and convert to uint8 view
        flat_view = tensor.reshape(-1).view(torch.uint8)
        if not flat_view.is_contiguous():
            flat_view = flat_view.contiguous()
        
        # Boundary check: skip compression for very small tensors
        if flat_view.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=tensor.device)
        
        # Step 2: Convert PyTorch tensor to nvCOMP array format
        nv_array = nvcomp.as_array(flat_view)
        
        # Step 3: Compress using nvCOMP
        comp_buffer = self.codec.encode(nv_array)

        # Step 4: Convert nvCOMP buffer to PyTorch tensor
        comp_tensor = torch.as_tensor(comp_buffer, dtype=torch.uint8, device=tensor.device)
        
        # Release intermediates immediately
        del flat_view, nv_array, comp_buffer

        return comp_tensor
        
    def decode(
        self, 
        compressed_tensor: torch.Tensor,
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
            compressed_tensor: Compressed tensor from encode()
            original_dtype: Original tensor dtype as string (e.g., "bfloat16", "float32")
            original_shape: Original tensor shape as list (e.g., [128, 32, 8, 128])
            device: Target device for decompressed tensor (e.g., "cuda:0", "cpu")
            **kwargs: Additional decoding parameters
            
        Returns:
            Decompressed tensor with original dtype and shape restored
        """
        if not compressed_tensor.is_contiguous():
            compressed_tensor = compressed_tensor.contiguous()
        
        # Step 1: Convert to nvCOMP array format            
        comp_buffer = nvcomp.as_array(compressed_tensor)
        
        # Step 2: Decompress using nvCOMP
        decomp_buffer = self.codec.decode(comp_buffer)
        
        # Release input buffer wrapper
        del comp_buffer
        
        # Step 3: Convert decompressed buffer to PyTorch tensor
        decomp_tensor = torch.as_tensor(decomp_buffer, dtype=torch.uint8, device=device)
        
        # Step 4: Restore original dtype and shape
        target_dtype = getattr(torch, original_dtype)
        
        # Reshape directly on the view
        reconstructed = decomp_tensor.view(target_dtype).reshape(original_shape)
        
        # Release intermediate buffer
        del decomp_buffer, decomp_tensor
        
        return reconstructed