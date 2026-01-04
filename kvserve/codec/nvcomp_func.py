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
        
        # Boundary check: skip compression for very small tensors
        if tensor.numel() == 0:
            # Empty tensor, return empty bytes
            return b""
        
        # Log tensor size for debugging large data issues
        tensor_size_mb = tensor.numel() / (1024 * 1024)
        # if tensor_size_mb > 200:  # Log if > 200 MB
        #     print(f"[nvCOMP] Large tensor compression: {tensor_size_mb:.1f} MB, shape {list(tensor.shape)}")
        
        # Step 2: Convert PyTorch tensor to nvCOMP array format
        nv_array = nvcomp.as_array(tensor)
        
        # Step 3: Compress using nvCOMP
        comp_buffer = self.codec.encode(nv_array)
        
        # [MEMORY FIX] Release nv_array after encoding
        del nv_array
        
        # Step 4: Check if comp_buffer is valid (not scalar)
        # Handle boundary cases where nvCOMP returns scalar or invalid format
        try:
            # Debug: log comp_buffer type for large tensors
            # if tensor_size_mb > 200:
            #     print(f"[nvCOMP] comp_buffer type: {type(comp_buffer)}, "
            #           f"hasattr size: {hasattr(comp_buffer, 'size')}, "
            #           f"hasattr shape: {hasattr(comp_buffer, 'shape')}")
            #     if hasattr(comp_buffer, 'size'):
            #         print(f"[nvCOMP] comp_buffer.size: {comp_buffer.size}")
            #     if hasattr(comp_buffer, 'shape'):
            #         print(f"[nvCOMP] comp_buffer.shape: {comp_buffer.shape}")
            
            # Check if comp_buffer is a Python scalar (int, float, etc.)
            if isinstance(comp_buffer, (int, float, bool)):
                # Scalar value, skip compression - return original tensor as bytes
                print(f"[nvCOMP] SKIP: comp_buffer is Python scalar: {type(comp_buffer)}")
                return tensor.cpu().numpy().tobytes()
            
            # Try to convert to CuPy array
            comp_cupy = cp.asarray(comp_buffer)
            
            # Check if result is scalar (0-d array) or empty
            if comp_cupy.ndim == 0 or comp_cupy.size <= 1:
                # Scalar or single-element array, skip compression
                print(f"[nvCOMP] SKIP: comp_cupy is scalar/empty, ndim={comp_cupy.ndim}, size={comp_cupy.size}")
                return tensor.cpu().numpy().tobytes()
            
            # Normal case: convert to bytes
            comp_bytes = comp_cupy.tobytes()
            
            # [MEMORY FIX] Release CuPy arrays
            del comp_cupy
            del comp_buffer
            
            # Log compression ratio for large tensors
            # if tensor_size_mb > 200:
            #     comp_size_mb = len(comp_bytes) / (1024 * 1024)
            #     ratio = tensor_size_mb / comp_size_mb if comp_size_mb > 0 else 0
            #     print(f"[nvCOMP] Compressed: {tensor_size_mb:.1f} MB -> {comp_size_mb:.1f} MB (ratio: {ratio:.2f}x)")
            
            return comp_bytes
            
        except (ValueError, TypeError) as e:
            # Conversion failed (e.g., "cannot coerce scalar to array")
            # Fallback: return original tensor as bytes (skip compression)
            print(f"[nvCOMP] SKIP: Conversion failed: {type(e).__name__}: {e}")
            return tensor.cpu().numpy().tobytes()

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
        
        # [MEMORY FIX] Release CuPy array after conversion
        del comp_cupy

        # Step 3: Decompress using nvCOMP
        decomp_buffer = self.codec.decode(comp_buffer)
        
        # [MEMORY FIX] Release comp_buffer after decompression
        del comp_buffer
        
        # Step 4: Convert decompressed buffer to PyTorch tensor
        decomp_cupy = cp.asarray(decomp_buffer)
        decomp_tensor = torch.as_tensor(decomp_cupy, device=device)
        
        # Step 5: Restore original dtype
        reconstructed = decomp_tensor.view(getattr(torch, original_dtype))
        
        # Step 6: Restore original shape
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed