"""
Compression Manager for KV cache compression
Coordinates transform, quantization, and lossless compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch

from kvserve.manager.components import Transform, Quantizer, LosslessCompression


@dataclass
class CompressionConfig:
    """Configuration for KV cache compression"""
    enabled: bool = False
    
    # Component configurations
    transform_config: Optional[Dict[str, Any]] = None
    quantizer_config: Optional[Dict[str, Any]] = None
    lossless_config: Optional[Dict[str, Any]] = None
    
    # Compression pipeline: which components to use (ordered list)
    pipeline: Optional[List[str]] = None  # e.g., ["transform", "quantizer", "lossless"] or ["quantizer"]
    
    # Threshold: don't compress if data size is below this (bytes)
    min_compress_size: int = 0


@dataclass
class CompressedKVData:
    """Compressed KV cache data structure"""
    request_id: str
    compressed_bytes: bytes
    metadata: Dict[str, Any]  # Includes compression config, original size, etc.
    original_size: int
    compressed_size: int


class CompressionManager:
    """
    Manages KV cache compression pipeline
    Coordinates transform, quantization, and lossless compression components
    """
    
    def __init__(
        self,
        config: CompressionConfig,
        transform: Optional[Transform] = None,
        quantizer: Optional[Quantizer] = None,
        lossless: Optional[LosslessCompression] = None,
    ):
        """
        Initialize Compression Manager
        
        Args:
            config: Compression configuration
            transform: Transform component (optional)
            quantizer: Quantizer component (optional)
            lossless: Lossless compression component (optional)
        """
        self.config = config
        self.transform = transform
        self.quantizer = quantizer
        self.lossless = lossless
        
        # Validate pipeline components are available
        if config.enabled and config.pipeline:
            for component_name in config.pipeline:
                if component_name == "transform" and self.transform is None:
                    raise ValueError("Transform component required but not provided")
                elif component_name == "quantizer" and self.quantizer is None:
                    raise ValueError("Quantizer component required but not provided")
                elif component_name == "lossless" and self.lossless is None:
                    raise ValueError("Lossless compression component required but not provided")
    
    def compress(
        self,
        kv_data: Any,  # torch.Tensor or similar
        request_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[CompressedKVData]:
        """
        Compress KV cache data using configured pipeline
        
        Args:
            kv_data: KV cache tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
            request_id: Request ID for tracking
            metadata: Additional metadata (block indices, etc.)
            
        Returns:
            CompressedKVData if compression successful, None if disabled or failed
        """
        if not self.config.enabled:
            return None
        
        if not self.config.pipeline:
            return None
        
        try:
            # Track original size
            original_size = self._get_data_size(kv_data)
            
            # Skip compression if below threshold
            if original_size < self.config.min_compress_size:
                return None
            
            # Apply pipeline components in order
            current_data = kv_data
            compression_metadata = {
                "request_id": request_id,
                "pipeline": self.config.pipeline.copy(),
                "original_size": original_size,
                **metadata,
            }
            
            # Step 1: Transform (if in pipeline)
            if "transform" in self.config.pipeline:
                current_data = self.transform.encode(
                    current_data,
                    self.config.transform_config or {}
                )
                compression_metadata["transform_applied"] = True
            
            # Step 2: Quantization (if in pipeline)
            quantization_params = None
            if "quantizer" in self.config.pipeline:
                current_data, quantization_params = self.quantizer.quantize(
                    current_data,
                    self.config.quantizer_config or {}
                )
                compression_metadata["quantization_params"] = quantization_params
                compression_metadata["quantization_applied"] = True
            
            # Step 3: Lossless compression (if in pipeline)
            # Save tensor shape and dtype before converting to bytes
            if isinstance(current_data, torch.Tensor):
                compression_metadata["tensor_shape"] = list(current_data.shape)
                compression_metadata["tensor_dtype"] = str(current_data.dtype).replace("torch.", "")
            
            if "lossless" in self.config.pipeline:
                # Convert to bytes first (implementation depends on data format)
                data_bytes = self._tensor_to_bytes(current_data)
                compressed_bytes = self.lossless.compress(
                    data_bytes,
                    self.config.lossless_config or {}
                )
                compression_metadata["lossless_applied"] = True
            else:
                # If no lossless compression, just convert to bytes
                compressed_bytes = self._tensor_to_bytes(current_data)
            
            compressed_size = len(compressed_bytes)
            
            return CompressedKVData(
                request_id=request_id,
                compressed_bytes=compressed_bytes,
                metadata=compression_metadata,
                original_size=original_size,
                compressed_size=compressed_size,
            )
            
        except Exception as e:
            # Compression failed, return None to fallback to uncompressed
            print(f"[CompressionManager] Compression failed for {request_id}: {e}")
            return None
    
    def decompress(
        self,
        compressed_data: CompressedKVData,
    ) -> Optional[Any]:  # Returns torch.Tensor or similar
        """
        Decompress KV cache data using configured pipeline
        
        Args:
            compressed_data: CompressedKVData from compress()
            
        Returns:
            Decompressed KV cache tensor, None if decompression failed
        """
        if not self.config.enabled:
            return None
        
        try:
            pipeline = compressed_data.metadata.get("pipeline", [])
            if not pipeline:
                return None
            
            current_data = compressed_data.compressed_bytes
            
            # Step 1: Lossless decompression (reverse order)
            if "lossless" in pipeline:
                current_data = self.lossless.decompress(
                    current_data,
                    self.config.lossless_config or {}
                )
            
            # Step 2: Dequantization (if in pipeline)
            if "quantizer" in pipeline:
                quantization_params = compressed_data.metadata.get("quantization_params")
                if quantization_params is None:
                    raise ValueError("Quantization params missing in compressed data")
                
                # Convert bytes to tensor first
                tensor_data = self._bytes_to_tensor(current_data, compressed_data.metadata)
                current_data = self.quantizer.dequantize(
                    tensor_data,
                    quantization_params,
                    self.config.quantizer_config or {}
                )
            else:
                # Convert bytes to tensor
                current_data = self._bytes_to_tensor(current_data, compressed_data.metadata)
            
            # Step 3: Transform decode (if in pipeline)
            if "transform" in pipeline:
                current_data = self.transform.decode(
                    current_data,
                    self.config.transform_config or {}
                )
            
            return current_data
            
        except Exception as e:
            print(f"[CompressionManager] Decompression failed for {compressed_data.request_id}: {e}")
            return None
    
    def _get_data_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        import torch
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, bytes):
            return len(data)
        else:
            # Fallback: serialize to estimate size
            import pickle
            return len(pickle.dumps(data))
    
    def _tensor_to_bytes(self, tensor: Any) -> bytes:
        """Convert tensor to bytes"""
        import torch
        if isinstance(tensor, torch.Tensor):
            # Simple serialization (can be optimized)
            return tensor.cpu().numpy().tobytes()
        elif isinstance(tensor, bytes):
            return tensor
        else:
            import pickle
            return pickle.dumps(tensor)
    
    def _bytes_to_tensor(self, data_bytes: bytes, metadata: Dict[str, Any]) -> Any:
        """
        Convert bytes back to tensor
        
        Note: This is a simplified implementation.
        In practice, you need to preserve tensor shape and dtype information in metadata.
        """
        import torch
        import numpy as np
        
        # Extract shape and dtype from metadata if available
        shape = metadata.get("tensor_shape")
        dtype_str = metadata.get("tensor_dtype", "float16")
        
        if shape:
            # Map torch dtype to numpy dtype
            dtype_map = {
                "float16": np.float16,
                "float32": np.float32,
                "bfloat16": np.float16,  # bfloat16 may need special handling
            }
            np_dtype = dtype_map.get(dtype_str, np.float16)
            np_array = np.frombuffer(data_bytes, dtype=np_dtype)
            tensor = torch.from_numpy(np_array.reshape(shape))
            return tensor
        else:
            # Fallback: assume it was pickled
            import pickle
            return pickle.loads(data_bytes)

