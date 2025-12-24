"""
Compression Manager for KV cache compression
Coordinates transformer, quantizer, and codec compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch

from kvserve.manager.components import Transformer, Quantizer, Codec
from kvserve.quantizer import KVServeQuantizer


@dataclass
class CompressionConfig:
    """Configuration for KV cache compression"""
    enabled: bool = False
    
    # Component configurations
    transformer_config: Optional[Dict[str, Any]] = None
    quantizer_config: Optional[Dict[str, Any]] = None
    codec_config: Optional[Dict[str, Any]] = None
    
    # Compression pipeline: which components to use (ordered list)
    pipeline: Optional[List[str]] = None  # e.g., ["transformer", "quantizer", "codec"] or ["quantizer"]
    
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
    Coordinates transformer, quantizer, and codec compression components
    """
    
    def __init__(
        self,
        config: CompressionConfig,
        transformer: Optional[Transformer] = None,
        quantizer: Optional[Quantizer] = None,
        codec: Optional[Codec] = None,
    ):
        """
        Initialize Compression Manager
        
        Args:
            config: Compression configuration
            transformer: Transformer component (optional)
            quantizer: Quantizer component (optional)
            codec: Codec compression component (optional)
        """
        self.config = config
        self.transformer = transformer(**config.transformer_config) if transformer else None
        self.quantizer = quantizer(**config.quantizer_config) if quantizer else None
        self.codec = codec(**config.codec_config) if codec else None
        
        # Validate pipeline components are available
        if config.enabled and config.pipeline:
            for component_name in config.pipeline:
                if component_name == "transformer" and self.transformer is None:
                    raise ValueError("Transformer component required but not provided")
                elif component_name == "quantizer" and self.quantizer is None:
                    raise ValueError("Quantizer component required but not provided")
                elif component_name == "codec" and self.codec is None:
                    raise ValueError("Codec compression component required but not provided")
    
    def compress(
        self,
        layer_id: int,
        kv_data: Any,  # torch.Tensor or similar
        request_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[CompressedKVData]:
        """
        Compress KV cache data using configured pipeline
        
        Args:
            kv_data: KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
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
                "device": kv_data.device,
                **metadata,
            }
            
            # Step 1: Transformer (if in pipeline)
            if "transformer" in self.config.pipeline:
                current_data = self.transformer.transform(
                    layer_id,
                    current_data,
                    **self.config.transformer_config
                )
                compression_metadata["transformer_applied"] = True
            
            # Step 2: Quantization (if in pipeline)
            quantization_params = None
            if "quantizer" in self.config.pipeline:
                current_data, quantization_params = self.quantizer.quantize(
                    layer_id,
                    current_data,
                    **self.config.quantizer_config
                )
                compression_metadata["quantization_params"] = quantization_params
                compression_metadata["quantization_applied"] = True
            
            # Step 3: Codec compression (if in pipeline)
            # Save tensor shape and dtype before converting to bytes
            if isinstance(current_data, torch.Tensor):
                compression_metadata["original_shape"] = list(current_data.shape)
                compression_metadata["original_dtype"] = str(current_data.dtype).replace("torch.", "")
            
            if "codec" in self.config.pipeline:
                # Convert to bytes first (implementation depends on data format)
                compressed_bytes = self.codec.encode(
                    layer_id,
                    current_data,
                    **self.config.codec_config
                )
                compression_metadata["codec_applied"] = True
            else:
                # If no codec compression, just convert to bytes
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
        layer_id: int,
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
            
            # Step 1: Codec decompression (reverse order)
            if "codec" in pipeline:
                original_dtype = compressed_data.metadata.get("original_dtype")
                original_shape = compressed_data.metadata.get("original_shape")
                device = compressed_data.metadata.get("device")
                assert original_dtype is not None and original_shape is not None and device is not None, \
                    "Original dtype, shape, and device are required for decompression"
                current_data = self.codec.decode(
                    layer_id,
                    current_data,
                    original_dtype,
                    original_shape,
                    device,
                    **self.config.codec_config
                )
            
            # Step 2: Dequantization (if in pipeline)
            if "quantizer" in pipeline:
                quantization_params = compressed_data.metadata.get("quantization_params")
                assert quantization_params is not None, \
                    "Quantization params are required for dequantization"
                
                # Convert bytes to tensor first
                tensor_data = self._bytes_to_tensor(current_data, compressed_data.metadata)
                current_data = self.quantizer.dequantize(
                    layer_id,
                    tensor_data,
                    quantization_params,
                    **self.config.quantizer_config
                )
            else:
                # Convert bytes to tensor
                current_data = self._bytes_to_tensor(current_data, compressed_data.metadata)
            
            # Step 3: Transformer  (if in pipeline)
            if "transformer" in pipeline:
                current_data = self.transformer.inverse(
                    layer_id,
                    current_data,
                    **self.config.transformer_config
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
    
    def _bytes_to_tensor(self, data_bytes: Any, metadata: Dict[str, Any]) -> Any:
        """
        Convert bytes back to tensor
        
        Note: This is a simplified implementation.
        In practice, you need to preserve tensor shape and dtype information in metadata.
        """
        import torch
        import numpy as np
        
        if isinstance(data_bytes, torch.Tensor):
            return data_bytes

        # Extract shape and dtype from metadata if available
        shape = metadata.get("original_shape")
        dtype_str = metadata.get("original_dtype", "float16")
        
        if shape:
            # Map torch dtype to numpy dtype
            dtype_map = {
                "float32": np.float32,
                "float16": np.float16,
                "bfloat16": np.float16,  # bfloat16 may need special handling
                "uint8": np.uint8,
            }
            np_dtype = dtype_map.get(dtype_str, np.float16)
            np_array = np.frombuffer(data_bytes, dtype=np_dtype)
            tensor = torch.from_numpy(np_array.reshape(shape).copy())
            return tensor.to(metadata.get("device", "cpu"))
        else:
            # Fallback: assume it was pickled
            import pickle
            return pickle.loads(data_bytes)

