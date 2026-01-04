"""
Compression Manager for KV cache compression
Coordinates transformer, quantizer, and codec compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch

from kvserve.manager.components import Transformer, Quantizer, Codec
from kvserve.quantizer import KVServeQuantizer
from kvserve.engine.logger import log_info, log_error, log_warning


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
    layer_id: int
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

        # Cross-layer buffers (for batch compression)
        self.tensor_buffer: Optional[torch.Tensor] = None
        self.metadata_buffer: List[Any] = []
        self.buffer_capacity: int = 0
        self.buffer_fill: int = 0
        self.original_size: int = 0
        self.compressed_size: int = 0
        self.reconstructed_size: int = 0
        
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
        num_buffer_layers: int,
        num_total_layers: int,
        kv_data: Any,  # torch.Tensor or similar
        request_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[CompressedKVData]:
        """
        Compress KV cache data using configured pipeline (buffered multi-layer).
        Set num_buffer_layers==num_total_layers to compress all layers in one batch.
        """
        if not self.config.enabled:
            return None
        if not self.config.pipeline:
            return None

        try:
            original_size = self._get_data_size(kv_data)
            if original_size < self.config.min_compress_size:
                return None

            # Init buffers and refresh component params on first layer of a request
            if layer_id == 0 or self.tensor_buffer is None:
                self.buffer_capacity = num_buffer_layers
                self.buffer_fill = 0
                self.tensor_buffer = None
                self.metadata_buffer = []
                self.original_size = 0
                self.compressed_size = 0
                self.reconstructed_size = 0
                if self.transformer is not None:
                    self.transformer.update_params(**self.config.transformer_config)
                if self.quantizer is not None:
                    self.quantizer.update_params(**self.config.quantizer_config)
                if self.codec is not None:
                    self.codec.update_params(**self.config.codec_config)

            self.original_size += original_size

            current_data = kv_data
            compression_metadata = {
                "request_id": request_id,
                "pipeline": self.config.pipeline.copy(),
                "original_size": original_size,
                "device": kv_data.device,
                "start_layer_id": layer_id,  # will be overwritten when flushing
                **metadata,
            }

            # Step 1: Transformer
            if "transformer" in self.config.pipeline:
                current_data = self.transformer.transform(
                    layer_id,
                    current_data,
                    **self.config.transformer_config,
                )
                compression_metadata["transformer_applied"] = True

            # Step 2: Quantization
            quantization_params = None
            if "quantizer" in self.config.pipeline:
                current_data, quantization_params = self.quantizer.quantize(
                    layer_id,
                    current_data,
                    **self.config.quantizer_config,
                )
                compression_metadata["quantization_applied"] = True

            # Pre-allocate buffer to avoid an extra stack peak
            if self.tensor_buffer is None:
                if not isinstance(current_data, torch.Tensor):
                    # Fallback to list accumulation if non-tensor
                    self.tensor_buffer = [None] * self.buffer_capacity  # type: ignore
                else:
                    full_shape = (self.buffer_capacity,) + current_data.shape
                    self.tensor_buffer = torch.empty(
                        full_shape, dtype=current_data.dtype, device=current_data.device
                    )

            # Write into buffer
            if isinstance(self.tensor_buffer, list):
                self.tensor_buffer[self.buffer_fill] = current_data  # type: ignore
            else:
                self.tensor_buffer[self.buffer_fill].copy_(current_data)
            self.metadata_buffer.append(quantization_params)
            self.buffer_fill += 1

            # Decide flush
            is_last_layer = layer_id == num_total_layers - 1
            should_flush = self.buffer_fill == self.buffer_capacity or is_last_layer
            if not should_flush:
                return None

            # Slice valid portion
            if isinstance(self.tensor_buffer, list):
                batch_tensor = self.tensor_buffer[: self.buffer_fill]
                batch_tensor = torch.stack(batch_tensor, dim=0)
            else:
                batch_tensor = self.tensor_buffer[: self.buffer_fill]

            # Permute to codec-friendly layout
            if isinstance(batch_tensor, torch.Tensor):
                batch_tensor = batch_tensor.permute(0, 1, 4, 2, 3, 5)
                compression_metadata["compressed_shape"] = list(batch_tensor.shape)
                compression_metadata["compressed_dtype"] = str(batch_tensor.dtype).replace("torch.", "")

            # Record start layer id and num layers for this batch
            start_layer_id = layer_id - self.buffer_fill + 1
            compression_metadata["start_layer_id"] = start_layer_id
            compression_metadata["num_layers"] = self.buffer_fill

            # Attach per-layer quantization params
            compression_metadata["quantization_params"] = self.metadata_buffer.copy()

            # Codec compression with safeguards
            compressed_bytes, compression_metadata = self._handle_codec_compression(
                batch_tensor, compression_metadata, layer_id, request_id
            )
            # Use actual payload bytes as compressed_size; metadata is sent separately
            compressed_size = len(compressed_bytes)
            self.compressed_size += compressed_size

            # Clear buffers for next batch
            self.tensor_buffer = None
            self.metadata_buffer = []
            self.buffer_fill = 0
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            return CompressedKVData(
                request_id=request_id,
                layer_id=layer_id,
                compressed_bytes=compressed_bytes,
                metadata=compression_metadata,
                original_size=self.original_size,
                compressed_size=self.compressed_size,
            )
        except Exception as e:
            log_error(f"[CompressionManager] Compression failed for {request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def decompress(
        self,
        layer_id: int,
        num_buffer_layers: int,
        num_total_layers: int,
        compressed_data: CompressedKVData,
    ) -> Optional[Any]:
        """
        Decompress buffered multi-layer data produced by compress().
        Returns a tensor stacked for the flushed batch.
        """
        if not self.config.enabled:
            return None

        try:
            pipeline = compressed_data.metadata.get("pipeline", [])
            current_data = compressed_data.compressed_bytes

            # Step 1: Codec decompress (may return bytes if skipped)
            current_data = self._handle_codec_decompression(
                current_data, compressed_data, pipeline, layer_id
            )

            # If still bytes, convert using stored shape/dtype
            if isinstance(current_data, bytes):
                current_data = self._bytes_to_tensor(current_data, compressed_data.metadata)

            if isinstance(current_data, torch.Tensor):
                # Permute back to [layers, 2, num_blocks, block_size, num_heads, head_size]
                if current_data.dim() == 6:
                    current_data = current_data.permute(0, 1, 3, 4, 2, 5)
                current_data = current_data.to(
                    getattr(torch, compressed_data.metadata.get("original_dtype", "bfloat16"), torch.bfloat16)
                )

            # Step 2: Dequantize (batch-capable)
            if "quantizer" in pipeline:
                quantization_params = compressed_data.metadata.get("quantization_params")
                assert quantization_params is not None, "Quantization params are required for dequantization"
                if not isinstance(current_data, torch.Tensor):
                    current_data = self._bytes_to_tensor(current_data, compressed_data.metadata)

                start_layer_id = compressed_data.metadata.get("start_layer_id", 0)
                layer_end_id = start_layer_id + (current_data.shape[0] - 1 if current_data.dim() == 6 else 0)
                
                # Use batch dequantize (supports both single and multi-layer)
                current_data = self.quantizer.dequantize(
                    layer_end_id,
                    current_data,
                    quantization_params,
                    **self.config.quantizer_config,
                )

            # Step 3: Transformer inverse
            if "transformer" in pipeline and isinstance(current_data, torch.Tensor):
                restored_layers = []
                if current_data.dim() == 6:
                    start_layer_id = compressed_data.metadata.get("start_layer_id", 0)
                    for i in range(current_data.shape[0]):
                        restored_layers.append(
                            self.transformer.inverse(
                                start_layer_id + i,
                                current_data[i],
                                **self.config.transformer_config,
                            )
                        )
                    current_data = torch.stack(restored_layers, dim=0)
                else:
                    current_data = self.transformer.inverse(
                        layer_id,
                        current_data,
                        **self.config.transformer_config,
                    )

            if isinstance(current_data, list):
                current_data = torch.stack(current_data, dim=0)

            self.reconstructed_size += self._get_data_size(current_data)
            return current_data
        except Exception as e:
            log_error(f"[CompressionManager] Decompression failed for {compressed_data.request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def compress_all_layers(
        self,
        all_layers_data: Any,  # torch.Tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        request_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[CompressedKVData]:
        """
        Optimized fast path: compress all layers in a single batch with minimal overhead.
        Avoids repeated function calls by processing all layers in a tight loop.
        """
        if not self.config.enabled or not self.config.pipeline:
            return None
        if not isinstance(all_layers_data, torch.Tensor) or all_layers_data.dim() != 6:
            log_error(f"[CompressionManager] all_layers_data must be 6D torch.Tensor, got {type(all_layers_data)}, dim={all_layers_data.dim() if isinstance(all_layers_data, torch.Tensor) else 'N/A'}")
            return None

        try:
            num_layers = all_layers_data.shape[0]
            original_size = self._get_data_size(all_layers_data)
            
            if original_size < self.config.min_compress_size:
                return None

            # One-time initialization
            self.original_size = original_size
            self.compressed_size = 0
            
            # Update component parameters once
            if self.transformer is not None:
                self.transformer.update_params(**self.config.transformer_config)
            if self.quantizer is not None:
                self.quantizer.update_params(**self.config.quantizer_config)
            if self.codec is not None:
                self.codec.update_params(**self.config.codec_config)

            compression_metadata = {
                "request_id": request_id,
                "pipeline": self.config.pipeline.copy(),
                "original_dtype": str(all_layers_data.dtype).replace("torch.", ""),
                "original_size": original_size,
                "device": str(all_layers_data.device),
                "start_layer_id": 0,
                "num_layers": num_layers,
                **metadata,
            }

            # Pre-allocate buffer for transformed/quantized data
            first_layer = all_layers_data[0]
            processed_first = first_layer
            
            # Process first layer to determine output shape
            if "transformer" in self.config.pipeline:
                processed_first = self.transformer.transform(0, processed_first, **self.config.transformer_config)
                compression_metadata["transformer_applied"] = True
            
            quantization_params_list = []
            if "quantizer" in self.config.pipeline:
                processed_first, first_qparams = self.quantizer.quantize(
                    0, processed_first, **self.config.quantizer_config
                )
                quantization_params_list.append(first_qparams)
                compression_metadata["quantization_applied"] = True
            
            # Pre-allocate output buffer
            buffer_shape = (num_layers,) + processed_first.shape
            processed_buffer = torch.empty(buffer_shape, dtype=processed_first.dtype, device=processed_first.device)
            processed_buffer[0] = processed_first

            # Tight loop: process remaining layers directly into buffer
            for layer_id in range(1, num_layers):
                current_data = all_layers_data[layer_id]
                
                if "transformer" in self.config.pipeline:
                    current_data = self.transformer.transform(
                        layer_id, current_data, **self.config.transformer_config
                    )
                
                if "quantizer" in self.config.pipeline:
                    current_data, qparams = self.quantizer.quantize(
                        layer_id, current_data, **self.config.quantizer_config
                    )
                    quantization_params_list.append(qparams)
                
                processed_buffer[layer_id] = current_data

            # Attach quantization params
            if "quantizer" in self.config.pipeline:
                compression_metadata["quantization_params"] = quantization_params_list

            # Permute to codec-friendly layout
            processed_buffer = processed_buffer.permute(0, 1, 4, 2, 3, 5)
            compression_metadata["compressed_shape"] = list(processed_buffer.shape)
            compression_metadata["compressed_dtype"] = str(processed_buffer.dtype).replace("torch.", "")

            # Codec compression
            compressed_bytes, compression_metadata = self._handle_codec_compression(
                processed_buffer, compression_metadata, num_layers - 1, request_id
            )
            
            # Release buffer
            del processed_buffer
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Use actual payload bytes as compressed_size; metadata is sent separately
            compressed_size = len(compressed_bytes)
            self.compressed_size = compressed_size

            return CompressedKVData(
                request_id=request_id,
                layer_id=num_layers - 1,
                compressed_bytes=compressed_bytes,
                metadata=compression_metadata,
                original_size=original_size,
                compressed_size=compressed_size,
            )

        except Exception as e:
            log_error(f"[CompressionManager] All-layer compression failed for {request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def decompress_all_layers(
        self,
        compressed_data: CompressedKVData,
    ) -> Optional[Any]:  # Returns torch.Tensor [num_layers, 2, ...]
        """
        Optimized fast path: decompress all layers with minimal overhead.
        Uses batch quantizer interface and tight loops.
        
        Args:
            compressed_data: CompressedKVData from compress_all_layers()
            
        Returns:
            Decompressed KV cache tensor [num_layers, 2, ...], None if decompression failed
        """
        if not self.config.enabled:
            return None
        
        try:
            import torch
            
            pipeline = compressed_data.metadata.get("pipeline", [])
            if not pipeline:
                return None
            
            num_layers = compressed_data.metadata.get("num_layers")
            if num_layers is None:
                log_error(f"[CompressionManager] Missing num_layers in metadata. Available keys: {list(compressed_data.metadata.keys())}")
                return None
            
            log_info(f"[CompressionManager] Decompressing {num_layers} layers, pipeline={pipeline}")
            
            # One-time component parameter update
            if self.transformer is not None:
                self.transformer.update_params(**self.config.transformer_config)
            if self.quantizer is not None:
                self.quantizer.update_params(**self.config.quantizer_config)
            if self.codec is not None:
                self.codec.update_params(**self.config.codec_config)
            
            current_data = compressed_data.compressed_bytes
            
            # Step 1: Codec decompression (all layers together)
            current_data = self._handle_codec_decompression(
                current_data, compressed_data, pipeline, 0
            )
            
            # Convert bytes to tensor if codec was skipped
            if isinstance(current_data, bytes):
                current_data = self._bytes_to_tensor(current_data, compressed_data.metadata)
            
            # Permute back from codec layout
            if isinstance(current_data, torch.Tensor) and current_data.dim() == 6:
                current_data = current_data.permute(0, 1, 3, 4, 2, 5)
                current_data = current_data.to(
                    getattr(torch, compressed_data.metadata.get("original_dtype", "bfloat16"), torch.bfloat16)
                )
            
            if not isinstance(current_data, torch.Tensor):
                log_error(f"[CompressionManager] Expected tensor after codec decode, got {type(current_data)}")
                return None
            
            # Step 2: Batch dequantization (optimized)
            if "quantizer" in pipeline:
                quantization_params = compressed_data.metadata.get("quantization_params")
                if quantization_params is None:
                    log_error(f"[CompressionManager] Missing quantization params")
                    return None
                
                # Use batch dequantize interface
                current_data = self.quantizer.dequantize(
                    num_layers - 1,  # layer_end_id
                    current_data,
                    quantization_params,
                    **self.config.quantizer_config
                )
            
            # Step 3: Inverse transform per layer (tight loop)
            if "transformer" in pipeline:
                # Pre-allocate output
                first_transformed = self.transformer.inverse(
                    0, current_data[0], **self.config.transformer_config
                )
                output_shape = (num_layers,) + first_transformed.shape
                restored_kv = torch.empty(output_shape, dtype=first_transformed.dtype, device=first_transformed.device)
                restored_kv[0] = first_transformed
                
                # Tight loop for remaining layers
                for layer_id in range(1, num_layers):
                    restored_kv[layer_id] = self.transformer.inverse(
                        layer_id, current_data[layer_id], **self.config.transformer_config
                    )
                current_data = restored_kv
            
            log_info(f"[CompressionManager] Decompression SUCCESS: {num_layers} layers, shape={current_data.shape}")
            return current_data
            
        except Exception as e:
            log_error(f"[CompressionManager] All-layer decompression failed for {compressed_data.request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def _handle_codec_compression(
        self, 
        tensor_data: Any, 
        compression_metadata: Dict[str, Any], 
        layer_id: int, 
        request_id: str
    ) -> tuple[bytes, Dict[str, Any]]:
        """
        Handle codec compression with boundary condition detection
        
        Returns:
            Tuple of (compressed_bytes, updated_metadata)
        """
        # Save tensor shape and dtype before converting to bytes
        if isinstance(tensor_data, torch.Tensor):
            compression_metadata["original_shape"] = list(tensor_data.shape)
            compression_metadata["original_dtype"] = str(tensor_data.dtype).replace("torch.", "")
        
        if "codec" in self.config.pipeline:
            # Save original bytes to detect if compression was skipped
            original_bytes = self._tensor_to_bytes(tensor_data)
            
            # Log tensor information for debugging
            tensor_info = {
                "request_id": request_id,
                "layer_id": layer_id,
            }
            if isinstance(tensor_data, torch.Tensor):
                tensor_info.update({
                    "shape": list(tensor_data.shape),
                    "dtype": str(tensor_data.dtype),
                    "numel": tensor_data.numel(),
                    "size_bytes": len(original_bytes),
                    "device": str(tensor_data.device),
                })
            else:
                tensor_info["type"] = type(tensor_data).__name__
            
            # Try codec compression
            try:
                compressed_bytes = self.codec.encode(
                    layer_id,
                    tensor_data,
                    **self.config.codec_config
                )
                
                # Check if compression was actually applied
                if compressed_bytes == original_bytes:
                    # Compression was skipped (boundary condition)
                    print(f"[CompressionManager] Codec skipped (boundary condition) for {request_id}, layer {layer_id}:")
                    print(f"  Tensor info: {tensor_info}")
                    print(f"  Reason: Compressed bytes identical to original (likely scalar/invalid format returned by nvCOMP)")
                    compression_metadata["codec_skipped"] = True
                    compression_metadata["codec_applied"] = False
                else:
                    # Compression was successfully applied
                    compression_ratio = len(original_bytes) / len(compressed_bytes) if len(compressed_bytes) > 0 else 0
                    compression_metadata["codec_applied"] = True
                    compression_metadata["codec_skipped"] = False
                    
            except Exception as e:
                # Codec compression failed (e.g., OOM), skip compression
                print(f"[CompressionManager] Codec compression FAILED for {request_id}, layer {layer_id}:")
                print(f"  Tensor info: {tensor_info}")
                print(f"  Error: {type(e).__name__}: {e}")
                print(f"  Fallback: Using uncompressed data ({len(original_bytes)} bytes)")
                compressed_bytes = original_bytes
                compression_metadata["codec_skipped"] = True
                compression_metadata["codec_applied"] = False
        else:
            # If no codec compression, just convert to bytes
            compressed_bytes = self._tensor_to_bytes(tensor_data)
            compression_metadata["codec_applied"] = False
            compression_metadata["codec_skipped"] = False
        
        return compressed_bytes, compression_metadata
    
    def _handle_codec_decompression(
        self,
        compressed_bytes: bytes,
        compressed_data: CompressedKVData,
        pipeline: List[str],
        layer_id: int
    ) -> Any:
        """
        Handle codec decompression with skip detection
        
        Returns:
            Decompressed tensor or bytes (if codec was skipped)
        """
        # Check if codec was skipped during compression
        codec_skipped = compressed_data.metadata.get("codec_skipped", False)
        codec_applied = compressed_data.metadata.get("codec_applied", False)
        
        if "codec" in pipeline and codec_applied and not codec_skipped:
            # Codec was actually applied, perform decompression
            original_dtype = compressed_data.metadata.get("original_dtype")
            original_shape = compressed_data.metadata.get("original_shape")
            device = compressed_data.metadata.get("device")
            assert original_dtype is not None and original_shape is not None and device is not None, \
                "Original dtype, shape, and device are required for decompression"
            return self.codec.decode(
                layer_id,
                compressed_bytes,
                original_dtype,
                original_shape,
                device,
                **self.config.codec_config
            )
        else:
            # Codec was skipped or not in pipeline, return bytes for later conversion
            return compressed_bytes
    
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

