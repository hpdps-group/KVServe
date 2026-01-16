"""
Compression Manager for KV cache compression
Coordinates transformer, quantizer, and codec compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch
import gc
import msgpack
import numpy as np

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
    compressed_tensor: torch.Tensor
    metadata: Dict[str, Any]  # Includes compression config, original size, etc.
    original_size: int
    compressed_size: int
    # metadata_size: int

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
    
    def update_config(self, config: CompressionConfig):
        """
        Update Compression Manager configuration
        """
        self.config = config
        if self.transformer is not None:
            self.transformer.update_params(**self.config.transformer_config)
        if self.quantizer is not None:
            self.quantizer.update_params(**self.config.quantizer_config)
        if self.codec is not None:
            self.codec.update_params(**self.config.codec_config)

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
        config: CompressionConfig,
        metadata: Dict[str, Any],
    ) -> Optional[CompressedKVData]:
        """
        Optimized fast path: compress all layers in a single batch with minimal overhead.
        Avoids repeated function calls by processing all layers in a tight loop.
        """
        # Update compression config for every request
        self.update_config(config)

        if not self.config.enabled or not self.config.pipeline:
            return None
        if not isinstance(all_layers_data, torch.Tensor) or all_layers_data.dim() != 6:
            log_error(f"[CompressionManager] all_layers_data must be 6D torch.Tensor, got {type(all_layers_data)}, dim={all_layers_data.dim() if isinstance(all_layers_data, torch.Tensor) else 'N/A'}")
            return None

        try:
            num_layers = all_layers_data.shape[0]
            original_size = all_layers_data.numel() * all_layers_data.element_size()
            
            if original_size < self.config.min_compress_size:
                return None

            # One-time initialization
            self.original_size = original_size
            self.compressed_size = 0
            
            # Prepare compression metadata
            compression_metadata = {
                "request_id": request_id,
                "original_dtype": str(all_layers_data.dtype).replace("torch.", ""),
                "original_size": original_size,
                "device": str(all_layers_data.device),
                "start_layer_id": 0,
                "num_layers": num_layers,
                **metadata,
            }

            quantization_params_list = []
            
            # Process first layer to determine output shape and dtype
            first_layer = all_layers_data[0]
            processed_data = first_layer
            
            if "transformer" in self.config.pipeline:
                processed_data = self.transformer.transform(0, processed_data, **self.config.transformer_config)
                compression_metadata["transformer_applied"] = True
            
            if "quantizer" in self.config.pipeline:
                processed_data, qparams = self.quantizer.quantize(
                    0, processed_data, **self.config.quantizer_config
                )
                quantization_params_list.append(qparams)
                compression_metadata["quantization_applied"] = True
            
            # 2. Determine target permuted shape
            layer_permuted_shape = (
                processed_data.shape[0], # 2
                processed_data.shape[3], # heads
                processed_data.shape[1], # blocks
                processed_data.shape[2], # block_size
                processed_data.shape[4]  # head_size
            )
            
            # 3. Allocate ONE contiguous buffer for all layers
            full_shape = (num_layers,) + layer_permuted_shape
            processed_buffer = torch.empty(
                full_shape, 
                dtype=processed_data.dtype, 
                device=processed_data.device
            )

            # 4. Write first layer directly into buffer
            processed_buffer[0].copy_(processed_data.permute(0, 3, 1, 2, 4))
            
            # Clean up first layer intermediates immediately
            del processed_data
            
            # 5. Process remaining layers and write directly
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
                
                # Direct copy to pre-allocated buffer (no stacking)
                processed_buffer[layer_id].copy_(current_data.permute(0, 3, 1, 2, 4))
                
                # Release loop variable immediately
                del current_data

            if "quantizer" in self.config.pipeline:
                compression_metadata["quantization_params"] = quantization_params_list

            compression_metadata["codec_shape"] = list(processed_buffer.shape)
            compression_metadata["codec_dtype"] = str(processed_buffer.dtype).replace("torch.", "")

            # Codec compression
            compressed_tensor, compression_metadata = self._handle_codec_compression(
                processed_buffer, compression_metadata, num_layers - 1, request_id
            )
            
            # Release large buffer immediately
            del processed_buffer
            
            # Use actual payload bytes as compressed_size; metadata is sent separately
            self.compressed_size = compressed_tensor.numel() * compressed_tensor.element_size()
            compression_metadata["compressed_size"] = self.compressed_size

            return CompressedKVData(
                request_id=request_id,
                layer_id=num_layers - 1,
                compressed_tensor=compressed_tensor,
                metadata=compression_metadata,
                original_size=original_size,
                compressed_size=self.compressed_size,
            )

        except Exception as e:
            log_error(f"[CompressionManager] All-layer compression failed for {request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def decompress_all_layers(
        self,
        compressed_data: CompressedKVData,
        config: CompressionConfig,
    ) -> Optional[Any]:  # Returns torch.Tensor [num_layers, 2, ...]
        """
        Optimized fast path: decompress all layers with minimal overhead.
        Uses pre-allocation and tight loops to minimize memory peaks.
        
        Args:
            compressed_data: CompressedKVData from compress_all_layers()
            
        Returns:
            Decompressed KV cache tensor [num_layers, 2, ...], None if decompression failed
        """
        self.update_config(config)

        if not self.config.enabled or not self.config.pipeline:
            return None
        
        try:
            import torch
            
            num_layers = compressed_data.metadata.get("num_layers")
            if num_layers is None:
                log_error(f"[CompressionManager] Missing num_layers in metadata.")
                return None
            
            # Step 1: Codec decompression (all layers together)
            # Returns tensor of shape [layers, 2, heads, blocks, block_size, head_size]
            current_data = self._handle_codec_decompression(
                compressed_data.compressed_tensor, compressed_data, self.config.pipeline, 0
            )
            
            if not isinstance(current_data, torch.Tensor):
                log_error(f"[CompressionManager] Expected tensor after codec decode, got {type(current_data)}")
                return None
            
            target_layer_shape = (
                current_data.shape[1], # 2
                current_data.shape[3], # blocks
                current_data.shape[4], # block_size
                current_data.shape[2], # heads
                current_data.shape[5], # head_size
            )
            full_target_shape = (num_layers,) + target_layer_shape
            
            # Determine dtype (restore original dtype)
            target_dtype = getattr(torch, compressed_data.metadata.get("original_dtype", "bfloat16"), torch.bfloat16)
            
            processed_buffer = torch.empty(
                full_target_shape,
                dtype=target_dtype,
                device=current_data.device
            )
            
            # Step 2 & 3: Dequantize and Transform loop (write directly to buffer)
            for layer_id in range(num_layers):
                current_layer_data = current_data[layer_id].permute(0, 2, 3, 1, 4)
                
                # Batch dequantization
                if "quantizer" in self.config.pipeline:
                    quantization_params = compressed_data.metadata.get("quantization_params")[layer_id]
                    if quantization_params is None:
                        log_error(f"[CompressionManager] Missing quantization params")
                        return None
                    
                    current_layer_data = self.quantizer.dequantize(
                        layer_id,
                        current_layer_data,
                        quantization_params,
                        **self.config.quantizer_config
                    )
            
                # Inverse transform
                if "transformer" in self.config.pipeline:
                    current_layer_data = self.transformer.inverse(
                        layer_id, 
                        current_layer_data, 
                        **self.config.transformer_config
                    )
                
                # Write to buffer (cast if necessary, though operations usually preserve/set dtype)
                if current_layer_data.dtype != target_dtype:
                    current_layer_data = current_layer_data.to(target_dtype)
                    
                processed_buffer[layer_id].copy_(current_layer_data)
                
                # Release intermediates
                del current_layer_data

            # Release codec output buffer
            del current_data
            
            log_info(f"[CompressionManager] Decompression SUCCESS: {num_layers} layers, shape={processed_buffer.shape}")
            return processed_buffer
            
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
        if "codec" in self.config.pipeline:
            try:
                compressed_tensor = self.codec.encode(
                    layer_id,
                    tensor_data,
                    **self.config.codec_config
                )
                compression_metadata["codec_applied"] = True
                    
            except Exception as e:
                # Codec compression failed (e.g., OOM), skip compression
                print(f"[CompressionManager] Codec compression FAILED for {request_id}, layer {layer_id}:")
                print(f"  Tensor shape: {tensor_data.shape}, dtype: {tensor_data.dtype}, device: {tensor_data.device}")
                print(f"  Tensor size: {tensor_data.numel() * tensor_data.element_size() / (1024**2):.2f} MB")
                print(f"  Error: {type(e).__name__}: {e}")
                import traceback
                print(f"  Traceback:")
                traceback.print_exc()
                # Fallback: Use uncompressed data
                compressed_tensor = tensor_data.reshape(-1).view(torch.uint8).contiguous()
                compression_metadata["codec_skipped"] = True
                compression_metadata["codec_error"] = f"{type(e).__name__}: {str(e)}"
        else:
            # If no codec compression, just convert to bytes
            compressed_tensor = tensor_data.reshape(-1).view(torch.uint8).contiguous()
            # compression_metadata["codec_applied"] = False
            # compression_metadata["codec_skipped"] = True
        
        return compressed_tensor, compression_metadata
    
    def _handle_codec_decompression(
        self,
        compressed_tensor: torch.Tensor,
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
        # codec_skipped = compressed_data.metadata.get("codec_skipped", False)
        codec_applied = compressed_data.metadata.get("codec_applied", False)
        codec_dtype = compressed_data.metadata.get("codec_dtype")
        codec_shape = compressed_data.metadata.get("codec_shape")        
        if "codec" in pipeline and codec_applied:
            # Codec was actually applied, perform decompression
            device = compressed_data.metadata.get("device")
            assert codec_dtype is not None and codec_shape is not None and device is not None, \
                "Original dtype, shape, and device are required for decompression"
            return self.codec.decode(
                layer_id,
                compressed_tensor,
                codec_dtype,
                codec_shape,
                device,
                **self.config.codec_config
            )
        else:
            # Codec was skipped or not in pipeline, return bytes for later conversion
            return compressed_tensor.view(getattr(torch, codec_dtype)).reshape(codec_shape)
    
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


class EasyDist:
    # Memory alignment size (bytes), ensures Tensor reads are address-aligned to avoid illegal memory access
    ALIGNMENT = 256 

    @classmethod
    def pack_object(cls, obj: Any):
        """
        Pack arbitrary mixed objects (Dict, List, Tensor nested structures) for transmission.
        """
        # 1. Pack into a single Super Tensor on GPU
        super_tensor = cls._pack_to_gpu(obj)
        
        # 2. Send total size (handshake)
        # This step is necessary - the receiver needs to know how much memory to allocate
        size_tensor = torch.tensor([super_tensor.numel()], dtype=torch.int64, device=super_tensor.device)
        
        # 3. Send Super Tensor (one-shot transmission of all data)
        return super_tensor, size_tensor

    @classmethod
    def unpack_object(cls, super_tensor: torch.Tensor) -> Any:
        """
        Unpack arbitrary mixed objects from a super tensor.
        """
        # 3. Unpack
        return cls._unpack_from_gpu(super_tensor)

    @classmethod
    def compare(cls, obj1, obj2, path=""):
        """
        Compare two objects recursively.
        """
        if type(obj1) != type(obj2):
            raise AssertionError(f"Type mismatch at {path}: {type(obj1)} vs {type(obj2)}")
        if isinstance(obj1, dict):
            assert obj1.keys() == obj2.keys(), f"Key mismatch at {path}: {obj1.keys()} vs {obj2.keys()}"
            for k in obj1:
                cls.compare(obj1[k], obj2[k], path + f".{k}")
        elif isinstance(obj1, list):
            assert len(obj1) == len(obj2), f"List length mismatch at {path}: {len(obj1)} vs {len(obj2)}"
            for i, (x, y) in enumerate(zip(obj1, obj2)):
                cls.compare(x, y, path + f"[{i}]")
        elif isinstance(obj1, torch.Tensor):
            assert obj1.shape == obj2.shape, f"Tensor shape mismatch at {path}: {obj1.shape} vs {obj2.shape}"
            assert obj1.dtype == obj2.dtype, f"Tensor dtype mismatch at {path}: {obj1.dtype} vs {obj2.dtype}"
            assert torch.allclose(obj1, obj2, atol=1e-3, rtol=1e-3), f"Tensor value mismatch at {path}"
        else:
            assert obj1 == obj2, f"Value mismatch at {path}: {obj1} vs {obj2}"
            
    # ================= Internal Implementation (Black Box) =================

    @classmethod
    def _pack_to_gpu(cls, obj: Any) -> torch.Tensor:
        """Convert object into a single uint8 tensor"""
        tensors = []
        # Recursively extract Tensors and generate skeleton (CPU operation, very fast)
        skeleton = cls._extract_tensors(obj, tensors)
        
        # 1. Serialize skeleton (MsgPack)
        # Store dtype information for restoration
        if tensors:
            skeleton['__dtype__'] = str(tensors[0].dtype).split('.')[-1]
        
        meta_bytes = msgpack.packb(skeleton, use_bin_type=True)
        meta_np = np.frombuffer(meta_bytes, dtype=np.uint8).copy()
        
        # 2. Prepare shape information (Shapes)
        # Format: [N, Rank1, D1..., Rank2, D2...]
        shape_flat = [len(tensors)]
        for t in tensors:
            shape_flat.append(t.dim())
            shape_flat.extend(t.shape)
        shape_np = np.array(shape_flat, dtype=np.int64)
        
        # 3. Calculate offsets (Offset Calculation)
        # We need to concatenate metadata, shapes, and payloads together
        # Layout: [Header(3 ints)] + [Meta Bytes] + [Padding] + [Shape Bytes] + [Padding] + [Payloads]
        
        len_meta = len(meta_np)
        len_shapes = len(shape_np) * 8 # int64 = 8 bytes
        
        # Calculate total payload size in bytes
        payload_size = sum(t.numel() * t.element_size() for t in tensors)
        
        # Align offsets
        offset_meta = 32 # Header reserves 32 bytes
        offset_shapes = cls._align(offset_meta + len_meta)
        offset_payload = cls._align(offset_shapes + len_shapes)
        total_size = offset_payload + payload_size
        
        # 4. Allocate Super Buffer (GPU)
        # This is the only memory allocation, very efficient
        device = tensors[0].device if tensors else 'cuda'
        buffer = torch.zeros(total_size, dtype=torch.uint8, device=device)
        
        # 5. Write Header (record lengths/offsets of each region)
        # Header: [Meta_Len, Shape_Len, Payload_Offset]
        header_data = torch.tensor([len_meta, len_shapes, offset_payload], dtype=torch.long, device=device)
        # Write header to buffer head (int64 -> uint8 view)
        buffer[:24] = header_data.view(torch.uint8)
        
        # 6. Write Meta and Shapes (CPU -> GPU copy)
        buffer[offset_meta : offset_meta + len_meta] = torch.from_numpy(meta_np).to(device)
        buffer[offset_shapes : offset_shapes + len_shapes] = torch.from_numpy(shape_np).view(torch.uint8).to(device)
        
        # 7. Write Payloads (D2D copy, fastest)
        if tensors:
            # Flatten & Concat all tensor data
            # Note: Assumes all tensors have the same dtype. If different, need to convert to view(uint8) then cat
            flat_payload = torch.cat([t.view(torch.uint8).view(-1) for t in tensors])
            buffer[offset_payload : offset_payload + len(flat_payload)] = flat_payload
            
        return buffer

    @classmethod
    def _unpack_from_gpu(cls, buffer: torch.Tensor) -> Any:
        """Restore object from a single Tensor"""
        # 1. Read Header
        header = buffer[:24].view(torch.int64) # 3 int64 values
        len_meta = header[0].item()
        len_shapes = header[1].item()
        offset_payload = header[2].item()
        
        # 2. Read Meta (need to transfer back to CPU for parsing)
        offset_meta = 32
        meta_bytes = buffer[offset_meta : offset_meta + len_meta].cpu().numpy().tobytes()
        skeleton = msgpack.unpackb(meta_bytes, raw=False)
        
        # 3. Restore Tensor list
        reconstructed_tensors = []
        num_tensors = 0
        
        if len_shapes > 0:
            offset_shapes = cls._align(offset_meta + len_meta)
            # view as int64
            shape_data = buffer[offset_shapes : offset_shapes + len_shapes].view(torch.int64).cpu().tolist()
            num_tensors = shape_data[0]
            
            # Get dtype
            dtype_str = skeleton.get('__dtype__', 'bfloat16')
            if '__dtype__' in skeleton: del skeleton['__dtype__'] # Clean up helper key
            target_dtype = getattr(torch, dtype_str)
            element_size = torch.tensor([], dtype=target_dtype).element_size()
            
            ptr_shape = 1
            ptr_payload = offset_payload
            
            for _ in range(num_tensors):
                rank = shape_data[ptr_shape]
                dims = shape_data[ptr_shape + 1 : ptr_shape + 1 + rank]
                ptr_shape += (1 + rank)
                
                # Calculate byte length
                numel = 1
                for d in dims: numel *= d
                byte_size = numel * element_size
                
                # Zero-Copy slice + View
                # Note: buffer is uint8, need to view as target dtype
                raw_bytes = buffer[ptr_payload : ptr_payload + byte_size]
                tensor = raw_bytes.view(target_dtype).view(dims)
                reconstructed_tensors.append(tensor)
                
                ptr_payload += byte_size

        # 4. Recursively restore
        return cls._restore_tensors(skeleton, reconstructed_tensors)

    # --- Helper Functions ---
    @staticmethod
    def _align(ptr):
        """Align pointer to ALIGNMENT boundary (256 bytes)"""
        return (ptr + 255) & ~255

    @classmethod
    def _extract_tensors(cls, obj, tensor_list):
        """Recursively extract tensors from nested structure and generate skeleton"""
        if isinstance(obj, torch.Tensor):
            tensor_list.append(obj)
            return {'__tensor__': len(tensor_list) - 1} # Placeholder
        elif isinstance(obj, list):
            return [cls._extract_tensors(x, tensor_list) for x in obj]
        elif isinstance(obj, dict):
            return {k: cls._extract_tensors(v, tensor_list) for k, v in obj.items()}
        else:
            return obj

    @classmethod
    def _restore_tensors(cls, obj, tensor_list):
        """Recursively restore nested structure by replacing placeholders with actual tensors"""
        if isinstance(obj, dict) and '__tensor__' in obj and len(obj) == 1:
            return tensor_list[obj['__tensor__']]
        elif isinstance(obj, list):
            return [cls._restore_tensors(x, tensor_list) for x in obj]
        elif isinstance(obj, dict):
            return {k: cls._restore_tensors(v, tensor_list) for k, v in obj.items()}
        else:
            return obj
