"""
Compression Manager for KV cache compression
Coordinates transformer, quantizer, and codec compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import copy
import os
import torch
import gc
import msgpack
import numpy as np

from kvserve.manager.components import Transformer, Quantizer, Codec
from kvserve.quantizer import KVServeQuantizer
from kvserve.engine.logger import log_info, log_error, log_warning, log_debug


DEFAULT_COMPRESSION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "quantizer_config": {
        "model_name": "Llama-3.1-8B-Instruct",
        "hybrid_ratio": 0.8,
        "high_key_max_value": 12,
        "high_value_max_value": 8,
        "low_key_max_value": 6,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    "codec_config": {
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
    },
    "min_compress_size": 1024,
}


def get_default_compression_config() -> Dict[str, Any]:
    """Return a copy of the default compression config."""
    return copy.deepcopy(DEFAULT_COMPRESSION_CONFIG)


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
    compressed_tensor: Optional[torch.Tensor] = None  # Single tensor (for non-chunked)
    metadata: Dict[str, Any] = None  # Includes compression config, original size, etc.
    original_size: int = 0
    compressed_size: int = 0
    
    # Chunked compression support (for large KV caches)
    is_chunked: bool = False
    chunks: Optional[List[torch.Tensor]] = None  # List of compressed chunk tensors
    chunk_metadata: Optional[List[Dict[str, Any]]] = None  # Metadata for each chunk
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
        
        For large data (>500MB), automatically switches to chunked compression to reduce memory peak.
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
            
            # Auto-chunking: If data > threshold, use chunked compression to reduce memory peak
            # Allow override via env var for simulation tuning
            try:
                chunk_threshold_mb = int(os.getenv("KVSERVE_CHUNK_THRESHOLD_MB", "500"))
            except ValueError:
                chunk_threshold_mb = 500
            chunk_threshold_bytes = chunk_threshold_mb * 1024 * 1024
            
            if original_size > chunk_threshold_bytes:
                log_info(f"[CompressionManager] Large KV cache detected ({original_size/(1024**2):.2f} MB > {chunk_threshold_mb} MB), using chunked compression")
                return self._compress_all_layers_chunked(all_layers_data, request_id, metadata, chunk_size=8)
            
            # Otherwise, use fast single-batch compression (original logic below)

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
            # TODO: This tensor shape should be handled by the codec, not the manager
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
    
    def _compress_all_layers_chunked(
        self,
        all_layers_data: torch.Tensor,
        request_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = 8,
    ) -> Optional[CompressedKVData]:
        """
        Chunked compression for large KV caches to reduce memory peak.
        Compresses layers in chunks (e.g., 8 layers at a time) and stores as separate tensors.
        """
        num_layers = all_layers_data.shape[0]
        original_size = all_layers_data.numel() * all_layers_data.element_size()
        
        chunks = []
        chunk_metadata_list = []
        total_compressed_size = 0
        
        num_chunks = (num_layers + chunk_size - 1) // chunk_size
        log_info(f"[CompressionManager] Compressing {num_layers} layers in {num_chunks} chunks of {chunk_size} layers each")
        
        try:
            for chunk_idx in range(num_chunks):
                start_layer = chunk_idx * chunk_size
                end_layer = min(start_layer + chunk_size, num_layers)
                
                log_debug(f"[CompressionManager] Processing chunk {chunk_idx+1}/{num_chunks} (layers {start_layer}-{end_layer-1})")
                
                # Extract chunk
                chunk_data = all_layers_data[start_layer:end_layer]
                chunk_num_layers = end_layer - start_layer
                
                # Prepare chunk metadata
                chunk_meta = {
                    "request_id": request_id,
                    "original_dtype": str(chunk_data.dtype).replace("torch.", ""),
                    "device": str(chunk_data.device),
                    "start_layer_id": start_layer,
                    "num_layers": chunk_num_layers,
                    "chunk_index": chunk_idx,
                    "total_chunks": num_chunks,
                    **metadata,
                }
                
                quantization_params_list = []
                
                # Process first layer to determine output shape
                first_layer = chunk_data[0]
                processed_data = first_layer
                
                if "transformer" in self.config.pipeline:
                    processed_data = self.transformer.transform(start_layer, processed_data, **self.config.transformer_config)
                    chunk_meta["transformer_applied"] = True
                
                if "quantizer" in self.config.pipeline:
                    processed_data, qparams = self.quantizer.quantize(
                        start_layer, processed_data, **self.config.quantizer_config
                    )
                    quantization_params_list.append(qparams)
                    chunk_meta["quantization_applied"] = True
                
                # Determine permuted shape
                layer_permuted_shape = (
                    processed_data.shape[0], # 2
                    processed_data.shape[3], # heads
                    processed_data.shape[1], # blocks
                    processed_data.shape[2], # block_size
                    processed_data.shape[4]  # head_size
                )
                
                # Allocate buffer for this chunk only
                chunk_buffer_shape = (chunk_num_layers,) + layer_permuted_shape
                processed_buffer = torch.empty(
                    chunk_buffer_shape,
                    dtype=processed_data.dtype,
                    device=processed_data.device
                )
                
                # Write first layer
                processed_buffer[0].copy_(processed_data.permute(0, 3, 1, 2, 4))
                del processed_data
                
                # Process remaining layers in this chunk
                for i in range(1, chunk_num_layers):
                    layer_id = start_layer + i
                    current_data = chunk_data[i]
                    
                    if "transformer" in self.config.pipeline:
                        current_data = self.transformer.transform(
                            layer_id, current_data, **self.config.transformer_config
                        )
                    
                    if "quantizer" in self.config.pipeline:
                        current_data, qparams = self.quantizer.quantize(
                            layer_id, current_data, **self.config.quantizer_config
                        )
                        quantization_params_list.append(qparams)
                    
                    processed_buffer[i].copy_(current_data.permute(0, 3, 1, 2, 4))
                    del current_data
                
                # Release chunk_data after processing
                del chunk_data
                
                if "quantizer" in self.config.pipeline:
                    chunk_meta["quantization_params"] = quantization_params_list
                
                chunk_meta["codec_shape"] = list(processed_buffer.shape)
                chunk_meta["codec_dtype"] = str(processed_buffer.dtype).replace("torch.", "")
                
                # Codec compression for this chunk
                compressed_tensor, chunk_meta = self._handle_codec_compression(
                    processed_buffer, chunk_meta, end_layer - 1, request_id
                )
                
                # Release buffer immediately
                del processed_buffer
                
                # Store chunk
                chunk_size_bytes = compressed_tensor.numel() * compressed_tensor.element_size()
                total_compressed_size += chunk_size_bytes
                chunk_meta["compressed_size"] = chunk_size_bytes
                
                chunks.append(compressed_tensor)
                chunk_metadata_list.append(chunk_meta)
                
                log_debug(f"[CompressionManager] Chunk {chunk_idx+1} compressed: {chunk_size_bytes/(1024**2):.2f} MB")
            
            # Return chunked result
            log_info(f"[CompressionManager] Chunked compression complete: {original_size/(1024**2):.2f} MB -> {total_compressed_size/(1024**2):.2f} MB ({num_chunks} chunks)")
            
            return CompressedKVData(
                request_id=request_id,
                layer_id=num_layers - 1,
                compressed_tensor=None,  # No single tensor
                metadata={
                    "request_id": request_id,
                    "original_size": original_size,
                    "num_layers": num_layers,
                    "is_chunked": True,
                    "num_chunks": num_chunks,
                    **metadata,
                },
                original_size=original_size,
                compressed_size=total_compressed_size,
                is_chunked=True,
                chunks=chunks,
                chunk_metadata=chunk_metadata_list,
            )
        
        except Exception as e:
            log_error(f"[CompressionManager] Chunked compression failed for {request_id}: {e}")
            import traceback
            log_error(f"[CompressionManager] Traceback: {traceback.format_exc()}")
            return None
    
    def decompress_all_layers(
        self,
        compressed_data: CompressedKVData,
        config: CompressionConfig,
    ) -> Optional[Any]:  # Returns torch.Tensor [num_layers, 2, ...] or List[torch.Tensor] for chunked
        """
        Optimized fast path: decompress all layers with minimal overhead.
        Uses pre-allocation and tight loops to minimize memory peaks.
        
        For chunked data, returns a List[torch.Tensor] to avoid peak memory from concatenation.
        
        Args:
            compressed_data: CompressedKVData from compress_all_layers()
            
        Returns:
            Decompressed KV cache tensor [num_layers, 2, ...], or List of chunk tensors if is_chunked
        """
        self.update_config(config)

        if not self.config.enabled or not self.config.pipeline:
            return None
        
        # Handle chunked data
        if compressed_data.is_chunked:
            return self._decompress_all_layers_chunked(compressed_data)
        
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
    
    def _decompress_all_layers_chunked(
        self,
        compressed_data: CompressedKVData,
    ) -> Optional[List[torch.Tensor]]:
        """
        Decompress chunked KV cache data.
        Returns a list of decompressed chunk tensors to avoid memory peak from concatenation.
        Each chunk is [chunk_layers, 2, blocks, block_size, heads, head_size].
        """
        if not compressed_data.chunks or not compressed_data.chunk_metadata:
            log_error(f"[CompressionManager] Chunked data missing chunks or metadata")
            return None
        
        num_chunks = len(compressed_data.chunks)
        log_info(f"[CompressionManager] Decompressing {num_chunks} chunks")
        
        decompressed_chunks = []
        
        try:
            for chunk_idx in range(num_chunks):
                chunk_tensor = compressed_data.chunks[chunk_idx]
                chunk_meta = compressed_data.chunk_metadata[chunk_idx]
                
                start_layer = chunk_meta.get("start_layer_id", 0)
                chunk_num_layers = chunk_meta.get("num_layers", 8)
                
                log_debug(f"[CompressionManager] Decompressing chunk {chunk_idx+1}/{num_chunks} (layers {start_layer}-{start_layer+chunk_num_layers-1})")
                
                # Create a temporary CompressedKVData for this chunk
                chunk_compressed = CompressedKVData(
                    request_id=compressed_data.request_id,
                    layer_id=start_layer + chunk_num_layers - 1,
                    compressed_tensor=chunk_tensor,
                    metadata=chunk_meta,
                    original_size=chunk_meta.get("original_size", 0),
                    compressed_size=chunk_meta.get("compressed_size", 0),
                )
                
                # Decompress this chunk using codec
                current_data = self._handle_codec_decompression(
                    chunk_tensor, chunk_compressed, self.config.pipeline, start_layer
                )
                
                if not isinstance(current_data, torch.Tensor):
                    log_error(f"[CompressionManager] Chunk {chunk_idx} decode failed")
                    return None
                
                # Determine target shape for this chunk
                target_layer_shape = (
                    current_data.shape[1], # 2
                    current_data.shape[3], # blocks
                    current_data.shape[4], # block_size
                    current_data.shape[2], # heads
                    current_data.shape[5], # head_size
                )
                chunk_target_shape = (chunk_num_layers,) + target_layer_shape
                
                # Determine dtype
                target_dtype = getattr(torch, chunk_meta.get("original_dtype", "bfloat16"), torch.bfloat16)
                
                # Allocate buffer for this chunk
                chunk_buffer = torch.empty(
                    chunk_target_shape,
                    dtype=target_dtype,
                    device=current_data.device
                )
                
                # Dequantize and transform each layer in this chunk
                for i in range(chunk_num_layers):
                    layer_id = start_layer + i
                    current_layer_data = current_data[i].permute(0, 2, 3, 1, 4)
                    
                    if "quantizer" in self.config.pipeline:
                        quantization_params = chunk_meta.get("quantization_params", [])[i]
                        if quantization_params:
                            current_layer_data = self.quantizer.dequantize(
                                layer_id,
                                current_layer_data,
                                quantization_params,
                                **self.config.quantizer_config
                            )
                    
                    if "transformer" in self.config.pipeline:
                        current_layer_data = self.transformer.inverse(
                            layer_id,
                            current_layer_data,
                            **self.config.transformer_config
                        )
                    
                    chunk_buffer[i].copy_(current_layer_data)
                    del current_layer_data
                
                # Release codec-decoded data
                del current_data
                
                decompressed_chunks.append(chunk_buffer)
                log_debug(f"[CompressionManager] Chunk {chunk_idx+1} decompressed: {chunk_buffer.numel() * chunk_buffer.element_size()/(1024**2):.2f} MB")
            
            log_info(f"[CompressionManager] Chunked decompression complete: {num_chunks} chunks")
            return decompressed_chunks
        
        except Exception as e:
            log_error(f"[CompressionManager] Chunked decompression failed: {e}")
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
    # Memory alignment size (bytes); keeps tensor reads address-aligned to avoid illegal access
    ALIGNMENT = 256 

    @classmethod
    def pack_object(cls, obj: Any):
        """
        Pack arbitrary mixed objects (Dict, List, Tensor nested structures) for transmission.
        """
        # 1. Pack into a single super tensor on GPU
        super_tensor = cls._pack_to_gpu(obj)
        
        # 2. Send total size (handshake)
        # The receiver must know how much memory to allocate
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
        """Convert object into a single uint8 tensor (Fixed for Mixed Types & Alignment)"""
        tensors = []
        skeleton = cls._extract_tensors(obj, tensors)
        
        # Record dtype for every tensor, not just the first
        if tensors:
            skeleton['__dtypes__'] = [str(t.dtype).split('.')[-1] for t in tensors]
        
        meta_bytes = msgpack.packb(skeleton, use_bin_type=True)
        meta_np = np.frombuffer(meta_bytes, dtype=np.uint8).copy()
        
        shape_flat = [len(tensors)]
        for t in tensors:
            shape_flat.append(t.dim())
            shape_flat.extend(t.shape)
        shape_np = np.array(shape_flat, dtype=np.int64)
        
        # Align payload to 8 bytes so each tensor starts at an 8-byte boundary,
        # preventing invalid alignment errors when using view()
        payload_parts = []
        for t in tensors:
                # Convert to byte view
            data_bytes = t.view(torch.uint8).reshape(-1)
            payload_parts.append(data_bytes)
            
            # Calculate padding bytes needed
            rem = data_bytes.numel() % 8
            if rem > 0:
                # Pad with zeros
                padding = torch.zeros(8 - rem, dtype=torch.uint8, device=t.device)
                payload_parts.append(padding)
        
        if payload_parts:
            flat_payload = torch.cat(payload_parts)
            payload_size = flat_payload.numel()
        else:
            device = tensors[0].device if tensors else ('cuda' if torch.cuda.is_available() else 'cpu')
            flat_payload = torch.tensor([], dtype=torch.uint8, device=device)
            payload_size = 0
        
        len_meta = len(meta_np)
        len_shapes = len(shape_np) * 8 # int64 = 8 bytes
        
        # Align offsets
        offset_meta = 32 
        offset_shapes = cls._align(offset_meta + len_meta)
        offset_payload = cls._align(offset_shapes + len_shapes)
        total_size = offset_payload + payload_size
        
        # Allocate Super Buffer
        device = tensors[0].device if tensors else 'cuda'
        buffer = torch.zeros(total_size, dtype=torch.uint8, device=device)
        
        # Write Header
        header_data = torch.tensor([len_meta, len_shapes, offset_payload], dtype=torch.long, device=device)
        buffer[:24] = header_data.view(torch.uint8)
        
        # Write Meta and Shapes
        buffer[offset_meta : offset_meta + len_meta] = torch.from_numpy(meta_np).to(device)
        buffer[offset_shapes : offset_shapes + len_shapes] = torch.from_numpy(shape_np).view(torch.uint8).to(device)
        
        # Write Payloads
        if payload_size > 0:
            buffer[offset_payload : offset_payload + payload_size] = flat_payload
            
        return buffer

    @classmethod
    def _unpack_from_gpu(cls, buffer: torch.Tensor) -> Any:
        """Restore object from a single Tensor (Fixed for Mixed Types & Alignment)"""
        # 1. Read Header
        header = buffer[:24].view(torch.int64)
        len_meta = header[0].item()
        len_shapes = header[1].item()
        offset_payload = header[2].item()
        
        # 2. Read Meta
        offset_meta = 32
        meta_bytes = buffer[offset_meta : offset_meta + len_meta].cpu().numpy().tobytes()
        skeleton = msgpack.unpackb(meta_bytes, raw=False)
        
        # 3. Restore Tensor list
        reconstructed_tensors = []
        
        if len_shapes > 0:
            offset_shapes = cls._align(offset_meta + len_meta)
            shape_data = buffer[offset_shapes : offset_shapes + len_shapes].view(torch.int64).cpu().tolist()
            num_tensors = shape_data[0]
            
            # Retrieve stored dtype list
            dtypes = skeleton.get('__dtypes__', [])
            
            ptr_shape = 1
            ptr_payload = offset_payload
            
            for i in range(num_tensors):
                rank = shape_data[ptr_shape]
                dims = shape_data[ptr_shape + 1 : ptr_shape + 1 + rank]
                ptr_shape += (1 + rank)
                
                # Retrieve correct dtype for current tensor
                dtype_str = dtypes[i]
                target_dtype = getattr(torch, dtype_str)
                # Size in bytes for this dtype (e.g., int32=4, int16=2)
                element_size = torch.tensor([], dtype=target_dtype).element_size()
                
                # Compute actual byte length for the tensor
                numel = 1
                for d in dims: numel *= d
                byte_size = numel * element_size
                
                # Zero-copy slice + view; payload pointer is 8-byte aligned thanks to padding
                raw_bytes = buffer[ptr_payload : ptr_payload + byte_size]
                tensor = raw_bytes.view(target_dtype).view(dims)
                reconstructed_tensors.append(tensor)
                
                # Skip padding and move pointer to next 8-byte boundary
                rem = byte_size % 8
                padding = (8 - rem) if rem > 0 else 0
                ptr_payload += (byte_size + padding)

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
