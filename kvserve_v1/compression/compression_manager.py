"""
Compression Manager for KV cache compression
Coordinates transformer, quantizer, and codec compression components
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import copy
import os
import torch

from kvserve_v1.compression.components import Transformer, Quantizer, Codec
from kvserve_v1.utils.logger import log_info, log_error, log_warning, log_debug

# Codec packing layout (wire byte order seen by LC/nvCOMP):
#   "permuted" – [L, 2, heads, blocks, block_size, dim]  (default; preserves LC ratio)
#   "native"   – [L, 2, blocks, block_size, heads, dim]  (faster; can hurt LC ratio)
# Fast path still quantizes into a native staging buffer, then does ONE bulk
# permute into the codec layout (instead of per-layer permute+copy_).
_CODEC_LAYOUT_ENV = "KVSERVE_CODEC_LAYOUT"


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

        self.original_size: int = 0
        self.compressed_size: int = 0

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

    @staticmethod
    def _codec_layout() -> str:
        # Default permuted: LC compression ratio matches the original pipeline.
        layout = os.environ.get(_CODEC_LAYOUT_ENV, "permuted").strip().lower()
        return layout if layout in ("native", "permuted") else "permuted"

    def _quantizer_supports_out(self) -> bool:
        return bool(getattr(self.quantizer, "supports_out_buffer", False))

    @staticmethod
    def _native_to_permuted(native: torch.Tensor) -> torch.Tensor:
        """[L,2,B,S,H,D] -> [L,2,H,B,S,D] (one bulk transpose)."""
        return native.permute(0, 1, 4, 2, 3, 5).contiguous()

    @staticmethod
    def _permuted_to_native(permuted: torch.Tensor) -> torch.Tensor:
        """[L,2,H,B,S,D] -> [L,2,B,S,H,D] (one bulk transpose)."""
        return permuted.permute(0, 1, 3, 4, 2, 5).contiguous()

    def _pack_layer_into_codec_buffer(
        self,
        processed_data: torch.Tensor,
        dest: torch.Tensor,
        layout: str,
    ) -> None:
        if layout == "permuted":
            dest.copy_(processed_data.permute(0, 3, 1, 2, 4))
        else:
            dest.copy_(processed_data)

    def _layer_from_codec_buffer(
        self,
        codec_layer: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Map one codec-packed layer to quantizer layout [2, B, S, H, D]."""
        if layout == "permuted":
            # [2, H, B, S, D] -> [2, B, S, H, D]
            return codec_layer.permute(0, 2, 3, 1, 4)
        return codec_layer

    def _quantize_layers_direct(
        self,
        layers_data: torch.Tensor,
        start_layer_id: int,
        quantization_params_list: list,
    ) -> torch.Tensor:
        """Quantize into a native uint8 buffer via out= (TileLang fast path)."""
        native_buf = torch.empty(
            layers_data.shape,
            dtype=torch.uint8,
            device=layers_data.device,
        )
        for i in range(layers_data.shape[0]):
            layer_id = start_layer_id + i
            _, qparams = self.quantizer.quantize(
                layer_id,
                layers_data[i],
                out=native_buf[i],
                **self.config.quantizer_config,
            )
            quantization_params_list.append(qparams)
        return native_buf

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

        # Keep a compact, serializable description of the pipeline in the wire
        # metadata. Besides helping the receiver select the inverse path, this
        # gives AE scripts evidence about the components that actually ran.
        metadata = dict(metadata)
        metadata["compression_pipeline"] = list(self.config.pipeline)
        if "transformer" in self.config.pipeline:
            metadata["transform_type"] = (
                (self.config.transformer_config or {}).get("transform_type")
                or "hadamard"
            )
        if "codec" in self.config.pipeline:
            codec_config = self.config.codec_config or {}
            metadata["codec_type"] = codec_config.get("codec_type")
            metadata["codec_algorithm"] = (
                codec_config.get("nvcomp_algorithm")
                or codec_config.get("lc_algorithm")
            )
        if not isinstance(all_layers_data, torch.Tensor) or all_layers_data.dim() != 6:
            log_error(f"[CompressionManager] all_layers_data must be 6D torch.Tensor, got {type(all_layers_data)}, dim={all_layers_data.dim() if isinstance(all_layers_data, torch.Tensor) else 'N/A'}")
            return None

        try:
            num_layers = all_layers_data.shape[0]
            original_size = all_layers_data.numel() * all_layers_data.element_size()
            
            if original_size < self.config.min_compress_size:
                return None
            
            # Auto-chunking: If data > 500MB, use chunked compression to reduce memory peak
            CHUNK_THRESHOLD_MB = 500
            chunk_threshold_bytes = CHUNK_THRESHOLD_MB * 1024 * 1024
            
            if original_size > chunk_threshold_bytes:
                log_info(f"[CompressionManager] Large KV cache detected ({original_size/(1024**2):.2f} MB > {CHUNK_THRESHOLD_MB} MB), using chunked compression")
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
            layout = self._codec_layout()
            compression_metadata["codec_layout"] = layout
            use_direct_quant = (
                "quantizer" in self.config.pipeline
                and "transformer" not in self.config.pipeline
                and self._quantizer_supports_out()
            )

            if use_direct_quant:
                # Quantize into native staging, then optional ONE bulk permute for
                # LC-friendly permuted wire order (keeps ratio, avoids 32x permute).
                compression_metadata["quantization_applied"] = True
                native_buf = self._quantize_layers_direct(
                    all_layers_data, 0, quantization_params_list
                )
                if layout == "native":
                    processed_buffer = native_buf
                else:
                    processed_buffer = self._native_to_permuted(native_buf)
                    del native_buf
            else:
                # Process first layer to determine output shape and dtype
                processed_data = all_layers_data[0]

                if "transformer" in self.config.pipeline:
                    processed_data = self.transformer.transform(
                        0, processed_data, **self.config.transformer_config
                    )
                    compression_metadata["transformer_applied"] = True

                if "quantizer" in self.config.pipeline:
                    processed_data, qparams = self.quantizer.quantize(
                        0, processed_data, **self.config.quantizer_config
                    )
                    quantization_params_list.append(qparams)
                    compression_metadata["quantization_applied"] = True

                if layout == "permuted":
                    layer_pack_shape = (
                        processed_data.shape[0],  # 2
                        processed_data.shape[3],  # heads
                        processed_data.shape[1],  # blocks
                        processed_data.shape[2],  # block_size
                        processed_data.shape[4],  # head_size
                    )
                else:
                    layer_pack_shape = tuple(processed_data.shape)

                processed_buffer = torch.empty(
                    (num_layers,) + layer_pack_shape,
                    dtype=processed_data.dtype,
                    device=processed_data.device,
                )
                self._pack_layer_into_codec_buffer(
                    processed_data, processed_buffer[0], layout
                )
                del processed_data

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

                    self._pack_layer_into_codec_buffer(
                        current_data, processed_buffer[layer_id], layout
                    )
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
                layout = self._codec_layout()
                chunk_meta["codec_layout"] = layout
                use_direct_quant = (
                    "quantizer" in self.config.pipeline
                    and "transformer" not in self.config.pipeline
                    and self._quantizer_supports_out()
                )

                if use_direct_quant:
                    chunk_meta["quantization_applied"] = True
                    native_buf = self._quantize_layers_direct(
                        chunk_data, start_layer, quantization_params_list
                    )
                    if layout == "native":
                        processed_buffer = native_buf
                    else:
                        processed_buffer = self._native_to_permuted(native_buf)
                        del native_buf
                else:
                    processed_data = chunk_data[0]

                    if "transformer" in self.config.pipeline:
                        processed_data = self.transformer.transform(
                            start_layer, processed_data, **self.config.transformer_config
                        )
                        chunk_meta["transformer_applied"] = True

                    if "quantizer" in self.config.pipeline:
                        processed_data, qparams = self.quantizer.quantize(
                            start_layer, processed_data, **self.config.quantizer_config
                        )
                        quantization_params_list.append(qparams)
                        chunk_meta["quantization_applied"] = True

                    if layout == "permuted":
                        layer_pack_shape = (
                            processed_data.shape[0],
                            processed_data.shape[3],
                            processed_data.shape[1],
                            processed_data.shape[2],
                            processed_data.shape[4],
                        )
                    else:
                        layer_pack_shape = tuple(processed_data.shape)

                    processed_buffer = torch.empty(
                        (chunk_num_layers,) + layer_pack_shape,
                        dtype=processed_data.dtype,
                        device=processed_data.device,
                    )
                    self._pack_layer_into_codec_buffer(
                        processed_data, processed_buffer[0], layout
                    )
                    del processed_data

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

                        self._pack_layer_into_codec_buffer(
                            current_data, processed_buffer[i], layout
                        )
                        del current_data

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
            num_layers = compressed_data.metadata.get("num_layers")
            if num_layers is None:
                log_error(f"[CompressionManager] Missing num_layers in metadata.")
                return None
            
            # Step 1: Codec decompression (all layers together)
            # native:   [L, 2, blocks, block_size, heads, dim]
            # permuted: [L, 2, heads, blocks, block_size, dim]
            current_data = self._handle_codec_decompression(
                compressed_data.compressed_tensor, compressed_data, self.config.pipeline, 0
            )
            
            if not isinstance(current_data, torch.Tensor):
                log_error(f"[CompressionManager] Expected tensor after codec decode, got {type(current_data)}")
                return None

            layout = compressed_data.metadata.get("codec_layout", "permuted")
            # Bulk inverse-permute once so the dequant loop sees native layout.
            if layout == "permuted":
                current_data = self._permuted_to_native(current_data)
                layout = "native"
            target_layer_shape = tuple(current_data.shape[1:])
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
                current_layer_data = self._layer_from_codec_buffer(
                    current_data[layer_id], layout
                )
                
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

                layout = chunk_meta.get("codec_layout", "permuted")
                if layout == "permuted":
                    current_data = self._permuted_to_native(current_data)
                    layout = "native"
                target_layer_shape = tuple(current_data.shape[1:])
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
                    current_layer_data = self._layer_from_codec_buffer(
                        current_data[i], layout
                    )
                    
                    if "quantizer" in self.config.pipeline:
                        qparams_list = chunk_meta.get("quantization_params", [])
                        if i >= len(qparams_list) or qparams_list[i] is None:
                            log_error(
                                "[CompressionManager] Missing quantization params "
                                f"for chunk layer index {i}"
                            )
                            return None
                        current_layer_data = self.quantizer.dequantize(
                            layer_id,
                            current_layer_data,
                            qparams_list[i],
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
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        return 0
    
