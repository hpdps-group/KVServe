"""
nvCOMP compression functions for KV cache compression
Implements GPU-accelerated lossless compression using NVIDIA nvCOMP library
"""

import torch
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
        self._algorithm = algorithm
        self._options = dict(kwargs)
        self._codecs = {}
        self.codec = None

    def _codec_for_current_stream(self, device):
        stream = torch.cuda.current_stream(device)
        key = (device.index, stream.cuda_stream)
        if key not in self._codecs:
            options = dict(self._options)
            configured_device = options.pop("device_id", None)
            configured_stream = options.pop("cuda_stream", None)
            if configured_device is not None and int(configured_device) != device.index:
                raise ValueError("nvCOMP device_id must match the tensor device")
            if configured_stream is not None and int(configured_stream) != stream.cuda_stream:
                raise ValueError("nvCOMP cuda_stream must match the current PyTorch stream")
            codec = nvcomp.Codec(algorithm=self._algorithm, device_id=device.index,
                                 cuda_stream=stream.cuda_stream, **options)
            # Keep the stream alive as long as its codec/workspace is cached.
            self._codecs[key] = (stream, codec)
        self.codec = self._codecs[key][1]
        return stream, self.codec

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
        if not tensor.is_cuda:
            raise ValueError("nvCOMP.encode requires a CUDA tensor")
        flat_view = tensor.reshape(-1).view(torch.uint8)
        if not flat_view.is_contiguous():
            flat_view = flat_view.contiguous()
        
        # Boundary check: skip compression for very small tensors
        if flat_view.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=tensor.device)
        
        # Step 2: Convert PyTorch tensor to nvCOMP array format
        with torch.cuda.device(tensor.device):
            stream, codec = self._codec_for_current_stream(tensor.device)
            nv_array = nvcomp.as_array(flat_view, cuda_stream=stream.cuda_stream)
            comp_buffer = codec.encode(nv_array)
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
        target_device = torch.device(device)
        codec_device = target_device if target_device.type == "cuda" else compressed_tensor.device
        if codec_device.index is None:
            codec_device = torch.device("cuda", torch.cuda.current_device())
        compressed_tensor = compressed_tensor.to(codec_device)
        if not compressed_tensor.is_contiguous():
            compressed_tensor = compressed_tensor.contiguous()
        
        # Step 1: Convert to nvCOMP array format            
        with torch.cuda.device(codec_device):
            stream, codec = self._codec_for_current_stream(codec_device)
            comp_buffer = nvcomp.as_array(compressed_tensor, cuda_stream=stream.cuda_stream)
            decomp_buffer = codec.decode(comp_buffer)
            decomp_tensor = torch.as_tensor(decomp_buffer, dtype=torch.uint8, device=codec_device)
        del comp_buffer
        
        # Step 4: Restore original dtype and shape
        target_dtype = getattr(torch, original_dtype)
        
        # Reshape directly on the view
        reconstructed = decomp_tensor.view(target_dtype).reshape(original_shape)
        
        # Release intermediate buffer
        del decomp_buffer, decomp_tensor
        
        return reconstructed.to(target_device)
