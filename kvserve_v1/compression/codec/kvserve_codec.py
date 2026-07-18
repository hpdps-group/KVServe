"""
KVServe Codec for KV cache compression
Implements codec compression using libraries like nvCOMP
"""

import torch
from kvserve_v1.compression.components import Codec
from kvserve_v1.compression.codec.nvcomp_func import nvCOMPCodec

_KVSERVE_CODEC_KEYS = frozenset({
    "codec_type",
    "nvcomp_algorithm",
    "lc_algorithm",
    "lc_meta_path",
})


def _nvcomp_codec_kwargs(kwargs: dict) -> dict:
    """Strip KVServe routing keys before passing the rest to ``nvcomp.Codec``."""
    return {k: v for k, v in kwargs.items() if k not in _KVSERVE_CODEC_KEYS}


class KVServeCodec(Codec):
    """
    KVServe Codec implementation
    Supports nvCOMP compression library for lossless KV cache compression and other codecs
    """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize KVServe Codec
        
        Args:
            **kwargs: Configuration parameters including:
                codec_type: Type of codec, "nvcomp" or "lc" (default: "nvcomp")
                nvcomp_algorithm: Compression algorithm, one of:
                    "ANS", "Bitcomp", "Cascaded", "Deflate", "GDeflate", "LZ4", "Zstd"
                    (default: "ANS")
                lc_algorithm / lc_meta_path: LC runtime settings when codec_type="lc"
        """
        # Update parameters from kwargs
        self.codec_type = kwargs.get("codec_type", "nvcomp")
        self.nvcomp_algorithm = kwargs.get("nvcomp_algorithm", "ANS")
        self.lc_algorithm = kwargs.get("lc_algorithm")
        self.lc_meta_path = kwargs.get("lc_meta_path")
        self._codec_kwargs = dict(kwargs)

        self.codec = None

        # Validate codec parameters
        self.validate()

        # Initialize the codec based on codec_type
        match self.codec_type:
            case "nvcomp":
                self.codec = nvCOMPCodec(
                    algorithm=self.nvcomp_algorithm,
                    **_nvcomp_codec_kwargs(kwargs),
                )
            case "lc":
                from kvserve_v1.compression.codec.lc_codec import LCCodec
                self.codec = LCCodec(**kwargs)
            case _:
                raise ValueError(
                    f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp', 'lc']"
                )

    def validate(
        self,
    ) -> None:
        """
        Validate codec parameters
        
        Raises:
            AssertionError: If codec_type or nvcomp_algorithm is invalid
        """
        assert self.codec_type in ["nvcomp", "lc"], \
            f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp', 'lc']"
        if self.codec_type == "nvcomp":
            assert self.nvcomp_algorithm in ["ANS", "Bitcomp", "Cascaded", "Deflate", "GDeflate", "LZ4", "Zstd"], \
                f"Invalid nvcomp's algorithm: {self.nvcomp_algorithm}"

    def update_params(
        self,
        **kwargs
    ) -> None:
        """
        Update codec parameters dynamically.

        For LC, reuse the existing LCCodec instance when type/algorithm/meta are
        unchanged so the encode staging buffer pool survives across requests.
        """
        prev_type = self.codec_type
        prev_nv = self.nvcomp_algorithm
        prev_lc_algo = self.lc_algorithm
        prev_lc_meta = self.lc_meta_path

        self._codec_kwargs.update(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if "lc_algorithm" in kwargs:
            self.lc_algorithm = kwargs["lc_algorithm"]
        if "lc_meta_path" in kwargs:
            self.lc_meta_path = kwargs["lc_meta_path"]

        self.validate()

        match self.codec_type:
            case "nvcomp":
                if (
                    self.codec is not None
                    and prev_type == "nvcomp"
                    and self.nvcomp_algorithm == prev_nv
                ):
                    return
                self.codec = nvCOMPCodec(
                    algorithm=self.nvcomp_algorithm,
                    **_nvcomp_codec_kwargs(self._codec_kwargs),
                )
            case "lc":
                if (
                    self.codec is not None
                    and prev_type == "lc"
                    and self.lc_algorithm == prev_lc_algo
                    and self.lc_meta_path == prev_lc_meta
                ):
                    # Propagate any LC-specific knobs without rebuilding.
                    self.codec.update_params(**kwargs)
                    return
                from kvserve_v1.compression.codec.lc_codec import LCCodec
                self.codec = LCCodec(**self._codec_kwargs)
            case _:
                raise ValueError(
                    f"Invalid codec type: {self.codec_type}, expected one of: ['nvcomp', 'lc']"
                )

    def encode(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode (compress) KV cache tensor to bytes
        
        Args:
            layer_id: Layer ID for encoding
            tensor: KV cache tensor to compress, shape: [2, num_blocks, block_size, num_heads, head_size]
            **kwargs: Additional encoding parameters
            
        Returns:
            Compressed tensor 
        """
        # Don't need to update parameters here because compression manager will handle it
        # if layer_id == 0:
        #     self.update_params(**kwargs)
    
        # Compress tensor to bytes
        comp_tensor = self.codec.encode(tensor, **kwargs)
        
        return comp_tensor

    def decode(
        self,
        layer_id: int,
        compressed_data: bytes,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode (decompress) compressed bytes back to KV cache tensor
        
        Args:
            layer_id: Layer ID for decoding
            compressed_data: Compressed bytes from encode()
            original_dtype: Original tensor dtype as string (e.g., "bfloat16", "float32")
            original_shape: Original tensor shape as list (e.g., [32, 2, 128, 32, 8, 128])
            device: Target device for decompressed tensor (e.g., "cuda:0", "cpu")
            **kwargs: Additional decoding parameters
            
        Returns:
            Decompressed tensor with original dtype and shape restored
        """
        # Decompress bytes to tensor
        decomp_tensor = self.codec.decode(compressed_data, original_dtype, original_shape, device, **kwargs)
        
        return decomp_tensor
        
        