"""
Compression components interface
Transform, Quantizer, and Lossless Compression
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class Transform(ABC):
    """Transform component interface for KV cache compression"""
    
    @abstractmethod
    def encode(self, kv_data: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply transform encoding
        
        Args:
            kv_data: KV cache tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
            config: Transform configuration
            
        Returns:
            Transformed tensor
        """
        pass
    
    @abstractmethod
    def decode(self, transformed_data: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply transform decoding
        
        Args:
            transformed_data: Transformed tensor
            config: Transform configuration (must match encode config)
            
        Returns:
            Recovered KV cache tensor
        """
        pass


class Quantizer(ABC):
    """Quantizer component interface for KV cache compression"""
    
    @abstractmethod
    def quantize(self, data: torch.Tensor, config: Dict[str, Any]) -> tuple:
        """
        Quantize tensor
        
        Args:
            data: Input tensor to quantize
            config: Quantization configuration (e.g., bits, quantization method)
            
        Returns:
            Tuple of (quantized_data, quantization_params)
            quantization_params needed for dequantization
        """
        pass
    
    @abstractmethod
    def dequantize(self, quantized_data: Any, quantization_params: Dict[str, Any], config: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantize tensor
        
        Args:
            quantized_data: Quantized data
            quantization_params: Parameters from quantize() call
            config: Quantization configuration
            
        Returns:
            Dequantized tensor
        """
        pass


class LosslessCompression(ABC):
    """Lossless compression component interface for KV cache compression"""
    
    @abstractmethod
    def compress(self, data: bytes, config: Dict[str, Any]) -> bytes:
        """
        Compress data losslessly
        
        Args:
            data: Input bytes to compress
            config: Compression configuration
            
        Returns:
            Compressed bytes
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes, config: Dict[str, Any]) -> bytes:
        """
        Decompress data
        
        Args:
            compressed_data: Compressed bytes
            config: Compression configuration (must match compress config)
            
        Returns:
            Decompressed bytes
        """
        pass

