"""
KVServe Quantizer for KV cache quantization
Implements hybrid precision quantization with head/layer splitting strategies
"""

import torch
from typing import Optional, List, Tuple, Dict, Deque, Any
from kvserve.engine.logger import log_info, log_warning, log_error, log_debug
from kvserve.config.duo_config.get_config import DuoConfigGenerator
from kvserve.manager.components import Quantizer
from kvserve.quantizer.quantizer_func import (
    kivi_quantize, kivi_dequantize,
)

class KIVIQuantizer(Quantizer):
    """
    KVServe Quantizer implementation
    Supports hybrid precision quantization with configurable split strategies
    """
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize KVServe Quantizer

        Args:
            **kwargs: Configuration parameters including:
                model_name: Model name for loading head scores (default: "Llama-3.1-8B-Instruct")
                nbits: Number of bits for quantization (2 or 4, default: 2)
                axis_key: Quantization axis for keys, "channel" or "token" (default: "channel")
                axis_value: Quantization axis for values, "channel" or "token" (default: "token")
                group_size: Group size for quantization (default: 32)
        """
        # Update parameters from kwargs
        self.model_name = kwargs.get("model_name", "Llama-3.1-8B-Instruct")
        self.nbits = kwargs.get("nbits", 2)
        self.axis_key = kwargs.get("axis_key", "channel")
        self.axis_value = kwargs.get("axis_value", "token")
        self.group_size = kwargs.get("group_size", 32)
        
        # Validate quantizer parameters
        self.validate()

    def validate(
        self,
    ) -> None:
        """
        Validate quantizer parameters
        
        Args:
            **kwargs: Parameters to validate.
                nbits: Number of bits used for quantization (2 or 4)
                axis_key: Quantization axis for keys, "channel" or "token"
                axis_value: Quantization axis for values, "channel" or "token"
                group_size: Group size for quantization
        """
        assert self.nbits in [2, 4], \
            f"nbits must be 2 or 4, got {self.nbits}"
        assert self.axis_key in ["channel", "token"], \
            f"axis_key must be 'channel' or 'token', got {self.axis_key}"
        assert self.axis_value in ["channel", "token"], \
            f"axis_value must be 'channel' or 'token', got {self.axis_value}"
        assert self.group_size > 0, \
            f"group_size must be greater than 0, got {self.group_size}"

    def update_params(
        self,
        **kwargs,
    ) -> None:
        """
        Update quantizer parameters dynamically, used in quantize() to update the quantizer parameters for every request
        
        Args:
            **kwargs: Parameters to update. If hybrid_ratio changes and split_type is "head",
                     the head_scores_mask will be recalculated.
        """
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)        

        # Validate quantizer parameters
        self.validate()
    
    def quantize(
        self, 
        layer_id: int,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Quantize KV cache tensor using hybrid precision strategy
        
        Args:
            layer_id: Layer index (not used but retained for interface consistency).
            tensor: KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
                   where tensor[0] is keys and tensor[1] is values.
            **kwargs: Additional quantization parameters (unused; kept for API parity).
            
        Returns:
            Tuple of (quantized_tensor, metadata) where:
            quantized_tensor: Quantized tensor with same shape as input
            metadata: Dictionary containing quantization parameters for dequantization
                      (per-part metadata plus original dtype)
        """
        # Don't need to update parameters here because compression manager will handle it
        # if layer_id == 0:
        #     self.update_params(**kwargs)
            
        # Split the tensor into keys and values
        keys = tensor[0]
        values = tensor[1]
        original_dtype = str(tensor.dtype).replace("torch.", "")

        quantized_keys, meta_keys = kivi_quantize(self.nbits, keys, self.axis_key, self.group_size)
        quantized_values, meta_values = kivi_quantize(self.nbits, values, self.axis_value, self.group_size)

        out_tensor = torch.stack([quantized_keys, quantized_values], dim=0)
        del keys, values, quantized_keys, quantized_values

        # Prepare the metadata for dequantization
        meta_data = {
            "meta_keys": meta_keys,
            "meta_values": meta_values,

            "original_dtype": original_dtype,
        }
        
        return out_tensor, meta_data

    def dequantize(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        meta_data: Any,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dequantize tensor using stored hybrid quantization metadata.

        Restores original tensor values and dtype for both key and value components,
        using the associated quantization metadata for each part.

        Args:
            layer_id: Layer index (not used but retained for interface consistency).
            tensor: Quantized tensor, shape can be [layers, 2, num_blocks, block_size, num_heads, head_size].
            meta_data: Quantization metadata, a list of dicts aligned by layer (batch).

        Returns:
            Dequantized tensor with original dtype and shape restored.
        """
        
        # Split the tensor into keys and values
        keys = tensor[0]
        values = tensor[1]

        # Dequantize the keys and values
        dequantized_keys = kivi_dequantize(self.axis_key, keys, meta_data["meta_keys"], self.group_size)
        dequantized_values = kivi_dequantize(self.axis_value, values, meta_data["meta_values"], self.group_size)

        # Concatenate the keys and values
        tensor = torch.stack([dequantized_keys, dequantized_values], dim=0)
        
        # Release reconstructed intermediates
        del dequantized_keys, dequantized_values
        
        return tensor
