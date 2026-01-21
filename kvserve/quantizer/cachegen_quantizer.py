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
    cachegen_quantize, cachegen_dequantize,
)

class CachegenQuantizer(Quantizer):
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
                quantization_level: 1 (aggressive), 2 (medium), or 3 (conservative)
                high_max_value: Max quantization bin for the highest precision tier (default: 32)
                mid_max_value: Max quantization bin for the middle tier (default: 16)
                low_max_value: Max quantization bin for the lowest tier (default: 12)
        """
        # Update parameters from kwargs
        self.model_name = kwargs.get("model_name", "Llama-3.1-8B-Instruct")
        self.quantization_level = kwargs.get("quantization_level", 1)
        self.high_max_value = kwargs.get("high_max_value", 32)
        self.mid_max_value = kwargs.get("mid_max_value", 16)
        self.low_max_value = kwargs.get("low_max_value", 12)
        
        # Validate quantizer parameters
        self.validate()

        # Head level: get head scores and masks from model name
        head_scores = DuoConfigGenerator.get_scores_from_csv(self.model_name)
        self.model_layers = head_scores.shape[0]
        self.layer_max_value = self._make_quantized_bins()

    def _make_quantized_bins(self):
        """
        Automatically calculate the maximum quantization value for each layer based on model layers and quantization level.
        Results are stored in self.layer_max_value as list[dict] -> [{'k': k_val, 'v': v_val}, ...]
        """

        layer_max_value = []

        # Define layer boundaries (approximated ratios used by CacheGen)
        # Key layers: roughly first 1/3, middle 1/3, final 1/3
        key_first_boundary = int(self.model_layers * 0.32)
        key_second_boundary = int(self.model_layers * 0.63)
        
        # Value layers: only the earliest ~15% (at least 2 layers) stay higher precision
        val_first_boundary = max(2, int(self.model_layers * 0.15))

        for i in range(self.model_layers):
            # --- Key (K) quantization strategy ---
            if self.quantization_level == 1:  # Level 1: aggressive compression
                # First ~2/3 use Mid, last ~1/3 use Low
                k_max = self.mid_max_value if i < key_second_boundary else self.low_max_value
            elif self.quantization_level == 2:  # Level 2: medium compression
                # First ~1/3 use High, remaining use Mid
                k_max = self.high_max_value if i < key_first_boundary else self.mid_max_value
            else:  # Level 3: conservative/default
                k_max = self.high_max_value

            # --- Value (V) quantization strategy ---
            if self.quantization_level == 1:  # Level 1: aggressive compression
                # Early layers use Mid, the rest use Low
                v_max = self.mid_max_value if i < val_first_boundary else self.low_max_value
            elif self.quantization_level == 2:  # Level 2: medium compression
                # Early layers use High, the rest use Mid
                v_max = self.high_max_value if i < val_first_boundary else self.mid_max_value
            else:  # Level 3
                v_max = self.high_max_value

            layer_max_value.append({"k": k_max, "v": v_max})

        return layer_max_value

    def validate(
        self,
    ) -> None:
        """
        Validate quantizer parameters
        
        Args:
            **kwargs: Parameters to validate.
                quantization_level: 1, 2, or 3 (controls aggressiveness)
                high_max_value: Max quantization bin for highest precision
                mid_max_value: Max quantization bin for middle precision
                low_max_value: Max quantization bin for lowest precision
        """
        assert self.quantization_level in [1, 2, 3], \
            f"quantization_level must be 1, 2 or 3, got {self.quantization_level}"
        assert self.high_max_value > self.mid_max_value and self.mid_max_value > self.low_max_value, \
            f"high_max_value must be greater than mid_max_value and mid_max_value must be greater than low_max_value, got high_max_value: {self.high_max_value}, mid_max_value: {self.mid_max_value}, low_max_value: {self.low_max_value}"

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
            layer_id: Layer index used to pick per-layer bins.
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

        quantized_keys, meta_keys = cachegen_quantize(self.layer_max_value[layer_id]["k"], keys, [2, 3])
        quantized_values, meta_values = cachegen_quantize(self.layer_max_value[layer_id]["v"], values, [2, 3])

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
            layer_id: Layer index for which dequantization is performed.
            tensor: Quantized tensor, shape can be [layers, 2, num_blocks, block_size, num_heads, head_size].
            meta_data: Quantization metadata, a list of dicts aligned by layer (batch).

        Returns:
            Dequantized tensor with original dtype and shape restored.
        """
        
        # Split the tensor into keys and values
        keys = tensor[0]
        values = tensor[1]

        # Dequantize the keys and values
        dequantized_keys = cachegen_dequantize(self.layer_max_value[layer_id]["k"], keys, meta_data["meta_keys"])
        dequantized_values = cachegen_dequantize(self.layer_max_value[layer_id]["v"], values, meta_data["meta_values"])

        # Concatenate the keys and values
        tensor = torch.stack([dequantized_keys, dequantized_values], dim=0)
        
        # Release reconstructed intermediates
        del dequantized_keys, dequantized_values
        
        return tensor
