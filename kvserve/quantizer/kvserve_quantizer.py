"""
KVServe Quantizer for KV cache quantization
Implements hybrid precision quantization with head/layer splitting strategies
"""

import torch
from typing import Optional, List, Tuple, Dict, Deque, Any
from kvserve.engine.logger import log_info, log_warning, log_error, log_debug
from kvserve.config.duo_config.get_config import DuoConfigGenerator
from kvserve.manager.components import Quantizer
from kvserve.quantizer import (
    quantize, dequantize, 
    layer_split, layer_restore, layer_reconstruct,
    head_split, head_restore, head_reconstruct, 
)

class KVServeQuantizer(Quantizer):
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
                hybrid_ratio: Ratio of heads/layers to use low precision (default: 0.3)
                high_key_max_value: Max quantization value for high precision keys (default: 16)
                high_value_max_value: Max quantization value for high precision values (default: 16)
                low_key_max_value: Max quantization value for low precision keys (default: 12)
                low_value_max_value: Max quantization value for low precision values (default: 12)
                axis_key: Quantization axis for keys, "channel" "token" or "tensor" (default: "channel")
                axis_value: Quantization axis for values, "channel" "token" or "tensor" (default: "token")
                split_type: Split strategy, "head" or "layer" (default: "head")
        """
        # Update parameters from kwargs
        self.model_name = kwargs.get("model_name", "Llama-3.1-8B-Instruct")
        self.hybrid_ratio = kwargs.get("hybrid_ratio", 0.3)
        self.high_key_max_value = kwargs.get("high_key_max_value", 16)
        self.high_value_max_value = kwargs.get("high_value_max_value", 16)
        self.low_key_max_value = kwargs.get("low_key_max_value", 12)
        self.low_value_max_value = kwargs.get("low_value_max_value", 12)
        self.axis_key = kwargs.get("axis_key", "channel")
        self.axis_value = kwargs.get("axis_value", "token")
        self.split_type = kwargs.get("split_type", "head")

        # Head level: get head scores and masks from model name
        self.head_scores = DuoConfigGenerator.get_scores_from_csv(self.model_name)
        low_precision_num_heads = round(self.head_scores.numel() * self.hybrid_ratio)
        self.head_scores_mask = torch.zeros_like(self.head_scores, dtype=torch.bool)
        flat_indices = torch.argsort(self.head_scores.flatten())[:low_precision_num_heads]
        multi_indices = torch.unravel_index(flat_indices, self.head_scores.shape)
        self.head_scores_mask[multi_indices] = True

        # Layer level: currently only select the last 33% layers to be low precision
        self.layer_scores_mask = torch.zeros(self.head_scores.shape[0], dtype=torch.bool)
        low_precision_num_layers = round(self.layer_scores_mask.shape[0] * 0.33)
        self.layer_scores_mask[-low_precision_num_layers:] = True

        # TODO: add cache for quantized keys and values
        # self._high_quantized_key_cache = Deque[torch.Tensor]()
        # self._high_quantized_value_cache = Deque[torch.Tensor]()
        # self._low_quantized_key_cache = Deque[torch.Tensor]()
        # self._low_quantized_value_cache = Deque[torch.Tensor]()

        # Validate quantizer parameters
        self.validate()

    def validate(
        self,
    ) -> None:
        """
        Validate quantizer parameters
        
        Args:
            **kwargs: Parameters to validate.
                hybrid_ratio: Ratio of heads/layers to use low precision
                high_key_max_value: Max quantization value for high precision keys
                high_value_max_value: Max quantization value for high precision values
                low_key_max_value: Max quantization value for low precision keys
                low_value_max_value: Max quantization value for low precision values
                axis_key: Quantization axis for keys, "channel" "token" or "tensor"
                axis_value: Quantization axis for values, "channel" "token" or "tensor"
                split_type: Split strategy, "head" or "layer"
        """
        assert self.hybrid_ratio >= 0 and self.hybrid_ratio <= 1, \
            f"hybrid_ratio must be between 0 and 1, got {self.hybrid_ratio}"
        assert self.low_key_max_value > 0 and self.low_key_max_value < 255, \
            f"low_key_max_value must be greater than 0 and less than 255, got {self.low_key_max_value}"
        assert self.low_value_max_value > 0 and self.low_value_max_value < 255, \
            f"low_value_max_value must be greater than 0 and less than 255, got {self.low_value_max_value}"
        assert self.high_key_max_value > 0 and self.high_key_max_value < 255, \
            f"high_key_max_value must be greater than 0 and less than 255, got {self.high_key_max_value}"
        assert self.high_value_max_value > 0 and self.high_value_max_value < 255, \
            f"high_value_max_value must be greater than 0 and less than 255, got {self.high_value_max_value}"
        assert self.low_key_max_value <= self.high_key_max_value, \
            f"low_key_max_value must be less than high_key_max_value, got low_key_max_value: {self.low_key_max_value} and high_key_max_value: {self.high_key_max_value}"
        assert self.low_value_max_value <= self.high_value_max_value, \
            f"low_value_max_value must be less than high_value_max_value, got low_value_max_value: {self.low_value_max_value} and high_value_max_value: {self.high_value_max_value}"
        assert self.axis_key in ["channel", "token", "tensor"], \
            f"axis_key must be 'channel', 'token' or 'tensor', got {self.axis_key}"
        assert self.axis_value in ["channel", "token", "tensor"], \
            f"axis_value must be 'channel', 'token' or 'tensor', got {self.axis_value}"
        assert self.split_type in ["head", "layer"], \
            f"split_type must be 'head' or 'layer', got {self.split_type}"

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
        # If hybrid_ratio changes, recalculate the head_scores_mask
        if kwargs.get("hybrid_ratio", None) != self.hybrid_ratio:
            if kwargs.get("split_type", None) == "head":
                pruned_num_heads = round(self.head_scores.numel() * self.hybrid_ratio)
                self.head_scores_mask = torch.zeros_like(self.head_scores, dtype=torch.bool)
                flat_indices = torch.argsort(self.head_scores.flatten())[:pruned_num_heads]
                multi_indices = torch.unravel_index(flat_indices, self.head_scores.shape)
                self.head_scores_mask[multi_indices] = True
        
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
            layer_id: Layer ID for determining split strategy
            tensor: KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
                   where first dimension is [keys, values]
            **kwargs: Additional quantization parameters
            
        Returns:
            Tuple of (quantized_tensor, metadata) where:
                quantized_tensor: Quantized tensor with same shape as input
                metadata: Dictionary containing quantization parameters for dequantization
        """
        # Update quantizer parameters for every request
        # Only update parameters for the first layer
        if layer_id == 0:
            self.update_params(**kwargs)

        # Split the tensor into keys and values
        keys = tensor[0]
        values = tensor[1]

        # Split the keys and values into low and high precision parts
        if self.split_type == "head":
            low_keys, high_keys = head_split(layer_id, keys, self.head_scores_mask)
            low_values, high_values = head_split(layer_id, values, self.head_scores_mask)
        elif self.split_type == "layer":
            low_keys, high_keys = layer_split(layer_id, keys, self.layer_scores_mask)
            low_values, high_values = layer_split(layer_id, values, self.layer_scores_mask)

        # Quantize the high and low precision parts
        high_keys, high_key_meta = quantize(self.high_key_max_value, high_keys, self.axis_key)
        high_values, high_value_meta = quantize(self.high_value_max_value, high_values, self.axis_value)
        low_keys, low_key_meta = quantize(self.low_key_max_value, low_keys, self.axis_key)
        low_values, low_value_meta = quantize(self.low_value_max_value, low_values, self.axis_value)

        # Concatenate the low and high precision parts
        keys = torch.cat([low_keys, high_keys], dim=2)
        values = torch.cat([low_values, high_values], dim=2)
        tensor = torch.stack([keys, values], dim=0)
        
        # Prepare the metadata for dequantization
        meta_data = {
            "high_key_meta": high_key_meta,
            "high_value_meta": high_value_meta,
            "low_key_meta": low_key_meta,
            "low_value_meta": low_value_meta,
        }
        return tensor, meta_data

    def dequantize(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        meta_data: Dict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dequantize KV cache tensor using stored metadata
        
        Args:
            layer_id: Layer ID for determining split strategy
            tensor: Quantized KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
            meta_data: Dictionary containing quantization parameters from quantize()
            **kwargs: Additional dequantization parameters
            
        Returns:
            Dequantized tensor with original precision restored
        """
        # Split the tensor into keys and values
        keys = tensor[0]
        values = tensor[1]

        # Restore the low and high precision parts
        if self.split_type == "head":
            low_keys, high_keys = head_restore(layer_id, keys, self.head_scores_mask)
            low_values, high_values = head_restore(layer_id, values, self.head_scores_mask)
        elif self.split_type == "layer":
            low_keys, high_keys = layer_restore(layer_id, keys, self.layer_scores_mask)
            low_values, high_values = layer_restore(layer_id, values, self.layer_scores_mask)

        # Dequantize the high and low precision parts
        high_keys = dequantize(high_keys, meta_data["high_key_meta"])
        high_values = dequantize(high_values, meta_data["high_value_meta"])
        low_keys = dequantize(low_keys, meta_data["low_key_meta"])
        low_values = dequantize(low_values, meta_data["low_value_meta"])

        # Reconstruct the keys and values
        if self.split_type == "head":
            keys = head_reconstruct(layer_id, low_keys, high_keys, self.head_scores_mask)
            values = head_reconstruct(layer_id, low_values, high_values, self.head_scores_mask)
        elif self.split_type == "layer":
            keys = layer_reconstruct(layer_id, low_keys, high_keys, self.layer_scores_mask)
            values = layer_reconstruct(layer_id, low_values, high_values, self.layer_scores_mask)

        # Concatenate the keys and values
        tensor = torch.stack([keys, values], dim=0)
        return tensor

