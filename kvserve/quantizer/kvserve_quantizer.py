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
        
        # Validate quantizer parameters
        self.validate()

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

        # Store actual num_heads detected at runtime (initialized to None, set on first quantize call)
        self.actual_num_heads = None

        # TODO: add cache for quantized keys and values
        # self._high_quantized_key_cache = Deque[torch.Tensor]()
        # self._high_quantized_value_cache = Deque[torch.Tensor]()
        # self._low_quantized_key_cache = Deque[torch.Tensor]()
        # self._low_quantized_value_cache = Deque[torch.Tensor]()

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
    
    def _regenerate_head_mask(self, actual_num_heads: int, hybrid_ratio: float) -> None:
        """
        Regenerate head_scores_mask based on actual number of heads
        
        Args:
            actual_num_heads: Actual number of heads detected from tensor
            hybrid_ratio: Ratio of heads to use low precision
        """
        low_precision_num_heads = round(actual_num_heads * hybrid_ratio)
        dynamic_mask = torch.zeros(actual_num_heads, dtype=torch.bool)
        dynamic_mask[:low_precision_num_heads] = True
        
        # Expand to match layer structure [num_layers, num_heads]
        num_layers = self.head_scores_mask.shape[0] if len(self.head_scores_mask.shape) == 2 else 1
        self.head_scores_mask = dynamic_mask.unsqueeze(0).repeat(num_layers, 1)
        self.actual_num_heads = actual_num_heads

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
            
            # Dynamically adjust head_scores_mask if it doesn't match actual num_heads
            if self.split_type == "head":
                keys = tensor[0]
                actual_num_heads = keys.shape[2]  # [num_blocks, block_size, num_heads, head_size]
                
                # Check if mask needs adjustment
                expected_num_heads = self.head_scores_mask.shape[1] if len(self.head_scores_mask.shape) == 2 else self.head_scores_mask.shape[-1] if self.head_scores_mask.numel() > 0 else 0
                
                if self.actual_num_heads is None or expected_num_heads != actual_num_heads:
                    # Regenerate mask for the detected head count
                    self._regenerate_head_mask(actual_num_heads, self.hybrid_ratio)

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
            # Save head split info for correct decompression
            "split_type": self.split_type,
            "actual_num_heads": self.actual_num_heads,  # Detected at runtime
            "hybrid_ratio": self.hybrid_ratio,
        }
        
        # Add the actual mask used for this layer to ensure consistency across GPUs
        if self.split_type == "head":
            meta_data["head_mask_for_layer"] = self.head_scores_mask[layer_id].cpu().tolist()
        elif self.split_type == "layer":
            meta_data["layer_mask_for_layer"] = self.layer_scores_mask[layer_id].cpu().item()
        
        return tensor, meta_data

    def dequantize(
        self, 
        layer_end_id: int,
        tensor: torch.Tensor, 
        meta_data: Any,
        **kwargs,
    ) -> torch.Tensor:
        """
        Batch-capable dequantization.
        tensor: [layers, 2, num_blocks, block_size, num_heads, head_size] or single-layer.
        meta_data: list aligned by layer, or single dict.
        """
        single_layer = False
        if tensor.dim() == 5:
            tensor = tensor.unsqueeze(0)
            single_layer = True

        if isinstance(meta_data, dict):
            meta_list: List[Dict[str, Any]] = [meta_data]
        else:
            meta_list = list(meta_data)

        num_layers = tensor.shape[0]
        start_layer = layer_end_id + 1 - num_layers
        
        # Get target dtype from metadata and convert tensor upfront to avoid auto-casting back to uint8
        first_meta = meta_list[0] if meta_list else {}
        target_dtype = first_meta.get("high_key_meta", {}).get("original_dtype", torch.float16)
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)

        for i, layer_id in enumerate(range(start_layer, layer_end_id + 1)):
            per_meta = meta_list[i] if i < len(meta_list) else meta_list[-1]

            mask_for_layer = None
            if per_meta.get("split_type") == "head":
                if "head_mask_for_layer" in per_meta:
                    mask_for_layer = torch.tensor(per_meta["head_mask_for_layer"], dtype=torch.bool)
                elif "actual_num_heads" in per_meta and per_meta["actual_num_heads"] is not None:
                    actual_num_heads = per_meta["actual_num_heads"]
                    hybrid_ratio = per_meta.get("hybrid_ratio", self.hybrid_ratio)
                    self._regenerate_head_mask(actual_num_heads, hybrid_ratio)
            elif per_meta.get("split_type") == "layer":
                if "layer_mask_for_layer" in per_meta:
                    mask_for_layer = torch.tensor([per_meta["layer_mask_for_layer"]], dtype=torch.bool)

            # Restore the low and high precision parts
            if self.split_type == "head":
                if mask_for_layer is not None:
                    temp_mask = mask_for_layer.unsqueeze(0)  # [1, num_heads]
                    low_keys, high_keys = head_restore(0, tensor[i, 0], temp_mask)
                    low_values, high_values = head_restore(0, tensor[i, 1], temp_mask)
                else:
                    low_keys, high_keys = head_restore(layer_id, tensor[i, 0], self.head_scores_mask)
                    low_values, high_values = head_restore(layer_id, tensor[i, 1], self.head_scores_mask)
            elif self.split_type == "layer":
                if mask_for_layer is not None:
                    temp_mask = mask_for_layer
                    low_keys, high_keys = layer_restore(0, tensor[i, 0], temp_mask)
                    low_values, high_values = layer_restore(0, tensor[i, 1], temp_mask)
                else:
                    low_keys, high_keys = layer_restore(layer_id, tensor[i, 0], self.layer_scores_mask)
                    low_values, high_values = layer_restore(layer_id, tensor[i, 1], self.layer_scores_mask)

            # Dequantize the high and low precision parts
            high_keys = dequantize(high_keys, per_meta["high_key_meta"])
            high_values = dequantize(high_values, per_meta["high_value_meta"])
            low_keys = dequantize(low_keys, per_meta["low_key_meta"])
            low_values = dequantize(low_values, per_meta["low_value_meta"])

            # Reconstruct the keys and values
            if self.split_type == "head":
                if mask_for_layer is not None:
                    temp_mask = mask_for_layer.unsqueeze(0)  # [1, num_heads]
                    tensor[i, 0] = head_reconstruct(0, low_keys, high_keys, temp_mask)
                    tensor[i, 1] = head_reconstruct(0, low_values, high_values, temp_mask)
                else:
                    tensor[i, 0] = head_reconstruct(layer_id, low_keys, high_keys, self.head_scores_mask)
                    tensor[i, 1] = head_reconstruct(layer_id, low_values, high_values, self.head_scores_mask)
            elif self.split_type == "layer":
                if mask_for_layer is not None:
                    temp_mask = mask_for_layer
                    tensor[i, 0] = layer_reconstruct(0, low_keys, high_keys, temp_mask)
                    tensor[i, 1] = layer_reconstruct(0, low_values, high_values, temp_mask)
                else:
                    tensor[i, 0] = layer_reconstruct(layer_id, low_keys, high_keys, self.layer_scores_mask)
                    tensor[i, 1] = layer_reconstruct(layer_id, low_values, high_values, self.layer_scores_mask)

        if single_layer:
            tensor = tensor.squeeze(0)
        return tensor
