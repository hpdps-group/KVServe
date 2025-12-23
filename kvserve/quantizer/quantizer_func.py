"""
Quantization functions for KV cache compression
Implements min-max quantization with configurable axis strategies
"""

import torch
from typing import List, Tuple, Dict

def quantize(
    max_value: int,
    tensor: torch.Tensor, 
    axis: str, 
) -> Tuple[torch.Tensor, Dict]:
    """
    Quantize tensor using min-max quantization
    
    Maps tensor values from [min, max] range to [0, max_value] integer range.
    Uses per-axis min/max calculation for better precision preservation.
    
    Args:
        max_value: Maximum quantization value (typically 12, 16, or 255)
        tensor: Input tensor to quantize, shape: [num_blocks, block_size, num_heads, head_size]
        axis: Quantization axis strategy:
            - "channel": Compute min/max along dimensions [0, 1]
            - "token": Compute min/max along dimensions [2, 3]
            - "tensor": Compute min/max across all dimensions [0, 1, 2, 3]
            
    Returns:
        Tuple of (quantized_tensor, metadata) where:
            quantized_tensor: Quantized tensor as uint8
            metadata: Dictionary containing:
                min_val: Minimum value used for quantization
                quant_scale: Scale factor for dequantization
                original_dtype: Original tensor dtype
                quant_dtype: Quantized dtype (torch.uint8)
    """
    # If tensor is empty, return empty tensor and metadata
    if tensor.numel() == 0:
        return tensor.to(torch.uint8), {"min_val": None, "quant_scale": None, "original_dtype": tensor.dtype, "quant_dtype": torch.uint8}

    # Convert axis to list of dimensions
    if axis == "channel":
        axis = [0, 1]
    elif axis == "token":
        axis = [2, 3]
    elif axis == "tensor":
        axis = [0, 1, 2, 3]
    else:
        raise ValueError(f"Invalid axis: {axis}")

    # Step 1: Calculate Max/Min along specified axis
    max_ = torch.amax(tensor, dim=axis, keepdim=True)
    min_ = torch.amin(tensor, dim=axis, keepdim=True)

    # Step 2: Calculate scale factor (add eps to prevent division by zero)
    scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
    
    # Step 3: Quantization core logic
    # Formula: Q = clamp(round((x - min) / scale), 0, max_value)
    tensor_sub = tensor.sub(min_)  # Now tensor_sub >= 0
    
    quant_float = tensor_sub.div_(scale)  # In-place division
    
    # Clamp to prevent overflow from precision issues
    quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

    # Step 4: Prepare metadata for dequantization
    meta_data = {
        "min_val": min_.to(tensor.dtype),
        "quant_scale": scale.to(tensor.dtype),
        "original_dtype": tensor.dtype,
        "quant_dtype": torch.uint8,
    }
    
    return quantized_tensor, meta_data

def dequantize(
    quantized_tensor: torch.Tensor,
    meta_data: Dict,
) -> torch.Tensor:
    """
    Dequantize tensor using stored quantization metadata
    
    Restores original value range using inverse quantization formula.
    Formula: Real = Quantized * Scale + Min_Value
    
    Args:
        quantized_tensor: Quantized tensor (uint8)
        meta_data: Dictionary containing quantization parameters:
            min_val: Minimum value used during quantization
            quant_scale: Scale factor for dequantization
            original_dtype: Original tensor dtype to restore
            quant_dtype: Quantized dtype (torch.uint8)
            
    Returns:
        Dequantized tensor with original dtype and value range restored
    """
    # If tensor is empty, return empty tensor
    if meta_data["min_val"] is None:
        return quantized_tensor.to(meta_data["original_dtype"])

    # Step 1: Extract metadata
    min_val = meta_data["min_val"]
    quant_scale = meta_data["quant_scale"]

    # Step 2: Type conversion (cast to float)
    quant_float = quantized_tensor.to(meta_data["quant_dtype"])

    # Step 3: Dequantization calculation
    # Formula: result = quantized * scale + min_value
    dequantized_tensor = quant_float * quant_scale + min_val

    return dequantized_tensor