"""
Bitpacking compression functions for KV cache

Implements lightweight bit-packing of uint8 tensors into float32 containers
for efficient storage/transmission, with corresponding unpacking utilities.
"""

import torch
import torchac_cuda
from kvserve.manager import EasyDist

class BitpackingCodec:
    """
    Bit-packing codec for uint8 tensors

    Provides encode/decode helpers that pack multiple low-bit-width values into
    float32 containers (leveraging 32 available bits) and restore them back to
    the original dtype/shape.
    """

    def __init__(
        self, 
        **kwargs
    ) -> None:

        pass

    def encode(
        self, 
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Pack uint8 tensor to float32 container using bit packing

        Flattens a uint8 view, determines the minimal bits needed, packs
        multiple values into 32-bit integers (little-endian layout), reinterprets
        as float32, and serializes via EasyDist.

        Args:
            tensor: Input tensor to compress; will be viewed as uint8
            **kwargs: Reserved for future options

        Returns:
            Compressed tensor (EasyDist-packed) containing packed data and bit
            width metadata
        """
        # Compute the maximum value needed for bit packing
        tensor = tensor.view(torch.uint8)
        max_val = tensor.max().item()
        if max_val == 0:
            bits = 1
        else:
            bits = int(torch.ceil(torch.log2(torch.tensor(max_val + 1, dtype=torch.float32))).item())
        
        # 1. Calculate pack ratio
        pack_ratio = 32 // bits
        
        # 2. Flatten and convert to int32 for bitwise ops
        flat_tensor = tensor.flatten().to(torch.int32)
        
        # 3. Padding if needed
        num_elements = flat_tensor.numel()
        padding = (pack_ratio - (num_elements % pack_ratio)) % pack_ratio
        if padding > 0:
            flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding), value=0)
        
        # 4. Reshape to [N, pack_ratio]
        reshaped = flat_tensor.view(-1, pack_ratio)
        
        # 5. Shift values
        # Little Endian packing: first element at lowest bits
        shift_vals = torch.arange(0, 32, bits, device=tensor.device, dtype=torch.int32)
        
        # 6. Bitwise packing
        shifted = reshaped << shift_vals
        packed_int32 = torch.sum(shifted, dim=1).to(torch.int32)
        
        # 7. Reinterpret as float32
        packed_float32 = packed_int32.view(torch.float32)
        
        compressed_data = {"packed_tensor": packed_float32, "bits": bits}
        compressed_tensor, _ = EasyDist.pack_object(compressed_data)

        return compressed_tensor

        
    def decode(
        self, 
        compressed_tensor: torch.Tensor,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs
    ) -> torch.Tensor:
    
        """
        Unpack float32 container back to uint8 tensor

        Reverses `encode`: deserializes packed data, unpacks bit-packed values
        from 32-bit containers, trims padding, restores dtype and shape.

        Args:
            compressed_tensor: Packed tensor produced by encode()
            original_dtype: Original dtype name (e.g., "uint8")
            original_shape: Original tensor shape list
            device: Target device for output tensor
            **kwargs: Reserved for future options

        Returns:
            Tensor restored to original dtype and shape
        """
        compressed_data = EasyDist.unpack_object(compressed_tensor)
        packed_tensor = compressed_data["packed_tensor"].to(device)
        bits = compressed_data["bits"]
        # 1. Reinterpret as int32
        packed_int32 = packed_tensor.view(torch.int32)
        
        # 2. Prepare unshift
        shift_vals = torch.arange(0, 32, bits, device=device, dtype=torch.int32)
        
        # 3. Expand and shift right
        # packed_int32: [N] -> [N, 1]
        # shift_vals: [pack_ratio]
        # Broadcast -> [N, pack_ratio]
        unpacked = (packed_int32.unsqueeze(-1) >> shift_vals) & ((1 << bits) - 1)
        
        # 4. Flatten
        flat_tensor = unpacked.flatten()
        
        # 5. Remove padding (slice to original size)
        original_numel = torch.prod(torch.tensor(original_shape))
        flat_tensor = flat_tensor[:original_numel]
        
        # 6. Reshape to original shape and cast to uint8
        target_dtype = getattr(torch, original_dtype)

        return flat_tensor.to(target_dtype).reshape(original_shape)
