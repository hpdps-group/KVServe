"""
Hadamard transform functions for KV cache compression
Implements Fast Walsh-Hadamard Transform (FWHT) with Rademacher signs
"""

import torch
import math
import fast_hadamard_transform
from typing import List

class HadamardTransform:
    """
    Vectorized Hadamard Transform implementation
    
    Applies Fast Walsh-Hadamard Transform (FWHT) to KV cache tensors
    with deterministic Rademacher signs for compression optimization.
    """

    def __init__(
        self, 
        base_seed: int = 0xC0FEBABE
    ):
        """
        Initialize Hadamard Transform
        
        Args:
            base_seed: Base seed for generating Rademacher signs (default: 0xC0FEBABE)
        """
        self.base_seed = int(base_seed, 0) if isinstance(base_seed, str) else int(base_seed)
        # Cache signs tensors to avoid repeated CPU generation and Host-to-Device transfer
        self.signs_cache = {}

    def _get_seed(
        self, 
        layer_idx: int, 
        head_idx: int
    ) -> int:
        """
        Generate deterministic seed for a specific layer and head
        
        Args:
            layer_idx: Layer index
            head_idx: Head index within the layer
            
        Returns:
            Combined seed value: base_seed ^ (layer_idx << 16) ^ head_idx
        """
        return self.base_seed ^ (layer_idx << 16) ^ head_idx

    def _pow2_chunks(
        self, 
        dim: int
    ) -> List[int]:
        """
        Split dimension into a list of descending powers-of-two
        
        Used when head_dim is not a power of two, splitting it into
        power-of-two chunks for efficient FWHT computation.
        
        Args:
            dim: Dimension size to split
            
        Returns:
            List of power-of-two chunk sizes in descending order
            Example: 96 -> [64, 32], 100 -> [64, 32, 4]
        """
        chunks: List[int] = []
        remaining = dim
        while remaining > 0:
            chunk = 1 << (remaining.bit_length() - 1)
            chunks.append(chunk)
            remaining -= chunk
        return chunks

    def _fwht_in_chunks(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Fast Walsh-Hadamard Transform (FWHT) to the last dimension
        
        If head_dim is not a power of two, splits into power-of-two chunks
        and applies FWHT to each chunk separately, then concatenates results.
        
        Args:
            x: Input tensor of shape [..., head_dim]
            
        Returns:
            Transformed tensor with same shape as input
        """
        chunks = self._pow2_chunks(x.shape[-1])
        outputs = []
        start = 0
        for size in chunks:
            part = x.narrow(-1, start, size)

            transformed = fast_hadamard_transform.hadamard_transform(
                part.contiguous(), scale=1.0 / math.sqrt(size)
            )
            outputs.append(transformed)
            start += size
        
        return torch.cat(outputs, dim=-1)

    def get_rademacher_signs(
        self, 
        layer_idx: int,
        num_heads: int, 
        head_dim: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Generate the Rademacher sign tensor for ALL heads in a layer
        
        Generates deterministic random ±1 signs for each head using seed-based
        random number generation. Signs are cached to avoid repeated generation.
        
        Args:
            layer_idx: Current layer index
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            device: Target device for tensor
            dtype: Target dtype for tensor
            
        Returns:
            signs: Tensor of shape [1, 1, num_heads, head_dim] for broadcasting
                  Can be directly multiplied with [num_blocks, block_size, num_heads, head_dim]
        """
        # Check cache (includes device and dtype to handle context changes)
        cache_key = (layer_idx, num_heads, head_dim, device, dtype)
        if cache_key in self.signs_cache:
            return self.signs_cache[cache_key]

        signs_list = []

        gen = torch.Generator(device="cpu")
        
        # Generate signs for each head
        for h in range(num_heads):
            seed = self._get_seed(layer_idx, h)
            gen.manual_seed(seed)
            
            # Generate {0, 1} -> convert to {-1, 1}
            s = torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.int8)
            s = s.float().mul_(2).sub_(1)
            signs_list.append(s)
            
        # Stack heads -> [num_heads, head_dim]
        signs = torch.stack(signs_list).to(device=device, dtype=dtype)
        
        # Reshape for broadcasting: [1, 1, num_heads, head_dim]
        # This allows direct multiplication with [num_blocks, block_size, num_heads, head_dim]
        signs = signs.view(1, 1, num_heads, head_dim)
        
        # Store in cache
        self.signs_cache[cache_key] = signs
        
        return signs

    def transform(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Apply Hadamard transform to the KV tensor
        
        Forward transform pipeline:
        1. Generate Rademacher signs for all heads
        2. Element-wise multiply with signs
        3. Apply Fast Walsh-Hadamard Transform (FWHT)
        
        Args:
            layer_id: Current layer index for seed generation
            tensor: Input tensor [num_blocks, block_size, num_heads, head_dim]
            **kwargs: Additional transformation parameters
            
        Returns:
            transformed_tensor: Transformed tensor of same shape [num_blocks, block_size, num_heads, head_dim]
        """
        assert tensor.dim() == 4, f"Expected 4D tensor [num_blocks, block_size, num_heads, head_dim], got {tensor.shape}"
        num_blocks, block_size, num_heads, head_dim = tensor.shape

        # Step 1: Get Rademacher signs for all heads (Shape: [1, 1, num_heads, head_dim])
        # Signs are cached to avoid repeated generation
        signs = self.get_rademacher_signs(layer_id, num_heads, head_dim, tensor.device, tensor.dtype)

        # Step 2: Element-wise multiplication (using broadcasting)
        # [num_blocks, block_size, num_heads, head_dim] * [1, 1, num_heads, head_dim] -> [num_blocks, block_size, num_heads, head_dim]
        signed_kv = tensor * signs

        # Step 3: Apply FWHT (Batched)
        # Library function handles batch dimensions automatically, operates on last dimension
        output = self._fwht_in_chunks(signed_kv)
        return output

    def inverse(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Inverse Hadamard transform to restore original values
        
        Inverse transform logic:
        - Inverse(H * S * x) = S * Inverse(H) * (H * S * x)
        - Since H is symmetric orthogonal (scaled), H^{-1} is proportional to H
        - And S is its own inverse (1/1=1, 1/-1=-1)
        - Steps: 1. Apply FWHT, 2. Multiply by signs
        
        Args:
            layer_id: Current layer index for seed generation
            tensor: Transformed tensor [num_blocks, block_size, num_heads, head_dim]
            **kwargs: Additional transformation parameters
            
        Returns:
            restored_tensor: Restored original tensor [num_blocks, block_size, num_heads, head_dim]
        """
        assert tensor.dim() == 4, f"Expected 4D tensor [num_blocks, block_size, num_heads, head_dim], got {tensor.shape}"
        num_blocks, block_size, num_heads, head_dim = tensor.shape

        # Step 1: Apply FWHT first (Hadamard matrix is symmetric)
        rotated_back = self._fwht_in_chunks(tensor)

        # Step 2: Multiply by signs to restore original values
        signs = self.get_rademacher_signs(layer_id, num_heads, head_dim, tensor.device, tensor.dtype)
        original_kv = rotated_back * signs

        return original_kv