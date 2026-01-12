"""
KVServe Transformer for KV cache transformation
Implements transformations for compression with transform support
"""

import torch
from kvserve.manager.components import Transformer
from kvserve.transformer.hadamard_func import HadamardTransform

class KVServeTransformer(Transformer):
    """
    KVServe Transformer implementation
    Currently supports Hadamard transform for KV cache compression
    """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize KVServe Transformer
        
        Args:
            **kwargs: Configuration parameters including:
                transform_type: Type of transformation, "hadamard" (default: "hadamard")
                seed: Base seed for Rademacher sign generation (default: 0x3333)
        """
        # Update parameters from kwargs
        self.transform_type = kwargs.get("transform_type", "hadamard")
        self.seed = kwargs.get("seed", 0x3333)
        self.transformer = None

        # Validate transformer parameters
        self.validate()

        # Initialize the transform functions
        match self.transform_type:
            case "hadamard":
                self.transformer = HadamardTransform(self.seed)
            case _:
                raise ValueError(f"Invalid transform type: {self.transform_type}")

    def validate(
        self,
    ) -> None:
        """
        Validate transformer parameters
        
        Raises:
            AssertionError: If transform_type is invalid or seed is None
        """
        assert self.transform_type in ["hadamard"], \
            f"Invalid transform type: {self.transform_type}"
        assert self.seed is not None, \
            "Seed is required"

    def update_params(
        self,
        **kwargs
    ) -> None:
        """
        Update transformer parameters dynamically, used in transform() to update the transformer parameters for every request
        
        Args:
            **kwargs: Parameters to update
        """
        # Update the parameters with the new values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)  

        # Validate the parameters
        self.validate()

        match self.transform_type:
            case "hadamard":
                self.transformer = HadamardTransform(self.seed)
            case _:
                raise ValueError(f"Invalid transform type: {self.transform_type}")

    def transform(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Transform KV cache tensor using configured transformation
        
        Args:
            layer_id: Layer ID for transformation
            tensor: KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
                   where first dimension is [keys, values]
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed tensor with same shape as input
        """
        # Don't need to update parameters here because compression manager will handle it
        # if layer_id == 0:
        #     self.update_params(**kwargs)

        out_tensor = torch.empty_like(tensor)
        
        # Apply transformation directly into output slices
        # tensor[0] is keys, tensor[1] is values
        out_tensor[0] = self.transformer.transform(layer_id, tensor[0], **kwargs)
        out_tensor[1] = self.transformer.transform(layer_id, tensor[1], **kwargs)
        
        return out_tensor

    def inverse(
        self, 
        layer_id: int,
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Inverse transform KV cache tensor to restore original values
        
        Args:
            layer_id: Layer ID for inverse transformation
            tensor: Transformed KV cache tensor [2, num_blocks, block_size, num_heads, head_size]
            **kwargs: Additional transformation parameters
            
        Returns:
            Restored tensor with original values
        """        
        out_tensor = torch.empty_like(tensor)
        
        # Apply inverse transformation directly into output slices
        out_tensor[0] = self.transformer.inverse(layer_id, tensor[0], **kwargs)
        out_tensor[1] = self.transformer.inverse(layer_id, tensor[1], **kwargs)
        
        return out_tensor
