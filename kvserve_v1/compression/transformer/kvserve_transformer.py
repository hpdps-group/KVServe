"""Configurable transforms for the modular compression pipeline."""

import torch
from kvserve_v1.compression.components import Transformer


class KVServeTransformer(Transformer):
    def __init__(self, **kwargs):
        self.transform_type = None
        self.seed = None
        self.transformer = None
        self.update_params(**kwargs)

    def validate(self):
        if self.transform_type not in ("hadamard", "dct", "haar"):
            raise ValueError(f"Invalid transform type: {self.transform_type}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    def update_params(self, **kwargs):
        kind = kwargs.get("transform_type", self.transform_type or "hadamard")
        seed = kwargs.get("seed", self.seed if self.seed is not None else 0x3333)
        if kind not in ("hadamard", "dct", "haar") or type(seed) is not int:
            raise ValueError("Expected hadamard/dct/haar and an integer seed")
        if (kind, seed) == (self.transform_type, self.seed):
            return
        if kind == "hadamard":
            from .hadamard_func import HadamardTransform

            implementation = HadamardTransform(seed)
        else:
            from .orthogonal import OrthogonalTransform

            implementation = OrthogonalTransform(kind, seed)
        self.transform_type, self.seed, self.transformer = kind, seed, implementation

    def transform(self, layer_id: int, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.transform_type != "hadamard":
            return self.transformer.transform(layer_id, tensor, **kwargs)
        out = torch.empty_like(tensor)
        out[0] = self.transformer.transform(layer_id, tensor[0], **kwargs)
        out[1] = self.transformer.transform(layer_id, tensor[1], **kwargs)
        return out

    def inverse(self, layer_id: int, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.transform_type != "hadamard":
            return self.transformer.inverse(layer_id, tensor, **kwargs)
        out = torch.empty_like(tensor)
        out[0] = self.transformer.inverse(layer_id, tensor[0], **kwargs)
        out[1] = self.transformer.inverse(layer_id, tensor[1], **kwargs)
        return out
