"""Optional transformer implementations, loaded only when selected."""

__all__ = ["HadamardTransform", "KVServeTransformer", "OrthogonalTransform"]


def __getattr__(name):
    if name == "HadamardTransform":
        from .hadamard_func import HadamardTransform

        return HadamardTransform
    if name == "KVServeTransformer":
        from .kvserve_transformer import KVServeTransformer

        return KVServeTransformer
    if name == "OrthogonalTransform":
        from .orthogonal import OrthogonalTransform

        return OrthogonalTransform
    raise AttributeError(name)
