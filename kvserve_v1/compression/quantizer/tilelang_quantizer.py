"""TileLang fused Hadamard+quantizer — drop-in Quantizer for CompressionManager.

Selected when quantizer_config.impl / top-level impl == "tilelang_fused".
Replaces the lossy transformer+quantizer segment with KVHadamardQuantOp.compress_v3.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from kvserve_v1.compression.components import Quantizer

_ROUTING_KEYS = frozenset({
    "impl",
    "quantizer_impl",
    "tp_rank",
    "tensor_parallel_size",
})


def _f16(t: Any) -> Any:
    if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
        return t.to(torch.float16)
    return t


def _canon_meta(meta: dict) -> dict:
    """Make compress_v3 meta wire-safe (tensors only, no int dict keys)."""
    km = dict(meta["key_meta"])
    for k in ("min_val", "quant_scale"):
        if k in km:
            km[k] = _f16(km[k])
    out: dict[str, Any] = {"key_meta": km}
    vm = meta["value_meta"]
    if "per_group_meta" in vm:
        groups = [
            {
                "head_idx": g["head_idx"],
                "min_val": _f16(g["min_val"]),
                "quant_scale": _f16(g["quant_scale"]),
            }
            for g in vm["per_group_meta"].values()
        ]
        out["value_meta"] = {"per_group_list": groups}
    else:
        out_vm = dict(vm)
        for k in ("min_val", "quant_scale"):
            if k in out_vm:
                out_vm[k] = _f16(out_vm[k])
        out["value_meta"] = out_vm
    return out


def _decanon_meta(c: dict) -> dict:
    out: dict[str, Any] = {"key_meta": c["key_meta"]}
    vm = c["value_meta"]
    if "per_group_list" in vm:
        out["value_meta"] = {
            "per_group_meta": {i: g for i, g in enumerate(vm["per_group_list"])}
        }
    else:
        out["value_meta"] = vm
    return out


class TileLangFusedQuantizer(Quantizer):
    """Quantizer facade over TileLang KVHadamardQuantOp.compress_v3."""

    # CompressionManager uses this to write quant output straight into the codec buffer.
    supports_out_buffer = True

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = dict(kwargs)
        self._op = None
        self._op_key: Optional[tuple] = None

    def update_params(self, **kwargs: Any) -> None:
        if not kwargs:
            return
        changed = False
        for key, value in kwargs.items():
            if self._cfg.get(key) != value:
                self._cfg[key] = value
                changed = True
        if changed:
            self._op = None
            self._op_key = None

    def _op_kwargs(self, num_heads: int, head_dim: int) -> dict[str, Any]:
        cfg = {k: v for k, v in self._cfg.items() if k not in _ROUTING_KEYS}
        seed = cfg.pop("base_seed", None)
        if seed is None:
            seed = cfg.pop("seed", 0x3333)
        dtype = cfg.pop("dtype", torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        return dict(
            num_heads=num_heads,
            head_dim=head_dim,
            base_seed=int(seed),
            dtype=dtype,
            quant_type=cfg.pop("quant_type", "minmax"),
            model_name=cfg.get("model_name"),
            hybrid_ratio=float(cfg.get("hybrid_ratio", 0.8)),
            high_key_max_value=float(cfg.get("high_key_max_value", 12)),
            high_value_max_value=float(cfg.get("high_value_max_value", 8)),
            low_key_max_value=float(cfg.get("low_key_max_value", 6)),
            low_value_max_value=float(cfg.get("low_value_max_value", 4)),
            split_type=cfg.get("split_type", "head"),
            tensor_parallel_size=int(self._cfg.get("tensor_parallel_size", 1)),
            tp_rank=int(self._cfg.get("tp_rank", 0)),
        )

    def _ensure_op(self, sample: torch.Tensor):
        num_heads = int(sample.shape[-2])
        head_dim = int(sample.shape[-1])
        key = (
            num_heads,
            head_dim,
            int(self._cfg.get("base_seed", self._cfg.get("seed", 0x3333))),
            str(self._cfg.get("model_name")),
            float(self._cfg.get("hybrid_ratio", 0.8)),
            float(self._cfg.get("high_key_max_value", 12)),
            float(self._cfg.get("high_value_max_value", 8)),
            float(self._cfg.get("low_key_max_value", 6)),
            float(self._cfg.get("low_value_max_value", 4)),
            str(self._cfg.get("split_type", "head")),
            int(self._cfg.get("tensor_parallel_size", 1)),
            int(self._cfg.get("tp_rank", 0)),
        )
        if self._op is not None and self._op_key == key:
            return self._op
        from kvserve_v1.tilelang_ops.kv_hadamard_quant import KVHadamardQuantOp
        self._op = KVHadamardQuantOp(**self._op_kwargs(num_heads, head_dim))
        self._op_key = key
        return self._op

    def quantize(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, dict]:
        if kwargs:
            # `out` is a runtime buffer, not a config knobs.
            kwargs.pop("out", None)
            self.update_params(**kwargs)
        op = self._ensure_op(tensor)
        src = tensor if tensor.is_contiguous() else tensor.contiguous()
        q, meta = op.compress_v3(src, layer_id=layer_id, out=out)
        return q, _canon_meta(meta)

    def dequantize(
        self,
        layer_id: int,
        quantized_data: torch.Tensor,
        quantization_params: dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs:
            self.update_params(**kwargs)
        op = self._ensure_op(quantized_data)
        return op.decompress_v3(
            quantized_data.contiguous(),
            _decanon_meta(quantization_params),
            layer_id=layer_id,
        )
