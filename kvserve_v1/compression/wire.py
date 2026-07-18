"""GPU-resident wire format for compressed KV payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kvserve_v1.compression.compression_manager import CompressedKVData

# Tensor placeholder uses a 2-element list `[_AUX_PLACEHOLDER, idx]` rather
# than a single-key dict. qparams structures observed in the wild are dicts
# (per-layer) of dicts (per-precision-band), never lists at value positions,
# so a list-typed leaf is unambiguous and survives msgpack roundtrip.
_AUX_PLACEHOLDER = "__kvserve_aux_tensor__"


@dataclass
class CompressedWire:
    """Compressed KV payload split into msgpack metadata and NCCL tensors."""

    meta: dict[str, Any]
    body_chunks: list[torch.Tensor]
    aux_tensors: list[torch.Tensor]

    @property
    def nbytes(self) -> int:
        tensors = self.body_chunks + self.aux_tensors
        return sum(t.numel() * t.element_size() for t in tensors)


def build_wire(
    compressed: CompressedKVData,
    max_chunk_bytes: int,
) -> CompressedWire:
    """Build a GPU-resident wire payload from compressed KV data."""
    aux_tensors: list[torch.Tensor] = []

    if compressed.is_chunked:
        if not compressed.chunks or not compressed.chunk_metadata:
            raise ValueError("Chunked CompressedKVData is missing chunks or metadata")
        meta = {
            "is_chunked": True,
            "layer_id": compressed.layer_id,
            "original_size": compressed.original_size,
            "compressed_size": compressed.compressed_size,
            "metadata": _pack_metadata(compressed.metadata, aux_tensors),
            "chunk_metadata": [
                _pack_metadata(chunk_meta, aux_tensors)
                for chunk_meta in compressed.chunk_metadata
            ],
            "body_mode": "manager_chunks",
        }
        body_chunks = [chunk.contiguous() for chunk in compressed.chunks]
        wire_aux_tensors = _coalesce_aux_tensors(meta, aux_tensors)
        _assert_no_tensor(meta)
        return CompressedWire(
            meta=meta, body_chunks=body_chunks, aux_tensors=wire_aux_tensors)

    if compressed.compressed_tensor is None:
        raise ValueError("Non-chunked CompressedKVData is missing compressed_tensor")

    body = compressed.compressed_tensor.contiguous()
    body_spec = _tensor_spec(body)
    body_nbytes = body.numel() * body.element_size()
    if body_nbytes > max_chunk_bytes:
        body_chunks = _split_tensor_bytes(body, max_chunk_bytes)
        body_mode = "byte_chunks"
    else:
        body_chunks = [body]
        body_mode = "single"

    meta = {
        "is_chunked": False,
        "layer_id": compressed.layer_id,
        "original_size": compressed.original_size,
        "compressed_size": compressed.compressed_size,
        "metadata": _pack_metadata(compressed.metadata, aux_tensors),
        "body_mode": body_mode,
        "body_spec": body_spec,
    }
    wire_aux_tensors = _coalesce_aux_tensors(meta, aux_tensors)
    _assert_no_tensor(meta)
    return CompressedWire(
        meta=meta, body_chunks=body_chunks, aux_tensors=wire_aux_tensors)


def restore_from_wire(wire: CompressedWire) -> CompressedKVData:
    """Restore compressed KV data from a received wire payload.

    The original P-side request_id is recovered from the inner metadata; the
    connector's transfer_id (used as routing key) is intentionally not threaded
    in here.
    """
    meta = wire.meta
    aux_tensors = _expand_aux_tensors(
        wire.aux_tensors, meta.get("aux_layout"))
    is_chunked = bool(meta["is_chunked"])

    inner_meta = _unpack_metadata(meta.get("metadata"), aux_tensors)
    request_id = "unknown"
    if isinstance(inner_meta, dict):
        request_id = str(inner_meta.get("request_id", "unknown"))

    if is_chunked:
        return CompressedKVData(
            request_id=request_id,
            layer_id=int(meta.get("layer_id", -1)),
            compressed_tensor=None,
            metadata=inner_meta,
            original_size=int(meta.get("original_size", 0)),
            compressed_size=int(meta.get("compressed_size", 0)),
            is_chunked=True,
            chunks=wire.body_chunks,
            chunk_metadata=[
                _unpack_metadata(chunk_meta, aux_tensors)
                for chunk_meta in meta.get("chunk_metadata", [])
            ],
        )

    if meta.get("body_mode") == "byte_chunks":
        body_bytes = torch.cat([chunk.contiguous().view(torch.uint8) for chunk in wire.body_chunks], dim=0)
        compressed_tensor = _restore_tensor_from_bytes(body_bytes, meta["body_spec"])
    else:
        compressed_tensor = wire.body_chunks[0]

    return CompressedKVData(
        request_id=request_id,
        layer_id=int(meta.get("layer_id", -1)),
        compressed_tensor=compressed_tensor,
        metadata=inner_meta,
        original_size=int(meta.get("original_size", 0)),
        compressed_size=int(meta.get("compressed_size", 0)),
        is_chunked=False,
    )


def _pack_metadata(obj: Any, aux_tensors: list[torch.Tensor]) -> Any:
    if not isinstance(obj, dict):
        return obj
    packed = dict(obj)
    if "quantization_params" in packed:
        packed["quantization_params"] = _pack_qparams(
            packed["quantization_params"], aux_tensors)
    return packed


def _unpack_metadata(obj: Any, aux_tensors: list[torch.Tensor]) -> Any:
    if isinstance(obj, dict) and "quantization_params" in obj:
        obj = dict(obj)
        obj["quantization_params"] = _unpack_qparams(
            obj["quantization_params"], aux_tensors)
    return obj


def _pack_qparams(obj: Any, aux_tensors: list[torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        idx = len(aux_tensors)
        aux_tensors.append(obj.contiguous())
        return [_AUX_PLACEHOLDER, idx]
    if isinstance(obj, dict):
        return {k: _pack_qparams(v, aux_tensors) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack_qparams(v, aux_tensors) for v in obj]
    return obj


def _is_aux_placeholder(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and len(obj) == 2
        and isinstance(obj[0], str)
        and obj[0] == _AUX_PLACEHOLDER
    )


def _unpack_qparams(obj: Any, aux_tensors: list[torch.Tensor]) -> Any:
    if _is_aux_placeholder(obj):
        return aux_tensors[int(obj[1])]
    if isinstance(obj, dict):
        return {k: _unpack_qparams(v, aux_tensors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_qparams(v, aux_tensors) for v in obj]
    return obj


def _coalesce_aux_tensors(
    meta: dict[str, Any],
    aux_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Pack metadata tensors into one byte tensor to reduce NCCL ops.

    TileLang metadata can contain hundreds of small tensors. Sending each as a
    separate NCCL message is fragile on the Socket path and adds high control
    overhead, so the wire format keeps the existing placeholder indexes but
    stores their bytes in one auxiliary tensor plus a compact layout table.
    """
    if len(aux_tensors) <= 1:
        return aux_tensors

    byte_chunks: list[torch.Tensor] = []
    layout: list[dict[str, Any]] = []
    offset = 0
    device = aux_tensors[0].device

    for tensor in aux_tensors:
        tensor = tensor.contiguous()
        itemsize = tensor.element_size()
        padding = (-offset) % itemsize
        if padding:
            byte_chunks.append(torch.zeros(padding, dtype=torch.uint8, device=device))
            offset += padding

        tensor_bytes = tensor.view(torch.uint8).flatten()
        nbytes = tensor_bytes.numel()
        layout.append({
            "storage": 0,
            "offset": offset,
            "nbytes": nbytes,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        })
        byte_chunks.append(tensor_bytes)
        offset += nbytes

    meta["aux_layout"] = {
        "mode": "coalesced_bytes",
        "tensors": layout,
    }
    return [torch.cat(byte_chunks, dim=0).contiguous()]


def _expand_aux_tensors(
    wire_aux_tensors: list[torch.Tensor],
    aux_layout: Any,
) -> list[torch.Tensor]:
    if not aux_layout:
        return wire_aux_tensors
    if aux_layout.get("mode") != "coalesced_bytes":
        raise ValueError(f"Unknown aux_layout mode: {aux_layout.get('mode')}")
    if not wire_aux_tensors:
        return []

    storage = wire_aux_tensors[0].contiguous().view(torch.uint8).flatten()
    restored: list[torch.Tensor] = []
    for spec in aux_layout.get("tensors", []):
        if int(spec.get("storage", 0)) != 0:
            raise ValueError("Only single-storage aux_layout is supported")
        start = int(spec["offset"])
        end = start + int(spec["nbytes"])
        dtype = getattr(torch, spec["dtype"])
        restored.append(storage[start:end].view(dtype).reshape(spec["shape"]))
    return restored


def _split_tensor_bytes(tensor: torch.Tensor, max_chunk_bytes: int) -> list[torch.Tensor]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    byte_view = tensor.contiguous().view(torch.uint8).flatten()
    return [
        byte_view[start:start + max_chunk_bytes].contiguous()
        for start in range(0, byte_view.numel(), max_chunk_bytes)
    ]


def _restore_tensor_from_bytes(byte_tensor: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, spec["dtype"])
    return byte_tensor.contiguous().view(dtype).reshape(spec["shape"])


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def _assert_no_tensor(obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        raise TypeError(
            "CompressedWire metadata still contains a tensor; extend wire packing "
            "for this metadata field.")
    if isinstance(obj, dict):
        for value in obj.values():
            _assert_no_tensor(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _assert_no_tensor(value)
