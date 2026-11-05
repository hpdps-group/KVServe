# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from logging import getLogger as init_logger
from typing import Any

import torch

from .config import CompressionConfig
from .layout import ALIGNMENT, CompressionLayout, build_compression_layout
from .scores import HeadSelection, _load_scores, build_head_selection
from .spec import CacheSpec, round_up

DRIVER_GUARD_BYTES = 4 << 20
logger = init_logger(__name__)

# Both sizing probes and live codecs share nvCOMP's process-global allocator.
_workspace_lock = threading.RLock()
_workspace_allocators: dict[int, Callable[[int, Any], Any]] = {}


class _ExternalAllocation:
    def __init__(self, nbytes: int, stream: Any):
        cuda_stream = torch.cuda.ExternalStream(stream.ptr, device=stream.device)
        with torch.cuda.stream(cuda_stream):
            self.tensor = torch.empty(
                nbytes, dtype=torch.uint8, device=cuda_stream.device
            )
        self.ptr = self.tensor.data_ptr()


def _allocate_workspace(nbytes: int, stream: Any) -> Any:
    with _workspace_lock:
        allocator = _workspace_allocators.get(stream.ptr)
    return (
        allocator(nbytes, stream)
        if allocator is not None
        else _ExternalAllocation(nbytes, stream)
    )


def register_workspace_allocators(
    allocators: Mapping[int, Callable[[int, Any], Any]],
) -> None:
    from nvidia import nvcomp

    with _workspace_lock:
        if _workspace_allocators.keys() & allocators.keys():
            raise RuntimeError("nvCOMP stream already has a workspace allocator")
        if not _workspace_allocators:
            nvcomp.set_device_allocator(_allocate_workspace)
        _workspace_allocators.update(allocators)


def unregister_workspace_allocators(streams: Iterable[int]) -> None:
    from nvidia import nvcomp

    with _workspace_lock:
        for stream in streams:
            _workspace_allocators.pop(stream, None)
        if not _workspace_allocators:
            nvcomp.set_device_allocator()


@dataclass(frozen=True)
class CompressionMemoryPlan:
    config: CompressionConfig
    layout: CompressionLayout
    head_selection: HeadSelection
    codec_workspace_bytes: int
    codec_max_output_bytes: tuple[int, int]
    blocks_per_logical: int
    metadata_bytes: int
    retained_driver_bytes: int
    guard_bytes: int
    reserved_bytes: int


class _AuditAllocation:
    def __init__(self, tensor: torch.Tensor, allocator: _AuditAllocator, nbytes: int):
        self.tensor = tensor
        self.ptr = tensor.data_ptr()
        self._allocator = allocator
        self._nbytes = nbytes

    def __del__(self):
        self._allocator.release(self._nbytes)


class _AuditAllocator:
    def __init__(self, device: torch.device):
        self.device = device
        self.live_bytes = 0
        self.peak_bytes = 0
        self.requests: list[int] = []

    def __call__(self, nbytes: int, stream: Any) -> _AuditAllocation:
        self.requests.append(nbytes)
        tensor = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        accounted_bytes = round_up(nbytes, ALIGNMENT)
        self.live_bytes += accounted_bytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        allocation = _AuditAllocation(tensor, self, accounted_bytes)
        return allocation

    def release(self, nbytes: int) -> None:
        self.live_bytes -= nbytes


def _probe_operators(
    layout: CompressionLayout,
    device: torch.device,
    sizes: tuple[int, ...],
    warm_hadamard: bool,
) -> tuple[int, tuple[int, int], int]:
    from nvidia import nvcomp

    torch.cuda.synchronize(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    audit = _AuditAllocator(device)
    stream = torch.cuda.Stream(device=device)
    register_workspace_allocators({stream.cuda_stream: audit})
    max_outputs: list[int] = []
    codec = source = output = decoded = None
    source_array = output_array = decoded_array = encoded = encoded_array = None
    compression_configs = decompression_configs = None
    hadamard_input = hadamard_output = None
    try:
        if warm_hadamard:
            from .cpp_backend import hadamard_transform

            head_size = layout.layer_parts[0].head_size
            hadamard_input = torch.empty(
                (1, head_size), dtype=torch.bfloat16, device=device
            )
            hadamard_output = torch.empty_like(hadamard_input)
            hadamard_transform(hadamard_input, hadamard_output)

        codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
        probe_sizes = sorted(set(sizes))
        compression_configs = {
            size: codec.compression_config(size) for size in probe_sizes
        }
        decompression_configs = {
            size: codec.decompression_config(config)
            for size, config in compression_configs.items()
        }
        with torch.cuda.stream(stream):
            for u8_bytes in probe_sizes:
                source = torch.empty(u8_bytes, dtype=torch.uint8, device=device)
                source_array = nvcomp.as_array(source, cuda_stream=stream.cuda_stream)
                max_output = codec.get_max_comp_buffer_size(source_array)
                max_outputs.append(max_output)
                output = torch.empty(max_output, dtype=torch.uint8, device=device)
                decoded = torch.empty_like(source)
                output_array = nvcomp.as_array(output, cuda_stream=stream.cuda_stream)
                decoded_array = nvcomp.as_array(decoded, cuda_stream=stream.cuda_stream)
                encoded = codec.encode(
                    source_array,
                    out=output_array,
                    compression_config=compression_configs[u8_bytes],
                )
                stream.synchronize()
                encoded_view = output[: encoded.buffer_size]
                encoded_array = nvcomp.as_array(
                    encoded_view, cuda_stream=stream.cuda_stream
                )
                codec.decode(
                    encoded_array,
                    out=decoded_array,
                    decompression_config=decompression_configs[u8_bytes],
                )
            stream.synchronize()
    finally:
        try:
            stream.synchronize()
        finally:
            unregister_workspace_allocators([stream.cuda_stream])
    del (
        codec,
        stream,
        source,
        output,
        decoded,
        source_array,
        output_array,
        decoded_array,
        encoded,
        encoded_array,
        compression_configs,
        decompression_configs,
        hadamard_input,
        hadamard_output,
    )
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    retained = max(0, free_before - free_after)
    return round_up(audit.peak_bytes, ALIGNMENT), (max(max_outputs),) * 2, retained


def _probe_cpp_backend(device: torch.device, head_size: int) -> int:
    """Warm the shared CUDA module before KV sizing, even without ANS."""
    from .cpp_backend import hadamard_transform

    torch.cuda.synchronize(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    source = output = None
    try:
        with torch.cuda.device(device):
            source = torch.empty((1, head_size), dtype=torch.bfloat16, device=device)
            output = torch.empty_like(source)
            hadamard_transform(source, output)
            torch.cuda.synchronize(device)
    finally:
        del source, output
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    return max(0, free_before - free_after)


def build_memory_plan(
    config: CompressionConfig,
    specs: dict[str, CacheSpec],
    *,
    blocks_per_logical: int = 1,
    tp_size: int = 1,
    tp_rank: int = 0,
    num_layers: int = 0,
    device: torch.device | None = None,
    warm_cpp_backend: bool = False,
) -> CompressionMemoryPlan | None:
    if not config.available:
        return None
    device = device or torch.device("cuda", torch.cuda.current_device())
    nvidia_cuda = torch.device(device).type == "cuda" and torch.version.hip is None
    if warm_cpp_backend and not nvidia_cuda:
        raise ValueError("C++ compression warmup requires an NVIDIA CUDA device")
    transformer = config.uses_transformer
    codec_enabled = config.uses_codec
    if config.is_raw_aggregate:
        logger.info(
            "Mooncake raw_aggregate transport config: slot_count=%d "
            "logical_blocks_per_chunk=%d (lossless gather/scatter, no "
            "transformer/quantizer/codec)",
            config.slot_count,
            config.logical_blocks_per_chunk,
        )
    else:
        logger.info(
            "Mooncake compression config: transformer=%s split_type=%s "
            "axis_key=%s axis_value=%s levels=(K:%d/%d,V:%d/%d) "
            "slot_count=%d logical_blocks_per_chunk=%d codec=%s",
            config.transformer,
            config.split_type,
            config.axis_key,
            config.axis_value,
            config.low_key_num_levels,
            config.high_key_num_levels,
            config.low_value_num_levels,
            config.high_value_num_levels,
            config.slot_count,
            config.logical_blocks_per_chunk,
            config.codec,
        )

    scores = (
        _load_scores(config.score_path)
        if config.uses_quantizer and config.split_type == "head" and config.score_path
        else None
    )
    planning_config = config
    if "quantizer" not in config.pipeline:
        planning_config = replace(config, split_type="layer")
    selection = build_head_selection(specs, planning_config, tp_rank, tp_size, scores)
    num_layers = num_layers or max(selection.low_counts, default=-1) + 1
    layout = build_compression_layout(
        specs,
        planning_config,
        selection.low_counts,
        num_layers,
    )
    workspace_bytes, max_outputs, retained_bytes = (
        _probe_operators(
            layout,
            device,
            layout.packed_bytes_by_part,
            transformer,
        )
        if codec_enabled
        else (0, (0, 0), 0)
    )
    if nvidia_cuda and (
        (transformer and not codec_enabled) or (warm_cpp_backend and not transformer)
    ):
        # ANS probing already warms Hadamard when both stages are enabled.
        # For a C++ pipeline without Hadamard, any supported dimension warms
        # the same module without constraining the model's actual head size.
        head_size = layout.layer_parts[0].head_size if transformer else 128
        retained_bytes += _probe_cpp_backend(device, head_size)
    layout = replace(
        layout,
        arena_bytes=round_up(max(layout.arena_bytes, max(max_outputs)), ALIGNMENT),
    )
    if any(part.block_size % blocks_per_logical for part in layout.layer_parts):
        raise ValueError("Logical attention blocks must contain whole kernel blocks.")
    unique_layers = {part.layer_name: part for part in layout.layer_parts}
    sign_bytes = (
        sum(
            round_up(part.num_heads * part.head_size * 2, ALIGNMENT)
            for part in layout.layer_parts
        )
        if transformer
        else 0
    )
    order_bytes = sum(
        round_up(part.num_heads * 8, ALIGNMENT) for part in unique_layers.values()
    )
    metadata_bytes = (
        sign_bytes
        + order_bytes
        + config.slot_count
        * round_up(
            max(
                len(layout.index_rows_by_group),
                len({part.layer_name for part in layout.layer_parts}),
                1,
            )
            * config.logical_blocks_per_chunk
            * blocks_per_logical
            * 8,
            ALIGNMENT,
        )
    )
    # Each slot owns its writable signs and head-order metadata.
    shared_metadata = sign_bytes + order_bytes
    metadata_bytes += (config.slot_count - 1) * shared_metadata
    if codec_enabled:
        workspace_bytes *= config.slot_count
        retained_bytes *= config.slot_count
    arena_b_bytes = 0 if config.uses_aggregate else layout.arena_bytes
    slot_bytes = (
        layout.arena_bytes
        + arena_b_bytes
        + layout.aux_arena_bytes
        + layout.raw_tail_capacity
    )
    reserved = round_up(
        config.slot_count * slot_bytes
        + workspace_bytes
        + metadata_bytes
        + retained_bytes
        + DRIVER_GUARD_BYTES,
        ALIGNMENT,
    )
    plan = CompressionMemoryPlan(
        config=config,
        layout=layout,
        head_selection=selection,
        codec_workspace_bytes=workspace_bytes,
        codec_max_output_bytes=max_outputs,
        blocks_per_logical=blocks_per_logical,
        metadata_bytes=metadata_bytes,
        retained_driver_bytes=retained_bytes,
        guard_bytes=DRIVER_GUARD_BYTES,
        reserved_bytes=reserved,
    )
    logger.info(
        "Mooncake compression memory plan: pipeline=%s reserved=%d bulk_slots=%d "
        "arena=%d aux_arena=%d codec_workspace=%d metadata=%d "
        "retained_driver=%d blocks_per_logical=%d",
        config.pipeline,
        plan.reserved_bytes,
        config.slot_count,
        layout.arena_bytes,
        layout.aux_arena_bytes,
        workspace_bytes,
        metadata_bytes,
        retained_bytes,
        blocks_per_logical,
    )
    logger.debug(
        "Mooncake compression layout: raw_bytes_by_part=%s "
        "u8_bytes_by_part=%s aux_bytes_by_part=%s max_codec_output=%s "
        "fa_groups=%s",
        layout.raw_bytes_by_part,
        layout.u8_bytes_by_part,
        layout.aux_bytes_by_part,
        max_outputs,
        sorted(layout.fa_group_indices),
    )
    return plan


class _PoolAllocation:
    def __init__(self, allocator: FixedWorkspaceAllocator, offset: int, nbytes: int):
        self._allocator = allocator
        self._offset = offset
        self._nbytes = nbytes
        self.ptr = allocator.buffer.data_ptr() + offset

    def __del__(self):
        self._allocator.release(self._offset, self._nbytes)


class FixedWorkspaceAllocator:
    """Best-fit nvCOMP allocator backed by a fixed CUDA byte tensor."""

    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self._free: list[tuple[int, int]] = [(0, buffer.numel())]
        self._lock = threading.Lock()

    def __call__(self, nbytes: int, stream: Any) -> _PoolAllocation:
        size = round_up(nbytes, ALIGNMENT)
        with self._lock:
            candidates = [
                (length, index, offset)
                for index, (offset, length) in enumerate(self._free)
                if length >= size
            ]
            length, index, offset = min(candidates)
            del self._free[index]
            if length > size:
                self._free.append((offset + size, length - size))
        return _PoolAllocation(self, offset, size)

    def release(self, offset: int, nbytes: int) -> None:
        with self._lock:
            self._free.append((offset, nbytes))
            self._free.sort()
            merged: list[tuple[int, int]] = []
            for current_offset, current_length in self._free:
                if merged and sum(merged[-1]) == current_offset:
                    previous_offset, previous_length = merged[-1]
                    merged[-1] = (
                        previous_offset,
                        previous_length + current_length,
                    )
                else:
                    merged.append((current_offset, current_length))
            self._free = merged
