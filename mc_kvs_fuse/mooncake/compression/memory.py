# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheSpec

from .config import MooncakeCompressionConfig
from .layout import ALIGNMENT, CompressionLayout, build_compression_layout
from .scores import HeadSelection, build_head_selection

DRIVER_GUARD_BYTES = 4 << 20
logger = init_logger(__name__)


@dataclass(frozen=True)
class MooncakeCompressionMemoryPlan:
    config: MooncakeCompressionConfig
    layout: CompressionLayout
    head_selection: HeadSelection
    codec_workspace_bytes: int
    codec_max_output_bytes: tuple[int, int]
    physical_blocks_per_logical: int
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
) -> tuple[int, tuple[int, int], int]:
    from humming.ops import hadamard_transform
    from nvidia import nvcomp

    torch.cuda.synchronize(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    audit = _AuditAllocator(device)
    nvcomp.set_device_allocator(audit)
    max_outputs: list[int] = []
    codec = stream = source = output = decoded = None
    source_array = output_array = decoded_array = encoded = encoded_array = None
    compression_configs = decompression_configs = None
    try:
        head_size = layout.layer_parts[0].head_size
        hadamard_input = torch.empty(
            (1, head_size), dtype=torch.bfloat16, device=device
        )
        hadamard_output = torch.empty_like(hadamard_input)
        hadamard_transform(
            hadamard_input,
            head_size,
            scale=1.0,
            outputs=hadamard_output,
        )

        stream = torch.cuda.Stream(device=device)
        codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
        compression_configs = {
            size: codec.compression_config(size)
            for size in set(layout.u8_bytes_by_part)
        }
        decompression_configs = {
            size: codec.decompression_config(config)
            for size, config in compression_configs.items()
        }
        with torch.cuda.stream(stream):
            for u8_bytes in layout.u8_bytes_by_part:
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
        nvcomp.set_device_allocator()
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
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    retained = max(0, free_before - free_after)
    return round_up(audit.peak_bytes, ALIGNMENT), tuple(max_outputs), retained


def build_memory_plan(
    vllm_config: Any,
    specs: dict[str, KVCacheSpec],
) -> MooncakeCompressionMemoryPlan | None:
    assert vllm_config.kv_transfer_config is not None
    config = MooncakeCompressionConfig.from_extra_config(
        vllm_config.kv_transfer_config.kv_connector_extra_config
    )
    if not config.enabled:
        logger.info("Mooncake compression is disabled by connector config.")
        return None
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
    if not torch.cuda.is_available():
        raise ValueError("Mooncake compression requires CUDA.")

    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    tp_rank = get_tensor_model_parallel_rank()
    selection = build_head_selection(specs, config, tp_rank, tp_size)
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    num_layers = int(
        getattr(hf_config, "num_hidden_layers", 0)
        or max(selection.low_counts, default=-1) + 1
    )
    layout = build_compression_layout(
        specs,
        config,
        selection.low_counts,
        num_layers,
    )
    workspace_bytes, max_outputs, retained_bytes = _probe_operators(
        layout, torch.device("cuda")
    )
    if max(max_outputs) > layout.arena_bytes:
        raise ValueError(
            "nvCOMP's required fixed output capacity exceeds one compression "
            f"arena: max_output={max(max_outputs)}, arena={layout.arena_bytes}."
        )
    logical_block_sizes = {part.block_size for part in layout.layer_parts}
    if len(logical_block_sizes) != 1:
        raise ValueError(
            "Compression requires one logical block size across Full Attention "
            f"layers, got {sorted(logical_block_sizes)}."
        )
    logical_block_size = logical_block_sizes.pop()
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        get_current_attn_backends,
    )
    from vllm.v1.worker.utils import select_common_block_size

    kernel_block_size = select_common_block_size(
        vllm_config.cache_config.block_size,
        get_current_attn_backends(vllm_config),
    )
    if logical_block_size % kernel_block_size:
        raise ValueError(
            f"Compression logical block size {logical_block_size} is not "
            f"divisible by kernel block size {kernel_block_size}."
        )
    physical_blocks_per_logical = logical_block_size // kernel_block_size
    unique_layers = {part.layer_name: part for part in layout.layer_parts}
    metadata_bytes = (
        sum(
            round_up(part.num_heads * part.head_size * 2, ALIGNMENT)
            for part in layout.layer_parts
        )
        + sum(
            round_up(part.num_heads * 8, ALIGNMENT) for part in unique_layers.values()
        )
        + config.slot_count
        * round_up(
            config.logical_blocks_per_chunk * physical_blocks_per_logical * 8,
            ALIGNMENT,
        )
    )
    slot_bytes = 2 * layout.arena_bytes + layout.aux_arena_bytes
    reserved = round_up(
        config.slot_count * slot_bytes
        + workspace_bytes
        + metadata_bytes
        + retained_bytes
        + DRIVER_GUARD_BYTES,
        ALIGNMENT,
    )
    plan = MooncakeCompressionMemoryPlan(
        config=config,
        layout=layout,
        head_selection=selection,
        codec_workspace_bytes=workspace_bytes,
        codec_max_output_bytes=max_outputs,
        physical_blocks_per_logical=physical_blocks_per_logical,
        metadata_bytes=metadata_bytes,
        retained_driver_bytes=retained_bytes,
        guard_bytes=DRIVER_GUARD_BYTES,
        reserved_bytes=reserved,
    )
    logger.info(
        "Mooncake compression memory plan: reserved=%d bulk_slots=%d "
        "arena=%d aux_arena=%d codec_workspace=%d metadata=%d "
        "retained_driver=%d physical_blocks_per_logical=%d",
        plan.reserved_bytes,
        config.slot_count,
        layout.arena_bytes,
        layout.aux_arena_bytes,
        workspace_bytes,
        metadata_bytes,
        retained_bytes,
        physical_blocks_per_logical,
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
            if not candidates:
                raise MemoryError(
                    f"nvCOMP workspace exhausted: requested={nbytes}, "
                    f"capacity={self.buffer.numel()}."
                )
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
