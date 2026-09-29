# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from .codec import EncodedChunk
from .execution import ExecutionPlan, prepare_execution
from .layout import ALIGNMENT, KVPart
from .memory import FixedWorkspaceAllocator, MooncakeCompressionMemoryPlan
from .pipeline import CompressionPipeline, CompressionSlotBuffers

logger = init_logger(__name__)


class _PoolCarver:
    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self.offset = 0

    def take(self, nbytes: int) -> torch.Tensor:
        self.offset = round_up(self.offset, ALIGNMENT)
        end = self.offset + nbytes
        result = self.buffer[self.offset : end]
        self.offset = end
        return result


def _cap_switch_interval() -> None:
    """Cap the GIL slice when connector threads compete with the engine thread.

    Measured on the cross-host Qwen3-30B TP4 setup: a 1ms switch interval cuts
    the p50 inter-rank NCCL start skew and buys ~2-4% end-to-end throughput.
    """
    sys.setswitchinterval(0.001)


class MooncakeCompressionManager:
    def __init__(
        self,
        plan: MooncakeCompressionMemoryPlan,
        kv_cache_layout: str,
        blocks_per_logical: int,
        tp_size: int,
        pp_size: int,
        device: torch.device,
    ):
        _cap_switch_interval()
        self.plan = plan
        self.config = plan.config
        self.layout = plan.layout
        self.tp_size = tp_size
        self._owns_nvcomp_allocator = False

        self._raw_tail_capacity = self.layout.raw_tail_capacity
        slot_bytes = (
            2 * self.layout.arena_bytes
            + self.layout.aux_arena_bytes
            + self._raw_tail_capacity
        )
        fixed_bytes = (
            self.config.slot_count * slot_bytes
            + plan.codec_workspace_bytes
            + plan.metadata_bytes
        )
        self.bulk = torch.empty(fixed_bytes, dtype=torch.uint8, device=device)
        carver = _PoolCarver(self.bulk)

        max_physical_blocks = self.config.logical_blocks_per_chunk * blocks_per_logical
        arena_b_bytes = 0 if self.config.uses_aggregate else self.layout.arena_bytes
        self.slots: list[CompressionSlotBuffers] = []
        num_layers = len({part.layer_name for part in self.layout.layer_parts})
        for _ in range(self.config.slot_count):
            arena_a = carver.take(self.layout.arena_bytes)
            arena_b = carver.take(arena_b_bytes)
            aux = carver.take(self.layout.aux_arena_bytes)
            raw_tail = carver.take(self._raw_tail_capacity)
            indices_bytes = max_physical_blocks * num_layers * 8
            indices = (
                carver.take(indices_bytes)
                .view(torch.int64)
                .view(num_layers, max_physical_blocks)
            )
            self.slots.append(
                CompressionSlotBuffers(
                    arena_a=arena_a,
                    arena_b=arena_b,
                    aux=aux,
                    block_indices=indices,
                    completion_event=torch.cuda.Event(),
                    host_indices=torch.empty(
                        (num_layers, max_physical_blocks),
                        dtype=torch.int64,
                        pin_memory=True,
                    ),
                    raw_tail=raw_tail,
                )
            )

        # Compression kernels are tiny (gather/FWHT/quant/ANS) but sit on the
        # TTFT critical path; run them on high-priority streams so a saturated
        # prefill stream cannot starve them.
        self.streams = [
            torch.cuda.Stream(device=device, priority=-1) for _ in self.slots
        ]
        self.workspace_allocators: list[FixedWorkspaceAllocator] = []
        if self.config.uses_codec:
            self.workspace_allocators = [
                FixedWorkspaceAllocator(
                    carver.take(plan.codec_workspace_bytes // len(self.slots))
                )
                for _ in self.slots
            ]
            from nvidia import nvcomp

            nvcomp.set_device_allocator(self._allocate_workspace)
            self._owns_nvcomp_allocator = True
        self._workspace_by_stream = dict(
            zip(
                (stream.cuda_stream for stream in self.streams),
                self.workspace_allocators,
            )
        )
        try:
            self.stream = self.streams[0]

            self.head_orders: dict[str, torch.Tensor] = {}
            for part in self.layout.layer_parts:
                if part.layer_name in self.head_orders:
                    continue
                order_values = self.plan.head_selection.local_orders[part.layer_index]
                order = carver.take(len(order_values) * 8).view(torch.int64)
                order.copy_(torch.tensor(order_values, dtype=torch.int64, device="cpu"))
                self.head_orders[part.layer_name] = order

            self.signs: dict[tuple[str, KVPart], torch.Tensor] = {}
            if self.config.uses_transformer:
                for part in self.layout.layer_parts:
                    key = (part.layer_name, part.kv_part)
                    signs = (
                        carver.take(part.num_heads * part.head_size * 2)
                        .view(part.dtype)
                        .view(1, 1, part.num_heads, part.head_size)
                    )
                    order = self.plan.head_selection.local_orders[part.layer_index]
                    cpu_signs = torch.empty(
                        (part.num_heads, part.head_size), dtype=torch.float32
                    )
                    generator = torch.Generator(device="cpu")
                    for output_head, original_head in enumerate(order):
                        generator.manual_seed(
                            self.config.seed ^ (part.layer_index << 16) ^ original_head
                        )
                        row = torch.randint(
                            0,
                            2,
                            (part.head_size,),
                            generator=generator,
                            dtype=torch.int8,
                        )
                        cpu_signs[output_head].copy_(row).mul_(2).sub_(1)
                    signs.copy_(cpu_signs.to(dtype=part.dtype))
                    self.signs[key] = signs

            from .codec import ANSCodec

            codec = (
                ANSCodec(self.stream, self.layout.packed_bytes_by_part)
                if self.config.uses_codec
                else None
            )
            self.pipelines: list[CompressionPipeline] = [
                CompressionPipeline(
                    layout=self.layout,
                    kv_cache_layout=kv_cache_layout,
                    blocks_per_logical=blocks_per_logical,
                    slots=self.slots,
                    codec=codec,
                    stream=self.stream,
                    signs=self.signs,
                    head_orders=self.head_orders,
                    stages=self.config.pipeline,
                )
            ]
            self.slot_plans: list[ExecutionPlan | None] = [None] * len(self.slots)
            for index in range(1, len(self.slots)):
                signs, orders = {}, {}
                for part in self.layout.layer_parts:
                    if part.layer_name not in orders:
                        orders[part.layer_name] = carver.take(part.num_heads * 8).view(
                            torch.int64
                        )
                    if self.config.uses_transformer:
                        signs[(part.layer_name, part.kv_part)] = (
                            carver.take(part.num_heads * part.head_size * 2)
                            .view(part.dtype)
                            .view(1, 1, part.num_heads, part.head_size)
                        )
                codec = (
                    ANSCodec(
                        self.streams[index],
                        self.layout.packed_bytes_by_part,
                    )
                    if self.config.uses_codec
                    else None
                )
                self.pipelines.append(
                    CompressionPipeline(
                        layout=self.layout,
                        kv_cache_layout=kv_cache_layout,
                        blocks_per_logical=blocks_per_logical,
                        slots=self.slots,
                        codec=codec,
                        stream=self.streams[index],
                        signs=signs,
                        head_orders=orders,
                        stages=self.config.pipeline,
                    )
                )
            self._init_event = torch.cuda.Event()
            self._init_event.record(torch.cuda.current_stream(device))
            for stream in self.streams:
                stream.wait_event(self._init_event)
        except BaseException:
            nvcomp.set_device_allocator()
            self._owns_nvcomp_allocator = False
            raise
        logger.info(
            "Mooncake compression manager initialized: layout=%s pipeline=%s "
            "tp=%d pp=%d slots=%d bulk_bytes=%d arena_bytes=%d aux_arena_bytes=%d "
            "workspace_bytes=%d blocks_per_logical=%d",
            kv_cache_layout,
            self.config.pipeline,
            tp_size,
            pp_size,
            self.config.slot_count,
            fixed_bytes,
            self.layout.arena_bytes,
            self.layout.aux_arena_bytes,
            plan.codec_workspace_bytes,
            blocks_per_logical,
        )

    @property
    def fa_group_indices(self) -> frozenset[int]:
        return self.layout.fa_group_indices

    def _allocate_workspace(self, nbytes, stream):
        allocator = self._workspace_by_stream[stream.ptr]
        return allocator(nbytes, stream)

    def prepare(self) -> ExecutionPlan:
        return prepare_execution(self.plan, self.config, self.layout, self.tp_size)

    def _execution_pipeline(self, slot_id: int, execution: ExecutionPlan | None):
        pipeline = self.pipelines[slot_id]
        if execution is None:
            execution = self.default_execution
        previous = self.slot_plans[slot_id]
        if previous is not execution:
            with torch.cuda.stream(self.streams[slot_id]):
                for name, order in execution.orders.items():
                    pipeline.head_orders[name].copy_(order, non_blocking=True)
                for key, signs in execution.signs.items():
                    pipeline.signs[key].copy_(signs, non_blocking=True)
            pipeline.layout = execution.layout
            pipeline.stages = execution.config.pipeline
            pipeline.identity_orders = set(execution.identity_layers)
            pipeline._cpp_metas = None
            self.slot_plans[slot_id] = execution
        return pipeline

    def bind_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        group_indices: dict[str, int],
    ) -> None:
        layer_parts = tuple(
            replace(part, group_index=group_indices[part.layer_name])
            for part in self.layout.layer_parts
        )
        self.layout = replace(
            self.layout,
            layer_parts=layer_parts,
            fa_group_indices=frozenset(
                group_indices[part.layer_name] for part in layer_parts
            ),
        )
        for pipeline in self.pipelines:
            pipeline.layout = self.layout
            pipeline.bind_kv_caches(kv_caches)
        self.default_execution = self.prepare()
        logger.info(
            "Mooncake compression KV caches bound: layers=%d fa_groups=%s",
            len(self.layout.layer_parts) // 2,
            sorted(self.layout.fa_group_indices),
        )
        logger.debug(
            "Mooncake compression cache tensors: %s",
            {
                name: {
                    "shape": tuple(cache.shape),
                    "stride": tuple(cache.stride()),
                    "data_ptr": cache.data_ptr(),
                }
                for name, cache in kv_caches.items()
                if name in {part.layer_name for part in self.layout.layer_parts}
            },
        )

    def registered_memory(self) -> tuple[list[int], list[int]]:
        storage = self.bulk.untyped_storage()
        return [storage.data_ptr()], [storage.nbytes()]

    def slot_addresses(self, slot_id: int) -> tuple[int, int]:
        slot = self.slots[slot_id]
        return slot.arena_a.data_ptr(), slot.aux.data_ptr()

    def tail_slot_addresses(self, slot_id: int) -> int:
        slot = self.slots[slot_id]
        assert slot.raw_tail is not None
        return slot.raw_tail.data_ptr()

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        execution: ExecutionPlan | None = None,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> EncodedChunk:
        return self._execution_pipeline(slot_id, execution).encode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            valid_tokens_by_group=valid_tokens_by_group,
        )

    def decode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        codec_bytes: int,
        u8_bytes: int,
        execution: ExecutionPlan | None = None,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> torch.cuda.Event:
        return self._execution_pipeline(slot_id, execution).decode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            codec_bytes,
            u8_bytes,
            valid_tokens_by_group=valid_tokens_by_group,
        )

    def shutdown(self) -> None:
        if not self._owns_nvcomp_allocator:
            return
        from nvidia import nvcomp

        for stream in self.streams:
            stream.synchronize()
        self.pipelines.clear()
        nvcomp.set_device_allocator()
        self._owns_nvcomp_allocator = False
