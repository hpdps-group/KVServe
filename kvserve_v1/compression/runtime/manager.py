# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from logging import getLogger as init_logger

import torch

from .codec import EncodedChunk
from .layout import (
    ALIGNMENT,
    KVPart,
    LayerPartLayout,
)
from .memory import (
    CompressionMemoryPlan,
    FixedWorkspaceAllocator,
    register_workspace_allocators,
    unregister_workspace_allocators,
)
from .pipeline import CompressionPipeline, CompressionSlotBuffers
from .spec import round_up

logger = init_logger(__name__)


def build_signs(
    part: LayerPartLayout,
    order: tuple[int, ...],
    seed: int,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    cpu = torch.empty(
        (part.num_heads, part.head_size), dtype=part.dtype, pin_memory=pin_memory
    )
    generator = torch.Generator(device="cpu")
    for output_head, original_head in enumerate(order):
        generator.manual_seed(seed ^ (part.layer_index << 16) ^ original_head)
        cpu[output_head].copy_(
            torch.randint(
                0, 2, (part.head_size,), generator=generator, dtype=torch.int8
            )
        ).mul_(2).sub_(1)
    return cpu.view(1, 1, part.num_heads, part.head_size)


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


class CompressionRuntime:
    def __init__(
        self,
        plan: CompressionMemoryPlan,
        kv_cache_layout: str,
        blocks_per_logical: int,
        device: torch.device,
        *,
        codec_factory=None,
        external_execution=False,
        execution="both",
    ):
        if execution not in ("both", "encode", "decode"):
            raise ValueError("Unknown KVServe execution capability")
        self.execution = execution
        self.plan = plan
        self.config = plan.config
        self.layout = plan.layout
        self.output_slots = []
        self._output_slots_allocated = False
        self._external_execution = external_execution
        self._metadata_initialized = False
        self._owns_nvcomp_allocator = False

        self._raw_tail_capacity = self.layout.raw_tail_capacity
        arena_b_bytes = 0 if self.config.uses_aggregate else self.layout.arena_bytes
        slot_bytes = (
            self.layout.arena_bytes
            + arena_b_bytes
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
        self.slots: list[CompressionSlotBuffers] = []
        num_index_rows = max(
            len(self.layout.index_rows_by_group),
            len({part.layer_name for part in self.layout.layer_parts}),
            1,
        )
        for _ in range(self.config.slot_count):
            arena_a = carver.take(self.layout.arena_bytes)
            arena_b = carver.take(arena_b_bytes)
            aux = carver.take(self.layout.aux_arena_bytes)
            raw_tail = carver.take(self._raw_tail_capacity)
            indices_bytes = max_physical_blocks * num_index_rows * 8
            indices = (
                carver.take(indices_bytes)
                .view(torch.int64)
                .view(num_index_rows, max_physical_blocks)
            )
            self.slots.append(
                CompressionSlotBuffers(
                    arena_a=arena_a,
                    arena_b=arena_b,
                    aux=aux,
                    block_indices=indices,
                    completion_event=torch.cuda.Event(),
                    host_indices=torch.empty(
                        (num_index_rows, max_physical_blocks),
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
        self._workspace_by_stream = dict(
            zip(
                (stream.cuda_stream for stream in self.streams),
                self.workspace_allocators,
            )
        )
        if self.config.uses_codec:
            register_workspace_allocators(self._workspace_by_stream)
            self._owns_nvcomp_allocator = True
        try:
            self.stream = self.streams[0]

            from .codec import ANSCodec

            self.pipelines: list[CompressionPipeline] = []
            for slot_index, stream in enumerate(self.streams):
                orders: dict[str, torch.Tensor] = {}
                signs: dict[tuple[str, KVPart], torch.Tensor] = {}
                for part in self.layout.layer_parts:
                    if part.layer_name not in orders:
                        order = carver.take(part.num_heads * 8).view(torch.int64)
                        orders[part.layer_name] = order
                    if not self.config.uses_transformer:
                        continue
                    key = (part.layer_name, part.kv_part)
                    signs[key] = (
                        carver.take(part.num_heads * part.head_size * 2)
                        .view(part.dtype)
                        .view(1, 1, part.num_heads, part.head_size)
                    )
                codec = None
                if self.config.uses_codec:
                    if codec_factory is None:
                        codec = ANSCodec(stream, self.layout.packed_bytes_by_part)
                    else:
                        codec = codec_factory(
                            stream, self.layout.packed_bytes_by_part,
                            self.workspace_allocators[slot_index].buffer,
                        )
                self.pipelines.append(
                    CompressionPipeline(
                        layout=self.layout,
                        kv_cache_layout=kv_cache_layout,
                        blocks_per_logical=blocks_per_logical,
                        slots=self.slots,
                        codec=codec,
                        stream=stream,
                        signs=signs,
                        head_orders=orders,
                        stages=self.config.pipeline,
                    )
                )
            self.head_orders = self.pipelines[0].head_orders
            self.signs = self.pipelines[0].signs
            self._init_event = torch.cuda.Event()
            self._init_event.record(torch.cuda.current_stream(device))
            for stream in self.streams:
                stream.wait_event(self._init_event)
        except BaseException:
            self._release_workspace_allocator()
            raise
        logger.info(
            "Mooncake compression manager initialized: layout=%s pipeline=%s "
            "slots=%d bulk_bytes=%d arena_bytes=%d aux_arena_bytes=%d "
            "workspace_bytes=%d blocks_per_logical=%d",
            kv_cache_layout,
            self.config.pipeline,
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

    def _release_workspace_allocator(self):
        if self._owns_nvcomp_allocator:
            unregister_workspace_allocators(self._workspace_by_stream)
            self._owns_nvcomp_allocator = False

    def bind_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
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
            index_rows_by_group={
                group_index: row
                for row, group_index in enumerate(
                    sorted({part.group_index for part in layer_parts})
                )
            },
        )
        for pipeline in self.pipelines:
            pipeline.layout = self.layout
            if self._external_execution:
                pipeline.bind_kv_caches(kv_caches, prepare_backend=False)
            else:
                pipeline.bind_kv_caches(kv_caches)
        if not self._metadata_initialized:
            orders = {
                name: torch.tensor(
                    self.plan.head_selection.local_orders[part.layer_index],
                    dtype=torch.int64,
                ).pin_memory()
                for name, part in {p.layer_name: p for p in layer_parts}.items()
            }
            signs = {
                (part.layer_name, part.kv_part): build_signs(
                    part,
                    self.plan.head_selection.local_orders[part.layer_index],
                    self.config.seed,
                    pin_memory=True,
                )
                for part in layer_parts
                if self.config.uses_transformer
            }
            identity = {
                name
                for name, order in orders.items()
                if tuple(order.tolist()) == tuple(range(order.numel()))
            }
            for pipeline in self.pipelines:
                with torch.cuda.stream(pipeline.stream):
                    for name, order in orders.items():
                        pipeline.head_orders[name].copy_(order, non_blocking=True)
                    for key, sign in signs.items():
                        pipeline.signs[key].copy_(sign, non_blocking=True)
                pipeline.identity_orders = identity
            self._metadata_initialized = True
        config = asdict(self.config)
        for key in ("score_path", "slot_count"):
            config.pop(key, None)
        layout_description = asdict(self.layout)
        layout_description["fa_group_indices"] = sorted(self.layout.fa_group_indices)
        wire_layout = {
            "version": 2,
            "config": config,
            "layout": layout_description,
            "cache_layout": self.pipelines[0].kv_cache_layout,
            "blocks_per_logical": self.pipelines[0].blocks_per_logical,
            "orders": {
                part.layer_name: self.plan.head_selection.local_orders[part.layer_index]
                for part in layer_parts
            },
        }
        self.wire_profile = hashlib.sha256(
            json.dumps(wire_layout, sort_keys=True, default=str).encode()
        ).hexdigest()
        logger.info(
            "Mooncake compression KV caches bound: layers=%d fa_groups=%s",
            len(self.layout.layer_parts) // 2,
            sorted(self.layout.fa_group_indices),
        )
        logger.debug(
            "Mooncake compression cache tensors: %s",
            {
                name: {
                    "shape": tuple(cache[0].shape)
                    if isinstance(cache, tuple)
                    else tuple(cache.shape),
                    "stride": tuple(cache[0].stride())
                    if isinstance(cache, tuple)
                    else tuple(cache.stride()),
                    "data_ptr": [x.data_ptr() for x in cache]
                    if isinstance(cache, tuple)
                    else cache.data_ptr(),
                }
                for name, cache in kv_caches.items()
                if name in {part.layer_name for part in self.layout.layer_parts}
            },
        )

    def data_bytes(self, kv_part: int, tokens: dict[int, int] | None = None) -> int:
        if self.config.uses_aggregate:
            if tokens is None:
                return self.layout.raw_bytes_by_part[kv_part]
            total = 0
            for part in self.layout.parts(kv_part):
                count = tokens.get(part.group_index, 0)
                blocks = (count + part.block_size - 1) // part.block_size
                total += blocks * part.num_heads * part.block_size * part.head_size * 2
            return total
        if tokens is not None and not self.layout.has_body(tokens):
            return 0
        lengths = (
            self.layout.u8_bytes_by_part
            if "quantizer" in self.config.pipeline
            else self.layout.raw_bytes_by_part
        )
        return lengths[kv_part]

    def aux_bytes(self, kv_part: int, tokens: dict[int, int] | None = None) -> int:
        if tokens is not None and not self.layout.has_body(tokens):
            return 0
        return (
            self.layout.aux_bytes_by_part[kv_part]
            if "quantizer" in self.config.pipeline
            else 0
        )

    def registered_memory(self) -> tuple[list[int], list[int]]:
        storage = self.bulk.untyped_storage()
        pointers, sizes = [storage.data_ptr()], [storage.nbytes()]
        for payload, auxiliary in self.output_slots:
            for tensor in (payload, auxiliary):
                pointers.append(tensor.data_ptr())
                sizes.append(tensor.numel() * tensor.element_size())
        return pointers, sizes

    def allocate_output_slots(self, count: int) -> None:
        """Reserve connector-independent encoded output leases before KV sizing."""
        if type(count) is not int or not 0 <= count <= 16:
            raise ValueError("Encoded output slot count must be 0..16")
        if self._output_slots_allocated or self._metadata_initialized:
            raise RuntimeError("Encoded output banks must be reserved once before binding")
        with torch.cuda.device(self.bulk.device):
            self.output_slots = [
                (torch.empty_like(self.slots[0].arena_a),
                 torch.empty_like(self.slots[0].aux)) for _ in range(count)
            ]
        extra = sum(t.numel() * t.element_size() for bank in self.output_slots for t in bank)
        self.plan = replace(self.plan, reserved_bytes=self.plan.reserved_bytes + extra)
        self._output_slots_allocated = True

    def _validate_output_bank(self, slot_id, output, auxiliary):
        """Accept only the slot's own buffers or one intact output lease.

        The caller owns lease scheduling and must wait for completion before
        reusing a bank. Restricting the pair prevents aliases into active
        compression buffers and avoids copying auxiliary data over payloads.
        """
        if type(slot_id) is not int or not 0 <= slot_id < len(self.slots):
            raise ValueError("Invalid compression slot")
        slot = self.slots[slot_id]
        pairs = [(slot.arena_a, slot.aux), *self.output_slots]
        if not any(output is payload and auxiliary is aux for payload, aux in pairs):
            raise ValueError("Encoded output and auxiliary must be one Runtime-owned bank")
        return slot

    def _require_execution(self, operation):
        if self.execution not in ("both", operation):
            raise RuntimeError(f"KVServe {self.execution}-only runtime cannot {operation}")

    def encode_into(self, slot_id, logical_ids_by_group, kv_part, *,
                    valid_tokens_by_group, output, auxiliary) -> EncodedChunk:
        """Standard compressed execution with a completion-fenced output copy.

        This fallback may wait on the host and belongs on a caller's codec
        worker. The fused subclass writes directly into the leased output.
        """
        self._require_execution("encode")
        slot = self._validate_output_bank(slot_id, output, auxiliary)
        stream = self.streams[slot_id]
        aux_bytes = self.aux_bytes(kv_part, valid_tokens_by_group)
        for tensor, capacity in ((output, slot.arena_a.numel()), (auxiliary, aux_bytes)):
            if (tensor.device != slot.arena_a.device or tensor.dtype != torch.uint8
                    or not tensor.is_contiguous() or tensor.numel() < capacity):
                raise ValueError("Invalid encoded output or auxiliary bank")
        encoded = self.encode(slot_id, logical_ids_by_group, kv_part,
                              valid_tokens_by_group=valid_tokens_by_group)
        encoded.event.synchronize()
        if encoded.raw_tail_bytes:
            raise ValueError("Leased output requires a complete compressed token body")
        size = encoded.codec_bytes
        if not 0 < size <= output.numel():
            raise ValueError("Encoded payload exceeds output bank")
        with torch.cuda.stream(stream):
            output[:size].copy_(slot.arena_a[:size], non_blocking=True)
            auxiliary[:aux_bytes].copy_(slot.aux[:aux_bytes], non_blocking=True)
            slot.completion_event.record(stream)
        return EncodedChunk(encoded.array, slot.completion_event, encoded.u8_bytes,
                            payload_bytes=size)

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> EncodedChunk:
        self._require_execution("encode")
        return self.pipelines[slot_id].encode(
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
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> torch.cuda.Event:
        self._require_execution("decode")
        expected_bytes = self.data_bytes(kv_part, valid_tokens_by_group)
        if u8_bytes != expected_bytes:
            raise ValueError("Compressed chunk has an incompatible decoded size")
        if not 0 <= codec_bytes <= self.slots[slot_id].arena_a.numel():
            raise ValueError("Compressed chunk exceeds receive buffer capacity")
        if "codec" not in self.config.pipeline and codec_bytes != u8_bytes:
            raise ValueError("Unencoded chunk payload size does not match its layout")
        return self.pipelines[slot_id].decode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            codec_bytes,
            u8_bytes,
            valid_tokens_by_group=valid_tokens_by_group,
        )

    def shutdown(self) -> None:
        for stream in self.streams:
            stream.synchronize()
        self.pipelines.clear()
        self.output_slots.clear()
        self._release_workspace_allocator()
