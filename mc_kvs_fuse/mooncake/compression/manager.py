# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from .codec import EncodedChunk
from .layout import ALIGNMENT, KVPart
from .memory import FixedWorkspaceAllocator, MooncakeCompressionMemoryPlan
from .pipeline import CompressionPipeline, CompressionSlotBuffers
from .protocol import CompressionHandshake

logger = init_logger(__name__)


class _PoolCarver:
    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self.offset = 0

    def take(self, nbytes: int) -> torch.Tensor:
        self.offset = round_up(self.offset, ALIGNMENT)
        end = self.offset + nbytes
        if end > self.buffer.numel():
            raise MemoryError(
                f"Compression fixed pool exceeded: end={end}, "
                f"capacity={self.buffer.numel()}."
            )
        result = self.buffer[self.offset : end]
        self.offset = end
        return result


class MooncakeCompressionManager:
    PROTOCOL_VERSION = 2

    def __init__(
        self,
        plan: MooncakeCompressionMemoryPlan,
        kv_cache_layout: str,
        blocks_per_logical: int,
        tp_size: int,
        pp_size: int,
        device: torch.device,
    ):
        if device.type != "cuda":
            raise ValueError("Mooncake compression requires a CUDA device.")
        self.plan = plan
        self.config = plan.config
        self.layout = plan.layout
        self.kv_cache_layout = kv_cache_layout
        self.blocks_per_logical = blocks_per_logical
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.device = device
        self.fatal_error: BaseException | None = None
        self._owns_nvcomp_allocator = False
        if blocks_per_logical != plan.physical_blocks_per_logical:
            raise ValueError(
                "Compression logical-to-kernel block mapping changed between "
                f"memory planning and worker initialization: planned="
                f"{plan.physical_blocks_per_logical}, actual="
                f"{blocks_per_logical}."
            )

        slot_bytes = 2 * self.layout.arena_bytes + self.layout.aux_arena_bytes
        fixed_bytes = (
            self.config.slot_count * slot_bytes
            + plan.codec_workspace_bytes
            + plan.metadata_bytes
        )
        self.bulk = torch.empty(fixed_bytes, dtype=torch.uint8, device=device)
        if self.bulk.data_ptr() % ALIGNMENT:
            raise RuntimeError("Compression fixed pool is not 256-byte aligned.")
        carver = _PoolCarver(self.bulk)

        max_physical_blocks = self.config.logical_blocks_per_chunk * blocks_per_logical
        self.slots: list[CompressionSlotBuffers] = []
        for _ in range(self.config.slot_count):
            arena_a = carver.take(self.layout.arena_bytes)
            arena_b = carver.take(self.layout.arena_bytes)
            aux = carver.take(self.layout.aux_arena_bytes)
            indices_bytes = max_physical_blocks * 8
            indices = carver.take(indices_bytes).view(torch.int64)
            self.slots.append(
                CompressionSlotBuffers(
                    arena_a=arena_a,
                    arena_b=arena_b,
                    aux=aux,
                    block_indices=indices,
                    completion_event=torch.cuda.Event(),
                )
            )

        workspace = carver.take(plan.codec_workspace_bytes)
        self.workspace_allocator = FixedWorkspaceAllocator(workspace)
        from nvidia import nvcomp

        nvcomp.set_device_allocator(self.workspace_allocator)
        self._owns_nvcomp_allocator = True
        try:
            self.stream = torch.cuda.Stream(device=device)

            self.head_orders: dict[str, torch.Tensor] = {}
            for part in self.layout.layer_parts:
                if part.layer_name in self.head_orders:
                    continue
                order_values = self.plan.head_selection.local_orders[part.layer_index]
                order = carver.take(len(order_values) * 8).view(torch.int64)
                order.copy_(torch.tensor(order_values, dtype=torch.int64, device="cpu"))
                self.head_orders[part.layer_name] = order

            self.signs: dict[tuple[str, KVPart], torch.Tensor] = {}
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

            self.codec = ANSCodec(self.stream, self.layout.u8_bytes_by_part)
            self.pipeline = CompressionPipeline(
                layout=self.layout,
                kv_cache_layout=kv_cache_layout,
                blocks_per_logical=blocks_per_logical,
                slots=self.slots,
                codec=self.codec,
                stream=self.stream,
                signs=self.signs,
                head_orders=self.head_orders,
            )
        except BaseException:
            nvcomp.set_device_allocator()
            self._owns_nvcomp_allocator = False
            raise
        self._fingerprint = ""
        logger.info(
            "Mooncake compression manager initialized: layout=%s tp=%d pp=%d "
            "slots=%d bulk_bytes=%d arena_bytes=%d aux_arena_bytes=%d "
            "workspace_bytes=%d blocks_per_logical=%d",
            kv_cache_layout,
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
        self.pipeline.layout = self.layout
        self.pipeline.bind_kv_caches(kv_caches)
        self._fingerprint = self._compute_fingerprint()
        logger.info(
            "Mooncake compression KV caches bound: layers=%d fa_groups=%s "
            "fingerprint=%s",
            len(self.layout.layer_parts) // 2,
            sorted(self.layout.fa_group_indices),
            self._fingerprint,
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

    def _compute_fingerprint(self) -> str:
        config_values = self.config.fingerprint_values()
        config_values.pop("score_path", None)
        payload = {
            "protocol": self.PROTOCOL_VERSION,
            "config": config_values,
            "layout": self.kv_cache_layout,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "blocks_per_logical": self.blocks_per_logical,
            "layers": [
                {
                    "name": part.layer_name,
                    "index": part.layer_index,
                    "part": part.kv_part,
                    "group": part.group_index,
                    "block_size": part.block_size,
                    "heads": part.num_heads,
                    "head_size": part.head_size,
                    "dtype": str(part.dtype),
                    "order": self.plan.head_selection.local_orders[part.layer_index],
                }
                for part in self.layout.layer_parts
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def handshake(self) -> CompressionHandshake:
        if not self._fingerprint:
            raise RuntimeError("Compression KV caches have not been registered.")
        return CompressionHandshake(
            protocol_version=self.PROTOCOL_VERSION,
            fingerprint=self._fingerprint,
            kv_cache_layout=self.kv_cache_layout,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            logical_blocks_per_chunk=self.config.logical_blocks_per_chunk,
            slot_count=self.config.slot_count,
            payload_capacity=self.layout.arena_bytes,
            aux_capacity=self.layout.aux_arena_bytes // 2,
        )

    def validate_handshake(self, remote: CompressionHandshake) -> None:
        local = self.handshake()
        logger.debug(
            "Mooncake compression handshake: local_fingerprint=%s "
            "remote_fingerprint=%s local_payload=%d remote_payload=%d "
            "local_aux=%d remote_aux=%d",
            local.fingerprint,
            remote.fingerprint,
            local.payload_capacity,
            remote.payload_capacity,
            local.aux_capacity,
            remote.aux_capacity,
        )
        if remote != local:
            raise ValueError(
                "Mooncake compression handshake mismatch: "
                f"local={local}, remote={remote}."
            )

    def registered_memory(self) -> tuple[list[int], list[int]]:
        storage = self.bulk.untyped_storage()
        return [storage.data_ptr()], [storage.nbytes()]

    def slot_addresses(self, slot_id: int) -> tuple[int, int, int, int]:
        slot = self.slots[slot_id]
        return (
            slot.arena_a.data_ptr(),
            slot.arena_a.numel(),
            slot.aux.data_ptr(),
            self.layout.aux_arena_bytes // 2,
        )

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
    ) -> EncodedChunk:
        try:
            return self.pipeline.encode(slot_id, logical_ids_by_group, kv_part)
        except BaseException as exc:
            self.fatal_error = exc
            raise

    def decode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        codec_bytes: int,
        u8_bytes: int,
    ) -> torch.cuda.Event:
        try:
            return self.pipeline.decode(
                slot_id,
                logical_ids_by_group,
                kv_part,
                codec_bytes,
                u8_bytes,
            )
        except BaseException as exc:
            self.fatal_error = exc
            raise

    def raise_if_failed(self) -> None:
        if self.fatal_error is not None:
            raise RuntimeError(
                "Mooncake compression entered FAILED state."
            ) from self.fatal_error

    def shutdown(self) -> None:
        if not self._owns_nvcomp_allocator:
            return
        from nvidia import nvcomp

        nvcomp.set_device_allocator()
        self._owns_nvcomp_allocator = False
