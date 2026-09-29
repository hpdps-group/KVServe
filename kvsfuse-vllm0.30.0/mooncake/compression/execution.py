# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import MooncakeCompressionConfig
from .layout import CompressionLayout, KVPart, build_compression_layout
from .memory import MooncakeCompressionMemoryPlan
from .scores import build_head_selection


@dataclass(frozen=True)
class ExecutionPlan:
    config: MooncakeCompressionConfig
    layout: CompressionLayout
    signs: dict[tuple[str, KVPart], torch.Tensor]
    orders: dict[str, torch.Tensor]
    identity_layers: frozenset[str]

    @property
    def raw(self) -> bool:
        return not self.config.enabled or not self.config.pipeline

    @property
    def aggregate(self) -> bool:
        return "aggregate" in self.config.pipeline

    def data_bytes(self, kv_part: int, tokens: dict[int, int] | None = None) -> int:
        if self.aggregate:
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


def prepare_execution(
    memory: MooncakeCompressionMemoryPlan,
    config: MooncakeCompressionConfig,
    bound_layout: CompressionLayout,
    tp_size: int,
) -> ExecutionPlan:
    effective = config
    if not config.enabled or not config.pipeline:
        effective = replace(config, pipeline=())
    if "quantizer" not in effective.pipeline:
        effective = replace(effective, split_type="layer")
    if memory.specs:
        selection = build_head_selection(
            memory.specs, effective, memory.tp_rank, tp_size, memory.scores
        )
        num_layers = (
            memory.num_layers
            or max(part.layer_index for part in bound_layout.layer_parts) + 1
        )
        layout = build_compression_layout(
            memory.specs, effective, selection.low_counts, num_layers
        )
        groups = {
            part.layer_name: part.group_index for part in bound_layout.layer_parts
        }
        layout = replace(
            layout,
            layer_parts=tuple(
                replace(part, group_index=groups[part.layer_name])
                for part in layout.layer_parts
            ),
            fa_group_indices=bound_layout.fa_group_indices,
            arena_bytes=bound_layout.arena_bytes,
            aux_arena_bytes=bound_layout.aux_arena_bytes,
        )
    else:
        selection, layout = memory.head_selection, bound_layout
    signs, orders = {}, {}
    for part in layout.layer_parts:
        order = selection.local_orders[part.layer_index]
        if part.layer_name not in orders:
            orders[part.layer_name] = torch.tensor(
                order, dtype=torch.int64
            ).pin_memory()
        if not effective.uses_transformer:
            continue
        cpu = torch.empty(
            (part.num_heads, part.head_size), dtype=part.dtype, pin_memory=True
        )
        generator = torch.Generator(device="cpu")
        for output_head, original_head in enumerate(order):
            generator.manual_seed(
                effective.seed ^ (part.layer_index << 16) ^ original_head
            )
            cpu[output_head].copy_(
                torch.randint(
                    0, 2, (part.head_size,), generator=generator, dtype=torch.int8
                )
            ).mul_(2).sub_(1)
        signs[(part.layer_name, part.kv_part)] = cpu.view(
            1, 1, part.num_heads, part.head_size
        )
    identity_layers = frozenset(
        name
        for name, order in orders.items()
        if tuple(order.tolist()) == tuple(range(order.numel()))
    )
    return ExecutionPlan(effective, layout, signs, orders, identity_layers)
