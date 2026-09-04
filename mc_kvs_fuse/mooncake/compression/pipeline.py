# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

from .codec import ANSCodec, EncodedChunk
from .layout import CompressionLayout, KVPart, LayerPartLayout, QuantSection


@dataclass
class CompressionSlotBuffers:
    arena_a: torch.Tensor
    arena_b: torch.Tensor
    aux: torch.Tensor
    block_indices: torch.Tensor
    completion_event: torch.cuda.Event


class CompressionPipeline:
    def __init__(
        self,
        layout: CompressionLayout,
        kv_cache_layout: str,
        blocks_per_logical: int,
        slots: list[CompressionSlotBuffers],
        codec: ANSCodec,
        stream: torch.cuda.Stream,
        signs: dict[tuple[str, KVPart], torch.Tensor],
        head_orders: dict[str, torch.Tensor],
    ):
        if kv_cache_layout not in ("HND", "NHD"):
            raise ValueError(f"Unsupported KV cache layout: {kv_cache_layout}")
        self.layout = layout
        self.kv_cache_layout = kv_cache_layout
        self.blocks_per_logical = blocks_per_logical
        self.slots = slots
        self.codec = codec
        self.stream = stream
        self.signs = signs
        self.head_orders = head_orders
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.logger = init_logger(__name__)
        self._trace_stats = (
            os.environ.get("VLLM_MOONCAKE_COMPRESSION_TRACE", "0") == "1"
        )

    def _trace_tensor(
        self,
        stage: str,
        part: LayerPartLayout,
        tensor: torch.Tensor,
    ) -> None:
        if not self._trace_stats:
            return
        finite = torch.isfinite(tensor)
        self.logger.debug(
            "Mooncake compression trace: stage=%s layer=%s kv_part=%d "
            "shape=%s min=%.7g max=%.7g mean=%.7g absmax=%.7g "
            "nonfinite=%d",
            stage,
            part.layer_name,
            part.kv_part,
            tuple(tensor.shape),
            tensor.amin().item(),
            tensor.amax().item(),
            tensor.float().mean().item(),
            tensor.abs().amax().item(),
            (~finite).sum().item(),
        )

    @staticmethod
    def _debug_digest(tensor: torch.Tensor) -> str:
        # NumPy cannot represent BF16; hashing the raw contiguous bytes keeps
        # the trace lossless for every supported KV dtype.
        data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        for part in self.layout.layer_parts:
            if part.layer_name not in kv_caches:
                raise ValueError(
                    f"Missing Full Attention KV cache tensor: {part.layer_name}"
                )
            cache = kv_caches[part.layer_name]
            if not isinstance(cache, torch.Tensor):
                raise ValueError(
                    f"Full Attention cache {part.layer_name} is not a tensor."
                )
            if cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    f"Unsupported KV dtype for {part.layer_name}: {cache.dtype}"
                )
            if cache.ndim != 4:
                raise ValueError(
                    "Expected packed KV tensor [blocks,heads,tokens,K+V] "
                    f"for {part.layer_name}, got {tuple(cache.shape)}."
                )
            expected_content = sum(
                candidate.head_size
                for candidate in self.layout.layer_parts
                if candidate.layer_name == part.layer_name
            )
            if cache.shape[1] != part.num_heads or cache.shape[-1] != expected_content:
                raise ValueError(
                    f"Packed KV shape mismatch for {part.layer_name}: "
                    f"shape={tuple(cache.shape)}, heads={part.num_heads}, "
                    f"K+V={expected_content}."
                )
            content = cache.shape[-1]
            expected_inner_strides = (
                (cache.shape[2] * content, content)
                if self.kv_cache_layout == "HND"
                else (content, cache.shape[1] * content)
            )
            if cache.stride(-1) != 1 or cache.stride()[1:3] != expected_inner_strides:
                raise ValueError(
                    f"KV cache strides do not match {self.kv_cache_layout} for "
                    f"{part.layer_name}: shape={tuple(cache.shape)}, "
                    f"stride={tuple(cache.stride())}."
                )
            self.kv_caches[part.layer_name] = cache

    def _physical_ids(self, logical_ids: list[int]) -> list[int]:
        count = self.blocks_per_logical
        return [
            logical_id * count + offset
            for logical_id in logical_ids
            for offset in range(count)
        ]

    def _index_tensor(
        self, slot: CompressionSlotBuffers, logical_ids: list[int]
    ) -> torch.Tensor:
        max_logical_blocks = slot.block_indices.numel() // self.blocks_per_logical
        if not 1 <= len(logical_ids) <= max_logical_blocks:
            raise ValueError(
                "Compression chunk logical block count must be between 1 and "
                f"{max_logical_blocks}, got {len(logical_ids)}."
            )
        physical_count = len(logical_ids) * self.blocks_per_logical
        if physical_count > slot.block_indices.numel():
            raise RuntimeError("Compression block-index buffer capacity exceeded.")
        indices = slot.block_indices[:physical_count]
        logical = torch.tensor(logical_ids, dtype=torch.int64, device="cpu")
        offsets = torch.arange(self.blocks_per_logical, dtype=torch.int64, device="cpu")
        physical = (
            logical.unsqueeze(1) * self.blocks_per_logical + offsets.unsqueeze(0)
        ).flatten()
        indices.copy_(physical)
        return indices

    def _cache_part(self, part: LayerPartLayout) -> torch.Tensor:
        cache = self.kv_caches[part.layer_name]
        key_size = next(
            candidate.head_size
            for candidate in self.layout.layer_parts
            if candidate.layer_name == part.layer_name and candidate.kv_part == 0
        )
        start = 0 if part.kv_part == 0 else key_size
        return cache.narrow(-1, start, part.head_size)

    def _layer_views(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_part = self._cache_part(part)
        kernel_tokens = part.block_size // self.blocks_per_logical
        raw_numel = physical_blocks * kernel_tokens * part.num_heads * part.head_size
        raw_bytes = raw_numel * cache_part.element_size()
        raw_a = slot.arena_a[part.raw_offset : part.raw_offset + raw_bytes].view(
            cache_part.dtype
        )
        canonical_a = raw_a.view(
            physical_blocks, kernel_tokens, part.num_heads, part.head_size
        )
        if self.kv_cache_layout == "HND":
            scratch = raw_a.view(
                physical_blocks, part.num_heads, kernel_tokens, part.head_size
            )
        else:
            scratch = canonical_a.permute(0, 2, 1, 3)
        # Keep both raw ping-pong buffers in arena A.  Arena B is reserved for
        # the packed u8 stream and must not be touched by later layer gathers.
        canonical_b = (
            slot.arena_a[
                self.layout.scratch_stride
                + part.raw_offset : self.layout.scratch_stride
                + part.raw_offset
                + raw_bytes
            ]
            .view(cache_part.dtype)
            .view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        )
        return scratch, canonical_a, canonical_b

    def _gather_rearrange(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
    ) -> None:
        physical_blocks = indices.numel()
        scratch, _, canonical_b = self._layer_views(slot, part, physical_blocks)
        source = self._cache_part(part)
        torch.index_select(source, 0, indices, out=scratch)
        source_canonical = scratch.permute(0, 2, 1, 3)
        torch.index_select(
            source_canonical,
            2,
            self.head_orders[part.layer_name],
            out=canonical_b,
        )
        self._trace_tensor("rearranged", part, canonical_b)

    def _hadamard_forward(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> None:
        _, canonical_a, canonical_b = self._layer_views(slot, part, physical_blocks)
        canonical_b.mul_(self.signs[(part.layer_name, part.kv_part)])
        self._hadamard_transform(canonical_b, canonical_a, part.head_size)
        self._trace_tensor("hadamard", part, canonical_a)

    @staticmethod
    def _hadamard_transform(
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        block_size: int,
    ) -> None:
        """Run normalized FWHT using the two preallocated arena buffers.

        Humming's JIT runtime requires its call site to be the process main
        thread, while Mooncake serves compression requests from an asyncio
        worker thread. The buffer-ping-pong implementation keeps the same
        normalized Walsh-Hadamard transform without allocating a temporary
        CUDA tensor in the request path.
        """
        if os.environ.get("VLLM_MOONCAKE_COMPRESSION_IDENTITY_HADAMARD") == "1":
            outputs.copy_(inputs)
            return

        if threading.current_thread() is threading.main_thread():
            from humming.ops import hadamard_transform

            hadamard_transform(
                inputs,
                block_size,
                scale=1.0,
                outputs=outputs,
            )
            return

        if block_size < 2 or block_size & (block_size - 1):
            raise ValueError(
                f"Hadamard block size must be a power of two: {block_size}"
            )
        if not inputs.is_contiguous() or not outputs.is_contiguous():
            raise ValueError("Hadamard buffers must be contiguous.")
        if inputs.shape != outputs.shape or inputs.size(-1) != block_size:
            raise ValueError("Hadamard input/output shapes are incompatible.")

        src = inputs.reshape(-1, block_size)
        output_flat = outputs.reshape(-1, block_size)
        dst = output_flat
        step = 1
        while step < block_size:
            src_pairs = src.view(-1, block_size // (2 * step), 2, step)
            dst_pairs = dst.view(-1, block_size // (2 * step), 2, step)
            even = src_pairs[:, :, 0, :]
            odd = src_pairs[:, :, 1, :]
            torch.add(even, odd, out=dst_pairs[:, :, 0, :])
            torch.sub(even, odd, out=dst_pairs[:, :, 1, :])
            src, dst = dst, src
            step <<= 1

        if src.data_ptr() != output_flat.data_ptr():
            output_flat.copy_(src)
        output_flat.mul_(1.0 / math.sqrt(block_size))

    @staticmethod
    def _axes(section: QuantSection) -> tuple[int, ...]:
        if section.axis == "channel":
            return (0, 1)
        if section.axis == "token":
            return (2, 3)
        return (0, 1, 2, 3)

    def _aux_view(
        self,
        slot: CompressionSlotBuffers,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        temporary: bool = False,
    ) -> torch.Tensor:
        if temporary:
            offset += self.layout.aux_arena_bytes // 2
        nbytes = math.prod(shape) * 2
        return slot.aux[offset : offset + nbytes].view(dtype).view(shape)

    def _quantize(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> None:
        dtype = self._cache_part(part).dtype
        kernel_tokens = part.block_size // self.blocks_per_logical
        _, canonical_a, _ = self._layer_views(slot, part, physical_blocks)
        u8_layer = slot.arena_b[
            self.layout.u8_base + part.u8_offset : self.layout.u8_base
            + part.u8_offset
            + physical_blocks * kernel_tokens * part.num_heads * part.head_size
        ].view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        for section in part.sections:
            source = canonical_a.narrow(2, section.head_start, section.head_count)
            output = u8_layer.narrow(2, section.head_start, section.head_count)
            aux_shape = (
                (1, 1, section.head_count, part.head_size)
                if section.axis == "channel"
                else (physical_blocks, kernel_tokens, 1, 1)
                if section.axis == "token"
                else (1, 1, 1, 1)
            )
            minimum = self._aux_view(slot, section.aux_min_offset, aux_shape, dtype)
            scale = self._aux_view(slot, section.aux_scale_offset, aux_shape, dtype)
            maximum = self._aux_view(
                slot,
                section.aux_min_offset,
                aux_shape,
                dtype,
                temporary=True,
            )
            inverse_scale = self._aux_view(
                slot,
                section.aux_scale_offset,
                aux_shape,
                dtype,
                temporary=True,
            )
            torch.amin(source, dim=self._axes(section), keepdim=True, out=minimum)
            torch.amax(source, dim=self._axes(section), keepdim=True, out=maximum)
            inverse_scale.copy_(maximum).sub_(minimum).clamp_(min=1e-5)
            inverse_scale.div_(section.num_levels - 1)
            scale.copy_(inverse_scale)
            inverse_scale.reciprocal_()
            source.sub_(minimum).mul_(inverse_scale).round_().clamp_(
                0, section.num_levels - 1
            )
            output.copy_(source)
        self._trace_tensor("quantized", part, u8_layer)

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
    ) -> EncodedChunk:
        slot = self.slots[slot_id]
        logical_counts = {
            len(logical_ids_by_group[index]) for index in self.layout.fa_group_indices
        }
        if len(logical_counts) != 1:
            raise ValueError(
                "All compressed Full Attention groups must provide the same "
                "logical block count."
            )
        logical_count = logical_counts.pop()
        max_logical_blocks = slot.block_indices.numel() // self.blocks_per_logical
        if not 1 <= logical_count <= max_logical_blocks:
            raise ValueError(
                "Compression chunk logical block count must be between 1 and "
                f"{max_logical_blocks}, got {logical_count}."
            )
        physical_count = logical_count * self.blocks_per_logical
        self.logger.debug(
            "Mooncake compression encode: slot=%d kv_part=%d logical_ids=%s "
            "physical_blocks=%d raw_bytes=%d u8_bytes=%d",
            slot_id,
            kv_part,
            [group for group in logical_ids_by_group if group],
            physical_count,
            self.layout.raw_bytes_by_part[kv_part],
            self.layout.u8_bytes_by_part[kv_part],
        )
        with torch.cuda.stream(self.stream):
            # Layout offsets and codec capacities are planned for the maximum
            # chunk. Clear unused tail regions so a short final chunk cannot
            # encode stale data from the previous slot owner.
            u8_bytes = self.layout.u8_bytes_by_part[kv_part]
            aux_bytes = self.layout.aux_bytes_by_part[kv_part]
            slot.arena_b[self.layout.u8_base : self.layout.u8_base + u8_bytes].zero_()
            slot.aux[:aux_bytes].zero_()
            for part in self.layout.parts(kv_part):
                logical_ids = logical_ids_by_group[part.group_index]
                if len(logical_ids) != logical_count:
                    raise ValueError(
                        "Compressed Full Attention group block count changed "
                        "while encoding a chunk."
                    )
                indices = self._index_tensor(slot, logical_ids)
                self._gather_rearrange(slot, part, indices)
                self._hadamard_forward(slot, part, indices.numel())
                reference = None
                if self._trace_stats:
                    _, canonical_a, _ = self._layer_views(slot, part, indices.numel())
                    reference = canonical_a.clone()
                self._quantize(slot, part, indices.numel())
                if reference is not None:
                    self._dequantize(slot, part, indices.numel())
                    _, canonical_a, _ = self._layer_views(slot, part, indices.numel())
                    error = (canonical_a - reference).abs()
                    self.logger.debug(
                        "Mooncake compression trace: stage=quant_error "
                        "layer=%s kv_part=%d max=%.7g mean=%.7g",
                        part.layer_name,
                        part.kv_part,
                        error.amax().item(),
                        error.float().mean().item(),
                    )
            if self._trace_stats:
                self.logger.debug(
                    "Mooncake compression trace: stage=u8_encode kv_part=%d "
                    "bytes=%d byte_sum=%d sha256=%s",
                    kv_part,
                    u8_bytes,
                    int(
                        slot.arena_b[
                            self.layout.u8_base : self.layout.u8_base + u8_bytes
                        ]
                        .sum()
                        .item()
                    ),
                    self._debug_digest(
                        slot.arena_b[
                            self.layout.u8_base : self.layout.u8_base + u8_bytes
                        ]
                    ),
                )
            return self.codec.encode(
                slot.arena_b[self.layout.u8_base :],
                slot.arena_a,
                u8_bytes,
                slot.completion_event,
            )

    def _dequantize(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> None:
        dtype = self._cache_part(part).dtype
        kernel_tokens = part.block_size // self.blocks_per_logical
        _, canonical_a, _ = self._layer_views(slot, part, physical_blocks)
        u8_layer = slot.arena_b[
            self.layout.u8_base + part.u8_offset : self.layout.u8_base
            + part.u8_offset
            + physical_blocks * kernel_tokens * part.num_heads * part.head_size
        ].view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        for section in part.sections:
            source = u8_layer.narrow(2, section.head_start, section.head_count)
            output = canonical_a.narrow(2, section.head_start, section.head_count)
            aux_shape = (
                (1, 1, section.head_count, part.head_size)
                if section.axis == "channel"
                else (physical_blocks, kernel_tokens, 1, 1)
                if section.axis == "token"
                else (1, 1, 1, 1)
            )
            minimum = self._aux_view(slot, section.aux_min_offset, aux_shape, dtype)
            scale = self._aux_view(slot, section.aux_scale_offset, aux_shape, dtype)
            if self._trace_stats:
                self.logger.debug(
                    "Mooncake compression trace: stage=section_decode "
                    "layer=%s kv_part=%d precision=%s axis=%s heads=%d:%d "
                    "u8_offset=%d aux_min_offset=%d aux_scale_offset=%d "
                    "u8_sha256=%s min_sha256=%s scale_sha256=%s",
                    part.layer_name,
                    part.kv_part,
                    section.precision,
                    section.axis,
                    section.head_start,
                    section.head_start + section.head_count,
                    self.layout.u8_base + section.u8_offset,
                    section.aux_min_offset,
                    section.aux_scale_offset,
                    self._debug_digest(
                        slot.arena_b[
                            self.layout.u8_base
                            + section.u8_offset : self.layout.u8_base
                            + section.u8_offset
                            + section.raw_numel
                        ]
                    ),
                    self._debug_digest(minimum),
                    self._debug_digest(scale),
                )
            output.copy_(source).mul_(scale).add_(minimum)
        self._trace_tensor("dequantized", part, canonical_a)

    def _inverse_hadamard_scatter(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
    ) -> None:
        scratch, canonical_a, canonical_b = self._layer_views(
            slot, part, indices.numel()
        )
        self._hadamard_transform(canonical_a, canonical_b, part.head_size)
        canonical_b.mul_(self.signs[(part.layer_name, part.kv_part)])
        unordered = scratch.permute(0, 2, 1, 3)
        unordered.index_copy_(2, self.head_orders[part.layer_name], canonical_b)
        self._cache_part(part).index_copy_(0, indices, scratch)
        self._trace_tensor("inverse_hadamard", part, canonical_b)

    def decode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        codec_bytes: int,
        u8_bytes: int,
    ) -> torch.cuda.Event:
        expected_u8 = self.layout.u8_bytes_by_part[kv_part]
        if u8_bytes != expected_u8:
            raise ValueError(
                f"Compressed u8 size mismatch: remote={u8_bytes}, local={expected_u8}."
            )
        slot = self.slots[slot_id]
        logical_counts = {
            len(logical_ids_by_group[index]) for index in self.layout.fa_group_indices
        }
        if len(logical_counts) != 1:
            raise ValueError(
                "All compressed Full Attention groups must provide the same "
                "logical block count."
            )
        logical_count = logical_counts.pop()
        max_logical_blocks = slot.block_indices.numel() // self.blocks_per_logical
        if not 1 <= logical_count <= max_logical_blocks:
            raise ValueError(
                "Compression chunk logical block count must be between 1 and "
                f"{max_logical_blocks}, got {logical_count}."
            )
        self.logger.debug(
            "Mooncake compression decode: slot=%d kv_part=%d logical_ids=%s "
            "codec_bytes=%d u8_bytes=%d",
            slot_id,
            kv_part,
            [group for group in logical_ids_by_group if group],
            codec_bytes,
            u8_bytes,
        )
        with torch.cuda.stream(self.stream):
            self.codec.decode(
                slot.arena_a,
                slot.arena_b[self.layout.u8_base :],
                codec_bytes,
                u8_bytes,
                slot.completion_event,
            )
            if self._trace_stats:
                self.logger.debug(
                    "Mooncake compression trace: stage=u8_decode kv_part=%d "
                    "bytes=%d byte_sum=%d sha256=%s",
                    kv_part,
                    u8_bytes,
                    int(
                        slot.arena_b[
                            self.layout.u8_base : self.layout.u8_base + u8_bytes
                        ]
                        .sum()
                        .item()
                    ),
                    self._debug_digest(
                        slot.arena_b[
                            self.layout.u8_base : self.layout.u8_base + u8_bytes
                        ]
                    ),
                )
            for part in self.layout.parts(kv_part):
                logical_ids = logical_ids_by_group[part.group_index]
                if len(logical_ids) != logical_count:
                    raise ValueError(
                        "Compressed Full Attention group block count changed "
                        "while decoding a chunk."
                    )
                indices = self._index_tensor(slot, logical_ids)
                self._dequantize(slot, part, indices.numel())
                self._inverse_hadamard_scatter(slot, part, indices)
            slot.completion_event.record(self.stream)
        return slot.completion_event
