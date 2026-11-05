# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Model-independent capability selection for the optional fused execution path.

Dimensions come from the memory plan. D128 and bounded token tiles are kernel
capabilities, not a model allow-list. Unsupported geometry uses the existing
compression implementation; it never selects uncompressed transport.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedGeometry:
    layers: int
    heads: int
    head_size: int
    capacity: int

    @property
    def row_elements(self):
        return self.heads * self.head_size

    def packed_bytes(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.capacity:
            raise ValueError("Invalid fused frame token count")
        return self.layers * ((tokens + 15) // 16 * 16) * self.row_elements


def select_geometry(plan, cache_layout, blocks_per_logical):
    """Return (geometry, reason); read metadata only, without initializing CUDA."""
    config, layout = plan.config, plan.layout
    if (cache_layout != "HND" or blocks_per_logical != 1
            or config.pipeline != ("transformer", "quantizer", "codec")
            or config.transformer != "hadamard" or config.codec != "ans"
            or config.uses_aggregate):
        return None, "requires HND token views and Hadamard/quantizer/ANS"
    capacity = config.logical_blocks_per_chunk
    if not 0 < capacity <= 1024 or capacity % 16:
        return None, "fused token capacity must be 16-aligned and at most 1024"
    key, value = tuple(layout.parts(0)), tuple(layout.parts(1))
    if not key or len(key) != len(value):
        return None, "requires matching full-attention K/V layer lists"
    heads = key[0].num_heads
    if heads not in (1, 2, 4, 8, 16):
        return None, "no vector kernel specialization for this head count"
    for index, (k, v) in enumerate(zip(key, value)):
        if k.layer_name != v.layer_name or k.layer_index != v.layer_index:
            return None, "K/V layer identities differ"
        for part in (k, v):
            if (part.num_heads != heads or part.head_size != 128
                    or part.block_size != 1 or str(part.dtype) != "torch.bfloat16"
                    or len(part.sections) != 1):
                return None, "requires uniform BF16 D128 full-head sections"
            section = part.sections[0]
            if (section.head_start != 0 or section.head_count != heads
                    or section.axis != ("channel" if part.kv_part == 0 else "token")
                    or not 2 <= section.num_levels <= 256):
                return None, "unsupported quantization section"
            if part.raw_bytes != capacity * heads * 128 * 2:
                return None, "raw layout capacity differs from configured capacity"
            if part.u8_offset != index * capacity * heads * 128:
                return None, "packed layer offsets must be contiguous"
    return FusedGeometry(len(key), heads, 128, capacity), "supported"
