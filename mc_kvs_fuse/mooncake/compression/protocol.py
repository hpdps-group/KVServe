# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Literal

import msgspec

from ..data import MooncakeXferMetadata, ReqId, TransferId


class CompressionHandshake(msgspec.Struct, frozen=True):
    protocol_version: int
    fingerprint: str
    kv_cache_layout: str
    tp_size: int
    pp_size: int
    logical_blocks_per_chunk: int
    slot_count: int
    payload_capacity: int
    aux_capacity: int


class CompressionSessionOpen(msgspec.Struct, tag=True, frozen=True):
    metadata: MooncakeXferMetadata
    handshake: CompressionHandshake


class CompressionSessionOpened(msgspec.Struct, tag=True, frozen=True):
    transfer_id: TransferId
    d_req_id: ReqId
    logical_block_count: int
    total_chunks: int
    raw_complete: bool
    error: str | None = None


class CompressionChunkPull(msgspec.Struct, tag=True, frozen=True):
    transfer_id: TransferId
    d_req_id: ReqId
    chunk_id: int
    kv_part: Literal[0, 1]
    logical_start: int
    logical_block_count: int
    slot_id: int
    payload_addr: int
    payload_capacity: int
    aux_addr: int
    aux_capacity: int


class CompressionChunkReady(msgspec.Struct, tag=True, frozen=True):
    transfer_id: TransferId
    d_req_id: ReqId
    chunk_id: int
    kv_part: Literal[0, 1]
    slot_id: int
    logical_block_count: int
    u8_bytes: int
    codec_bytes: int
    aux_bytes: int
    error: str | None = None


class CompressionAbort(msgspec.Struct, tag=True, frozen=True):
    transfer_id: TransferId
    d_req_id: ReqId
    reason: str


CompressionRequest = CompressionSessionOpen | CompressionChunkPull | CompressionAbort
CompressionResponse = CompressionSessionOpened | CompressionChunkReady
