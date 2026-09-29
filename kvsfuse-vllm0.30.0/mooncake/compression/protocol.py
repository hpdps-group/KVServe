# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec

ReqId = str
TransferId = str


class MooncakeXferMetadata(  # type: ignore[call-arg]
    msgspec.Struct, omit_defaults=True
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[str, tuple[str, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    remote_pp_size: int = 1


class CompressionSessionOpen(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    metadata: MooncakeXferMetadata
    nonce: str = ""


class CompressionSessionOpened(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    logical_block_count: int
    total_chunks: int
    error: str | None = None


class CompressionChunkPull(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    chunk_id: int
    slot_id: int
    payload_addr: int
    aux_addr: int
    tail_addr: int
    nonce: str = ""


class CompressionChunkReady(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    chunk_id: int
    data_bytes: int
    payload_bytes: int
    aux_bytes: int
    error: str | None = None

    @property
    def u8_bytes(self) -> int:
        return self.data_bytes

    @property
    def codec_bytes(self) -> int:
        return self.payload_bytes


class CompressionAbort(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    reason: str
    nonce: str = ""


class CompressionAbortAck(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    nonce: str
    error: str | None = None


CompressionRequest = CompressionSessionOpen | CompressionChunkPull | CompressionAbort
CompressionResponse = (
    CompressionSessionOpened | CompressionChunkReady | CompressionAbortAck
)
