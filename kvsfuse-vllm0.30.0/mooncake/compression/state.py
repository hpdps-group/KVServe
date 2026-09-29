# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .protocol import ReqId, TransferId

SendBlockMeta = Any


class SessionState(Enum):
    PLANNED = auto()
    OPENING = auto()
    ACTIVE = auto()
    DRAINING = auto()
    COMPLETE = auto()
    FAILED = auto()
    CANCELLED = auto()


class ReceiveSlotState(Enum):
    FREE = auto()
    GRANTED = auto()
    RECEIVED = auto()
    DECODING = auto()
    WRITEBACK = auto()


@dataclass
class ReceiveSlot:
    slot_id: int
    state: ReceiveSlotState = ReceiveSlotState.FREE
    chunk_id: int | None = None


@dataclass
class ProducerSession:
    transfer_id: TransferId
    d_req_id: ReqId
    send_meta: SendBlockMeta
    local_block_ids: list[list[int]]
    remote_block_ids: list[list[int]]
    remote_session: str
    logical_block_count: int = 0
    state: SessionState = SessionState.PLANNED
    completed_chunks: int = 0
    total_chunks: int = 0
    inflight_chunks: int = 0
    error: str | None = None
    execution: Any = None
    nonce: str = ""
    valid_tokens_by_group: dict[int, int] = field(default_factory=dict)


@dataclass
class ConsumerSession:
    transfer_id: TransferId
    d_req_id: ReqId
    block_ids: list[list[int]]
    logical_block_count: int
    total_chunks: int
    state: SessionState = SessionState.PLANNED
    slots: list[ReceiveSlot] = field(default_factory=list)
    completed_chunks: int = 0
    error: str | None = None
    execution: Any = None
    nonce: str = ""
    valid_tokens_by_group: dict[int, int] = field(default_factory=dict)
