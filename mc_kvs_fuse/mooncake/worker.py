# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import hashlib
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMemoryPlan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

from .compression.protocol import (
    CompressionAbort,
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionRequest,
    CompressionResponse,
    CompressionSessionOpen,
    CompressionSessionOpened,
)
from .compression.state import (
    ConsumerSession,
    ProducerSession,
    ReceiveSlot,
    ReceiveSlotState,
    SessionState,
)
from .data import (
    MooncakeConnectorMetadata,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    ReqId,
    SendBlockMeta,
    TransferId,
    TransferRegion,
)
from .mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from .stats import MooncakeKVConnectorStats

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Mooncake's TCP transport becomes unreliable when tens of thousands of
# non-contiguous KV-cache regions are submitted as one batch. Keep each raw
# transfer batch bounded by both descriptor count and payload size. This does
# not change request-level completion semantics: the decoder is acknowledged
# only after every batch succeeds.
_RAW_TRANSFER_MAX_DESCRIPTORS = 1024
_RAW_TRANSFER_MAX_BYTES = 64 * 1024 * 1024


def _compression_total_chunks(
    logical_block_count: int, logical_blocks_per_chunk: int
) -> int:
    if logical_block_count < 0:
        raise ValueError("logical_block_count cannot be negative.")
    if logical_blocks_per_chunk < 1:
        raise ValueError("logical_blocks_per_chunk must be positive.")
    logical_chunks = (
        logical_block_count + logical_blocks_per_chunk - 1
    ) // logical_blocks_per_chunk
    return logical_chunks * 2


def _compression_chunk_window(
    chunk_id: int,
    logical_block_count: int,
    logical_blocks_per_chunk: int,
) -> tuple[int, int, int]:
    total_chunks = _compression_total_chunks(
        logical_block_count, logical_blocks_per_chunk
    )
    if not 0 <= chunk_id < total_chunks:
        raise IndexError(
            f"Compression chunk {chunk_id} is outside [0, {total_chunks})."
        )
    logical_chunk = chunk_id // 2
    logical_start = logical_chunk * logical_blocks_per_chunk
    count = min(logical_blocks_per_chunk, logical_block_count - logical_start)
    return logical_start, count, chunk_id % 2


def _compression_trace_enabled() -> bool:
    return os.environ.get("VLLM_MOONCAKE_COMPRESSION_TRACE", "0") == "1"


def _compression_trace_digest(tensor: torch.Tensor, nbytes: int) -> str:
    """Return a host digest for trace-only byte-for-byte comparisons."""
    return hashlib.sha256(tensor[:nbytes].detach().cpu().numpy().tobytes()).hexdigest()


try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    is_kv_layout_blocks_first: bool,
    group_indices: list[int] | None = None,
    split_kv_regions: list[bool] | None = None,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), (
        "Mooncake transfer regions require matching metadata lengths, got "
        f"base_addrs={len(base_addrs)}, block_lens={len(block_lens)}, "
        f"kv_block_lens={len(kv_block_lens)}, "
        f"layer_names={len(layer_names)}, "
        f"layer_indices={len(layer_indices)}."
    )
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names), (
        "Mooncake transfer regions require matching group metadata lengths, "
        f"got group_indices={len(group_indices)}, layer_names={len(layer_names)}."
    )
    if split_kv_regions is None:
        split_kv_regions = [is_kv_layout_blocks_first] * len(layer_names)
    assert len(split_kv_regions) == len(layer_names), (
        "Mooncake transfer regions require matching split metadata, "
        f"got split_kv_regions={len(split_kv_regions)}, "
        f"layer_names={len(layer_names)}."
    )
    regions: list[TransferRegion] = []

    for i in range(len(base_addrs)):
        base_addr = base_addrs[i]
        block_len = block_lens[i]
        kv_block_len = kv_block_lens[i]
        layer_name = layer_names[i]
        layer_index = layer_indices[i]
        group_index = group_indices[i]
        split_kv_region = split_kv_regions[i]

        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
            )
        )
        if split_kv_region:
            regions.append(
                TransferRegion(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                    group_index=group_index,
                )
            )

    return regions


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    """Align KV transfer regions by registered layer-name occurrence.

    PP shards own different layer subsets. Positional matching is therefore
    wrong once producer and consumer have different PP layouts. Multiple
    registered transfer buffers for the same layer are represented by repeated
    layer names and matched by occurrence order.
    """

    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, int], TransferRegion]]:
        counts: dict[str, int] = defaultdict(int)
        keyed: list[tuple[tuple[str, int], TransferRegion]] = []
        for region in regions:
            occurrence = counts[region.layer_name]
            counts[region.layer_name] += 1
            keyed.append(((region.layer_name, occurrence), region))
        return keyed

    local_keyed = keyed_regions(local_regions)
    remote_keyed = keyed_regions(remote_regions)
    remote_by_key = dict(remote_keyed)
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in local_keyed:
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            return (
                [],
                [],
                (
                    "Mooncake producer registered layer has no matching "
                    f"consumer occurrence: {key[0]} occurrence {key[1]}."
                ),
            )
        if local_region.layer_index != remote_region.layer_index:
            return (
                [],
                [],
                (
                    "Mooncake registered layer index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.layer_index}, consumer="
                    f"{remote_region.layer_index}."
                ),
            )
        if local_region.group_index != remote_region.group_index:
            return (
                [],
                [],
                (
                    "Mooncake registered group index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.group_index}, consumer="
                    f"{remote_region.group_index}."
                ),
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)

    return aligned_local, aligned_remote, None


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # Only the TP=0, PP=0 worker of the designated engine should launch it.
    if get_tensor_model_parallel_rank() != 0:
        return False
    if get_pp_group().rank_in_group != 0:
        return False

    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    if parallel_config.local_engines_only:
        return parallel_config.data_parallel_rank_local == 0

    # In internal LB mode,
    # only the first data-parallel engine should launch the bootstrap server.
    return parallel_config.data_parallel_index == 0


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    elif parallel_config.nnodes_within_dp > 1:
        # Internal LB multi-node TP/PP uses the model-parallel master as the
        # single bootstrap endpoint for all ranks in the engine.
        host = parallel_config.master_addr
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)


r"""
==============================================================
     __  __  ___   ___  _   _  ____    _    _  _______
    |  \/  |/ _ \ / _ \| \ | |/ ___|  / \  | |/ / ____|
    | |\/| | | | | | | |  \| | |     / _ \ | ' /|  _|
    | |  | | |_| | |_| | |\  | |___ / ___ \| . \| |___
    |_|  |_|\___/ \___/|_| \_|\____/_/   \_\_|\_\_____|

      ____ ___  _   _ _   _ _____ ____ _____ ___  ____
     / ___/ _ \| \ | | \ | | ____/ ___|_   _/ _ \|  _ \
    | |  | | | |  \| |  \| |  _|| |     | || | | | |_) |
    | |__| |_| | |\  | |\  | |__| |___  | || |_| |  _ <
     \____\___/|_| \_|_| \_|_____\____| |_| \___/|_| \_\

    __        _____  ____  _  _______ ____
    \ \      / / _ \|  _ \| |/ / ____|  _ \
     \ \ /\ / / | | | |_) | ' /|  _| | |_) |
      \ V  V /| |_| |  _ <| . \| |___|  _ <
       \_/\_/  \___/|_| \_\_|\_\_____|_| \_\

==============================================================
"""


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
        # may change the current CUDA device during engine.initialize().
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        extra_config = kv_transfer_config.kv_connector_extra_config
        self._timing_enabled = bool(
            extra_config.get("enable_timing", False)
            or os.environ.get("VLLM_MOONCAKE_TIMING", "0") == "1"
        )
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        device_name = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "device_name", ""
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", protocol, device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.block_len_per_layer: list[int] = []
        self.kv_block_len_per_layer: list[int] = []
        self.registered_layer_names: list[str] = []
        self.registered_layer_indices: list[int] = []
        self.registered_group_indices: list[int] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            # Each pool thread must be bound to the correct CUDA device
            # because CUDA device selection is thread-local.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
                initializer=self._bind_sender_thread_device,
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.xfer_stats = MooncakeKVConnectorStats()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug(
            "Detected attention backends %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._layer_specs: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                self._layer_specs[layer_name] = specs_by_layer.get(
                    layer_name, group_spec
                )
        attention_block_sizes = {
            spec.block_size
            for spec in self._layer_specs.values()
            if isinstance(spec, AttentionSpec)
        }
        if len(attention_block_sizes) > 1:
            raise ValueError(
                "Mooncake requires one logical block size across attention "
                f"groups, got {sorted(attention_block_sizes)}."
            )
        logical_block_size = (
            attention_block_sizes.pop() if attention_block_sizes else self.block_size
        )
        kernel_block_size = select_common_block_size(
            self.cache_config.block_size, self.attn_backends
        )
        if logical_block_size % kernel_block_size:
            raise ValueError(
                f"Mooncake logical block size {logical_block_size} is not "
                f"divisible by kernel block size {kernel_block_size}."
            )
        self._physical_blocks_per_logical_kv_block = (
            logical_block_size // kernel_block_size
        )
        self.block_size = kernel_block_size
        if self._physical_blocks_per_logical_kv_block != 1:
            logger.info_once(
                "Mooncake maps each %s-token logical attention block to %s "
                "%s-token kernel blocks.",
                logical_block_size,
                self._physical_blocks_per_logical_kv_block,
                kernel_block_size,
            )
        self._layer_group_indices: dict[str, int] = {
            layer: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=kv_cache_config.has_mamba_layers,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
        self._compression_req_decoder = msgspec.msgpack.Decoder(CompressionRequest)
        self._compression_resp_decoder = msgspec.msgpack.Decoder(CompressionResponse)
        self.compression = None
        self._compression_producer_sessions: dict[
            tuple[bytes, TransferId, ReqId], ProducerSession
        ] = {}
        self._compression_sender_slots: asyncio.Queue[int] | None = None
        self._compression_receiver_slots: asyncio.Queue[int] | None = None

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
        correct CUDA device.  CUDA device selection is thread-local, so
        without this, NVLink transfers fail for TP ranks > 0."""
        current_platform.set_device(self.device_id)

    def _record_phase(self, phase: str, started_at: float) -> None:
        if getattr(self, "_timing_enabled", False):
            self.xfer_stats.record_phase(phase, time.perf_counter() - started_at)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        async_zmq_ctx = getattr(self, "async_zmq_ctx", None)
        if async_zmq_ctx is not None:
            async_zmq_ctx.term()
        if getattr(self, "compression", None) is not None:
            self.compression.shutdown()
        if not getattr(self, "is_kv_consumer", True):
            sender_executor = getattr(self, "_sender_executor", None)
            if sender_executor is not None:
                sender_executor.shutdown(wait=False)
            sender_loop = getattr(self, "sender_loop", None)
            if sender_loop is not None and sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if sender_loop is not None and not sender_loop.is_closed():
                sender_loop.close()
            if (
                hasattr(self, "vllm_config")
                and hasattr(self, "bootstrap_server")
                and should_launch_bootstrap_server(self.vllm_config)
            ):
                self.bootstrap_server.shutdown()
        receiver_loop = getattr(self, "receiver_loop", None)
        if (
            not getattr(self, "is_kv_producer", True)
            and receiver_loop is not None
            and receiver_loop.is_running()
        ):
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()
        if receiver_loop is not None and not receiver_loop.is_closed():
            receiver_loop.close()

    r"""
    ——————————————————————————————————————————————————————————————————————————————————————————————————
                                             VLLM CONNECTOR API
                             _     _               _                             _
              _ __ ___  __ _(_)___| |_ ___ _ __   | | ____   __    ___ __ _  ___| |__   ___  ___
             | '__/ _ \/ _` | / __| __/ _ \ '__|  | |/ /\ \ / /   / __/ _` |/ __| '_ \ / _ \/ __|
             | | |  __/ (_| | \__ \ ||  __/ |     |   <  \ V /   | (_| (_| | (__| | | |  __/\__ \
             |_|  \___|\__, |_|___/\__\___|_|     |_|\_\  \_/     \___\__,_|\___|_| |_|\___||___/
                       |___/
    ——————————————————————————————————————————————————————————————————————————————————————————————————
    """

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        region_base_addresses: list[int] = []
        seen_storage_ptrs: set[int] = set()
        self.block_len_per_layer = []
        self.kv_block_len_per_layer = []
        self.registered_layer_names = []
        self.registered_layer_indices = []
        self.registered_group_indices = []

        for layer_name, cache_or_caches in kv_caches.items():
            layer_index = extract_layer_index(layer_name)
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s because no KV cache spec is present.",
                    layer_name,
                )
                continue
            if isinstance(layer_spec, MambaSpec):
                conv, _ = cache_or_caches
                cache_list = [conv]
            else:
                cache_list = self.transfer_topo.get_transfer_cache_regions(
                    cache_or_caches, layer_spec
                )

            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                # INLINED FROM _log_debug_cache_registration().
                if logger.isEnabledFor(logging.DEBUG):
                    # INLINED FROM _get_tensor_dense_flag().
                    is_dense = getattr(cache, "is_non_overlapping_and_dense", None)
                    dense_flag = bool(is_dense()) if callable(is_dense) else None
                    logger.debug(
                        "Mooncake register view layer=%s shape=%s stride=%s "
                        "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
                        layer_name,
                        tuple(cache.shape),
                        tuple(cache.stride()),
                        cache.storage_offset(),
                        cache.is_contiguous(),
                        dense_flag,
                        cache.data_ptr(),
                    )
                base_addr = cache.data_ptr()
                block_len = cache.stride(0) * cache.element_size()
                region_base_addresses.append(base_addr)

                if isinstance(layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_block_len = layer_spec.page_size_bytes
                elif self.transfer_topo.virtually_split_kv_in_blocks and not isinstance(
                    layer_spec, MambaSpec
                ):
                    kv_block_len = block_len // 2
                else:
                    kv_block_len = block_len
                self.block_len_per_layer.append(block_len)
                self.kv_block_len_per_layer.append(kv_block_len)
                self.registered_layer_names.append(layer_name)
                self.registered_layer_indices.append(layer_index)
                self.registered_group_indices.append(
                    self._layer_group_indices[layer_name]
                )
                storage = cache.untyped_storage()
                storage_addr = storage.data_ptr()
                if storage_addr not in seen_storage_ptrs:
                    seen_storage_ptrs.add(storage_addr)
                    kv_data_ptrs.append(storage_addr)
                    kv_data_lens.append(storage.nbytes())

        logger.info("Mooncake registered KV layer metadata:")
        logger.info("  block_len_per_layer=%s", self.block_len_per_layer)
        logger.info("  kv_block_len_per_layer=%s", self.kv_block_len_per_layer)
        logger.info("  registered_layer_names=%s", self.registered_layer_names)
        logger.info("  registered_layer_indices=%s", self.registered_layer_indices)
        logger.info("  registered_group_indices=%s", self.registered_group_indices)

        self.kv_caches_base_addr = region_base_addresses
        self.seen_base_addresses = kv_data_ptrs

        if not kv_data_ptrs:
            raise RuntimeError("No KV cache tensors were registered with Mooncake.")

        if self.compression is not None:
            self.compression.bind_kv_caches(kv_caches, self._layer_group_indices)
            compression_ptrs, compression_lens = self.compression.registered_memory()
            kv_data_ptrs.extend(compression_ptrs)
            kv_data_lens.extend(compression_lens)

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        self.device_kv_caches = kv_caches
        logger.debug(
            "registered block_lens=%s kv_block_lens=%s",
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def initialize_worker_memory(
        self, memory_plan: KVConnectorWorkerMemoryPlan | None
    ) -> None:
        if memory_plan is None or memory_plan.connector_data is None:
            return
        from .compression.manager import MooncakeCompressionManager
        from .compression.memory import MooncakeCompressionMemoryPlan

        plan = memory_plan.connector_data
        if not isinstance(plan, MooncakeCompressionMemoryPlan):
            raise TypeError("Invalid Mooncake compression memory plan.")
        if self.is_kv_producer == self.is_kv_consumer:
            raise ValueError(
                "Mooncake compression v1 requires an explicit kv_producer or "
                "kv_consumer role."
            )
        self.compression = MooncakeCompressionManager(
            plan=plan,
            kv_cache_layout=self.kv_cache_layout,
            blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            device=torch.device("cuda", self.device_id),
        )
        self._compression_sender_slots = asyncio.Queue()
        self._compression_receiver_slots = asyncio.Queue()
        for slot_id in range(plan.config.slot_count):
            self._compression_sender_slots.put_nowait(slot_id)
            self._compression_receiver_slots.put_nowait(slot_id)
        logger.info(
            "Initialized Mooncake compression fixed pool: reserved=%d, "
            "arena=%d, aux_per_slot=%d, workspace=%d, slots=%d",
            plan.reserved_bytes,
            plan.layout.arena_bytes,
            plan.layout.aux_arena_bytes,
            plan.codec_workspace_bytes,
            plan.config.slot_count,
        )
        logger.debug(
            "Mooncake compression pool addresses: bulk=%d slots=%s",
            self.compression.bulk.data_ptr(),
            [
                {
                    "slot": index,
                    "arena_a": slot.arena_a.data_ptr(),
                    "arena_b": slot.arena_b.data_ptr(),
                    "aux": slot.aux.data_ptr(),
                    "indices": slot.block_indices.data_ptr(),
                }
                for index, slot in enumerate(self.compression.slots)
            ],
        )

    "———————————————— depth: 1, src: register_kv_caches() ————————————————"

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    "———————————————— depth: 2, src: _mooncake_sender_listener() ————————————————"

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    "———————————————— depth: 2, src: _mooncake_sender_listener() ————————————————"

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    logger.debug(
                        "Mooncake sender dequeued request: identity=%s bytes=%d "
                        "queue_size=%d",
                        identity.hex(),
                        len(metadata_bytes),
                        self.sender_worker_queue.qsize(),
                    )
                    compression_request = None
                    with contextlib.suppress(msgspec.DecodeError):
                        compression_request = self._compression_req_decoder.decode(
                            metadata_bytes
                        )
                    if compression_request is not None:
                        if self.compression is None:
                            response = CompressionSessionOpened(
                                transfer_id="",
                                d_req_id="",
                                logical_block_count=0,
                                total_chunks=0,
                                raw_complete=False,
                                error=(
                                    "Mooncake compression is not enabled on producer."
                                ),
                            )
                            await sock.send_multipart(
                                (identity, self._encoder.encode(response))
                            )
                        else:
                            await self._handle_compression_request(
                                identity, sock, compression_request
                            )
                    else:
                        metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                        if self.compression is not None:
                            response = MooncakeXferResponse(
                                status=MooncakeXferResponseStatus.ERROR,
                                err_msg=(
                                    "Mooncake compression is enabled on producer "
                                    "but not on consumer."
                                ),
                            )
                            await sock.send_multipart(
                                (identity, self._encoder.encode(response))
                            )
                        else:
                            await self.send_kv_to_decode(identity, sock, metadata)
                    logger.debug(
                        "Mooncake sender finished request: identity=%s",
                        identity.hex(),
                    )
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR,
                        err_msg=str(e),
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def _handle_compression_request(
        self,
        identity: bytes,
        sock: zmq.asyncio.Socket,
        request: CompressionRequest,
    ) -> None:
        if isinstance(request, CompressionSessionOpen):
            response = await self._open_compression_session(identity, request)
        elif isinstance(request, CompressionChunkPull):
            response = await self._serve_compression_chunk(identity, request)
        else:
            assert isinstance(request, CompressionAbort)
            key = (identity, request.transfer_id, request.d_req_id)
            session = self._compression_producer_sessions.get(key)
            if session is None:
                session_entry = next(
                    (
                        (candidate_key, candidate)
                        for candidate_key, candidate in (
                            self._compression_producer_sessions.items()
                        )
                        if candidate.transfer_id == request.transfer_id
                        and candidate.d_req_id == request.d_req_id
                    ),
                    None,
                )
                if session_entry is not None:
                    key, session = session_entry
            if session is not None:
                session.state = SessionState.CANCELLED
                if session.inflight_chunks == 0:
                    self._discard_compression_session(key, session)
            response = CompressionChunkReady(
                transfer_id=request.transfer_id,
                d_req_id=request.d_req_id,
                chunk_id=-1,
                kv_part=0,
                slot_id=-1,
                logical_block_count=0,
                u8_bytes=0,
                codec_bytes=0,
                aux_bytes=0,
                error=None,
            )
        logger.debug(
            "Mooncake sender responding: identity=%s request=%s "
            "transfer_id=%s d_req_id=%s chunk=%s error=%s",
            identity.hex(),
            type(request).__name__,
            getattr(response, "transfer_id", None),
            getattr(response, "d_req_id", None),
            getattr(response, "chunk_id", None),
            getattr(response, "error", None),
        )
        await sock.send_multipart((identity, self._encoder.encode(response)))
        logger.debug(
            "Mooncake sender response sent: identity=%s request=%s "
            "transfer_id=%s d_req_id=%s chunk=%s",
            identity.hex(),
            type(request).__name__,
            getattr(response, "transfer_id", None),
            getattr(response, "d_req_id", None),
            getattr(response, "chunk_id", None),
        )

    def _match_compression_blocks(
        self,
        send_meta: SendBlockMeta,
        remote_block_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        if len(send_meta.local_block_ids) != len(remote_block_ids):
            raise ValueError("KV group count mismatch in compression session.")
        local_result: list[list[int]] = []
        remote_result: list[list[int]] = []
        for group_index, (local_group, remote_group) in enumerate(
            zip(send_meta.local_block_ids, remote_block_ids)
        ):
            is_mamba = isinstance(
                self.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec,
                MambaSpec,
            )
            if is_mamba:
                local_group = [value for value in local_group if value != NULL_BLOCK_ID]
                remote_group = [
                    value for value in remote_group if value != NULL_BLOCK_ID
                ]
            if len(local_group) < len(remote_group):
                raise ValueError("Producer has fewer KV blocks than the consumer.")
            if len(local_group) > len(remote_group):
                local_group = local_group[-len(remote_group) :] if remote_group else []
            local_result.append(list(local_group))
            remote_result.append(list(remote_group))
        return local_result, remote_result

    @staticmethod
    def resolve_need_send(send_meta: SendBlockMeta, remote_tp_ranks: list[int]) -> None:
        """Record how many consumer TP workers must finish this transfer."""
        if send_meta.need_send:
            return
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: TP ranks=%s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    async def _open_compression_session(
        self,
        identity: bytes,
        request: CompressionSessionOpen,
    ) -> CompressionSessionOpened:
        assert self.compression is not None
        metadata = request.metadata
        if len(metadata.req_blocks) != 1:
            return CompressionSessionOpened(
                transfer_id="",
                d_req_id="",
                logical_block_count=0,
                total_chunks=0,
                raw_complete=False,
                error="Compression SessionOpen must contain exactly one request.",
            )
        d_req_id, (transfer_id, remote_block_ids) = next(
            iter(metadata.req_blocks.items())
        )
        logger.info(
            "Mooncake compression SessionOpen: d_req_id=%s transfer_id=%s "
            "remote=%s remote_tp=%d remote_pp=%d block_counts=%s",
            d_req_id,
            transfer_id,
            identity,
            metadata.remote_tp_size,
            request.handshake.pp_size,
            [len(group) for group in remote_block_ids],
        )
        send_meta: SendBlockMeta | None = None
        send_target_claimed = False
        created_send_meta = False
        try:
            self.compression.validate_handshake(request.handshake)
            if metadata.remote_tp_size != self.tp_size:
                raise ValueError("Compression v1 requires homogeneous TP.")
            if request.handshake.pp_size != self.pp_size:
                raise ValueError("Compression v1 requires homogeneous PP.")

            key = (identity, transfer_id, d_req_id)
            if key in self._compression_producer_sessions:
                raise ValueError("Compression session is already active.")

            send_meta = self.reqs_need_send.get(transfer_id)
            if send_meta is None:
                send_meta = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
                self.reqs_need_send[transfer_id] = send_meta
                created_send_meta = True
            await asyncio.wait_for(
                send_meta.ready.wait(),
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
            )
            local_ids, remote_ids = self._match_compression_blocks(
                send_meta, remote_block_ids
            )
            remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
                metadata.remote_tp_size
            )
            send_meta.sending += 1
            send_target_claimed = True
            self.resolve_need_send(send_meta, remote_tp_ranks)

            local_regions = self._get_transfer_regions(
                self.kv_caches_base_addr,
                self.block_len_per_layer,
                self.kv_block_len_per_layer,
                self.registered_layer_names,
                self.registered_layer_indices,
                self.registered_group_indices,
            )
            remote_regions = self._get_transfer_regions(
                metadata.kv_caches_base_addr,
                metadata.block_lens,
                metadata.kv_block_lens,
                metadata.registered_layer_names,
                metadata.registered_layer_indices,
                metadata.registered_group_indices,
            )
            local_regions, remote_regions, align_error = _align_transfer_regions(
                local_regions, remote_regions
            )
            if align_error:
                raise ValueError(align_error)
            raw_pairs = [
                (local, remote)
                for local, remote in zip(local_regions, remote_regions)
                if local.group_index not in self.compression.fa_group_indices
            ]
            raw_local = [pair[0] for pair in raw_pairs]
            raw_remote = [pair[1] for pair in raw_pairs]
            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
            ) = await self._build_transfer_params(
                [(d_req_id, send_meta)], metadata, raw_local, raw_remote
            )
            if err_reqs:
                raise RuntimeError(err_msg or "Raw GDN transfer planning failed.")
            remote_session = f"{metadata.remote_hostname}:{metadata.remote_port}"
            if src_ptrs:
                result = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )
                if result != 0:
                    raise RuntimeError(f"Mooncake raw transfer returned {result}.")
            logger.debug(
                "Mooncake compression SessionOpen raw path complete: "
                "d_req_id=%s raw_regions=%d raw_bytes=%d fa_groups=%s",
                d_req_id,
                len(lengths),
                sum(lengths),
                sorted(self.compression.fa_group_indices),
            )

            fa_counts = {
                len(remote_ids[index]) for index in self.compression.fa_group_indices
            }
            if len(fa_counts) != 1:
                raise ValueError(
                    "All compressed Full Attention groups must have equal block counts."
                )
            logical_count = fa_counts.pop() if fa_counts else 0
            total_chunks = _compression_total_chunks(
                logical_count, self.compression.config.logical_blocks_per_chunk
            )
            session = ProducerSession(
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                send_meta=send_meta,
                local_block_ids=local_ids,
                remote_block_ids=remote_ids,
                remote_session=remote_session,
                logical_block_count=logical_count,
                state=SessionState.ACTIVE,
                total_chunks=total_chunks,
                started_at=time.perf_counter(),
            )
            self._compression_producer_sessions[key] = session
            send_target_claimed = False
            logger.info(
                "Mooncake compression session ACTIVE: d_req_id=%s "
                "transfer_id=%s logical_blocks=%d total_chunks=%d",
                d_req_id,
                transfer_id,
                logical_count,
                total_chunks,
            )
            if total_chunks == 0:
                self._complete_compression_session(identity, session)
            return CompressionSessionOpened(
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                logical_block_count=logical_count,
                total_chunks=total_chunks,
                raw_complete=True,
            )
        except Exception as exc:
            if send_target_claimed:
                assert send_meta is not None
                send_meta.sending -= 1
            # SessionOpen can race record_send_reqs().  If this worker created
            # a placeholder and the producer never announced the request,
            # remove it after the wait expires instead of stranding it forever
            # with an empty p_req_id (which timeout sweeping cannot reclaim).
            if (
                created_send_meta
                and send_meta is not None
                and self.reqs_need_send.get(transfer_id) is send_meta
                and not send_meta.ready.is_set()
                and not send_meta.p_req_id
                and send_meta.sending == 0
            ):
                self.reqs_need_send.pop(transfer_id, None)
                logger.debug(
                    "Mooncake compression removed unannounced SessionOpen "
                    "placeholder: transfer_id=%s d_req_id=%s",
                    transfer_id,
                    d_req_id,
                )
            self.compression.fatal_error = exc
            return CompressionSessionOpened(
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                logical_block_count=0,
                total_chunks=0,
                raw_complete=False,
                error=str(exc),
            )

    def _complete_compression_session(
        self, identity: bytes, session: ProducerSession
    ) -> None:
        key = (identity, session.transfer_id, session.d_req_id)
        if self._compression_producer_sessions.pop(key, None) is not session:
            return
        session.state = SessionState.COMPLETE
        if session.started_at > 0:
            self._record_phase("session_duration", session.started_at)
            session.started_at = 0.0
        self._finish_compression_send_target(session)

    def _finish_compression_send_target(self, session: ProducerSession) -> None:
        send_meta = session.send_meta
        send_meta.sending -= 1
        send_meta.sent += 1
        if (
            send_meta.sent == send_meta.need_send
            and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
        ):
            self.finished_sending_reqs.add(send_meta.p_req_id)

    def _discard_compression_session(
        self,
        key: tuple[bytes, TransferId, ReqId],
        session: ProducerSession,
    ) -> None:
        if self._compression_producer_sessions.pop(key, None) is not session:
            return
        if session.started_at > 0:
            self._record_phase("session_duration", session.started_at)
            session.started_at = 0.0
        if session.state is SessionState.CANCELLED:
            self._finish_compression_send_target(session)
        else:
            session.send_meta.sending -= 1

    def _send_compressed_blocks(
        self,
        encoded,
        encode_started: float,
        remote_session: str,
        source_payload_addr: int,
        source_aux_addr: int,
        destination_payload_addr: int,
        destination_aux_addr: int,
        payload_capacity: int,
        aux_bytes: int,
    ) -> tuple[int, int]:
        encoded.event.synchronize()
        self._record_phase("compression_encode_duration", encode_started)
        codec_bytes = encoded.codec_bytes
        if codec_bytes > payload_capacity:
            raise MemoryError(
                f"Compressed payload {codec_bytes} exceeds destination capacity "
                f"{payload_capacity}."
            )
        result = self._send_blocks(
            remote_session,
            [source_payload_addr, source_aux_addr],
            [destination_payload_addr, destination_aux_addr],
            [codec_bytes, aux_bytes],
            phase="compressed_transfer_duration",
        )
        return result, codec_bytes

    async def _serve_compression_chunk(
        self,
        identity: bytes,
        request: CompressionChunkPull,
    ) -> CompressionChunkReady:
        assert self.compression is not None
        key = (identity, request.transfer_id, request.d_req_id)
        session = self._compression_producer_sessions.get(key)
        if session is None:
            return CompressionChunkReady(
                transfer_id=request.transfer_id,
                d_req_id=request.d_req_id,
                chunk_id=request.chunk_id,
                kv_part=request.kv_part,
                slot_id=request.slot_id,
                logical_block_count=request.logical_block_count,
                u8_bytes=0,
                codec_bytes=0,
                aux_bytes=0,
                error="Unknown or completed compression session.",
            )
        sender_slot = -1
        counted_inflight = False
        try:
            if session.state is not SessionState.ACTIVE:
                raise RuntimeError(f"Compression session is {session.state.name}.")
            logical_start, logical_block_count, kv_part = _compression_chunk_window(
                request.chunk_id,
                session.logical_block_count,
                self.compression.config.logical_blocks_per_chunk,
            )
            if (
                request.logical_start != logical_start
                or request.logical_block_count != logical_block_count
                or request.kv_part != kv_part
            ):
                raise ValueError(
                    "Compression chunk metadata does not match its configured window."
                )
            if (
                request.chunk_id in session.inflight_chunk_ids
                or request.chunk_id in session.completed_chunk_ids
            ):
                raise ValueError(
                    f"Compression chunk {request.chunk_id} was already requested."
                )
            session.inflight_chunks += 1
            session.inflight_chunk_ids.add(request.chunk_id)
            counted_inflight = True
            assert self._compression_sender_slots is not None
            sender_slot = await self._compression_sender_slots.get()
            local_chunk_ids = [
                group[
                    request.logical_start : request.logical_start
                    + request.logical_block_count
                ]
                if index in self.compression.fa_group_indices
                else []
                for index, group in enumerate(session.local_block_ids)
            ]
            encode_started = time.perf_counter()
            encoded = self.compression.encode(
                sender_slot, local_chunk_ids, request.kv_part
            )
            slot = self.compression.slots[sender_slot]
            aux_bytes = self.compression.layout.aux_bytes_by_part[request.kv_part]
            if aux_bytes > request.aux_capacity:
                raise MemoryError("Compression Aux payload exceeds slot capacity.")
            result, codec_bytes = await self.sender_loop.run_in_executor(
                self._sender_executor,
                self._send_compressed_blocks,
                encoded,
                encode_started,
                session.remote_session,
                slot.arena_a.data_ptr(),
                slot.aux.data_ptr(),
                request.payload_addr,
                request.aux_addr,
                request.payload_capacity,
                aux_bytes,
            )
            if result != 0:
                raise RuntimeError(f"Mooncake compressed transfer returned {result}.")
            if _compression_trace_enabled():
                logger.debug(
                    "Mooncake compression payload sent: chunk=%d kv_part=%d "
                    "codec_bytes=%d payload_sum=%d aux_sum=%d aux_sha256=%s",
                    request.chunk_id,
                    request.kv_part,
                    codec_bytes,
                    int(slot.arena_a[:codec_bytes].sum().item()),
                    int(slot.aux[:aux_bytes].sum().item()),
                    _compression_trace_digest(slot.aux, aux_bytes),
                )
            u8_bytes = self.compression.layout.u8_bytes_by_part[request.kv_part]
            if codec_bytes > u8_bytes * self.compression.config.codec_warn_ratio:
                logger.warning(
                    "Mooncake ANS output is %.2f%% of quantized input "
                    "(%d/%d bytes); continuing without fallback.",
                    100.0 * codec_bytes / u8_bytes,
                    codec_bytes,
                    u8_bytes,
                )
            logger.debug(
                "Mooncake compression chunk sent: d_req_id=%s transfer_id=%s "
                "chunk=%d kv_part=%d logical_start=%d slot=%d u8_bytes=%d "
                "codec_bytes=%d aux_bytes=%d completed=%d/%d",
                request.d_req_id,
                request.transfer_id,
                request.chunk_id,
                request.kv_part,
                request.logical_start,
                sender_slot,
                u8_bytes,
                codec_bytes,
                aux_bytes,
                session.completed_chunks,
                session.total_chunks,
            )
            session.completed_chunks += 1
            session.completed_chunk_ids.add(request.chunk_id)
            if (
                session.state is SessionState.ACTIVE
                and session.completed_chunks == session.total_chunks
            ):
                self._complete_compression_session(identity, session)
            return CompressionChunkReady(
                transfer_id=request.transfer_id,
                d_req_id=request.d_req_id,
                chunk_id=request.chunk_id,
                kv_part=request.kv_part,
                slot_id=request.slot_id,
                logical_block_count=request.logical_block_count,
                u8_bytes=u8_bytes,
                codec_bytes=codec_bytes,
                aux_bytes=aux_bytes,
            )
        except Exception as exc:
            session.state = SessionState.FAILED
            session.error = str(exc)
            self.compression.fatal_error = exc
            return CompressionChunkReady(
                transfer_id=request.transfer_id,
                d_req_id=request.d_req_id,
                chunk_id=request.chunk_id,
                kv_part=request.kv_part,
                slot_id=request.slot_id,
                logical_block_count=request.logical_block_count,
                u8_bytes=0,
                codec_bytes=0,
                aux_bytes=0,
                error=str(exc),
            )
        finally:
            if sender_slot >= 0:
                assert self._compression_sender_slots is not None
                self._compression_sender_slots.put_nowait(sender_slot)
            if counted_inflight:
                session.inflight_chunks -= 1
                session.inflight_chunk_ids.discard(request.chunk_id)
            if session.inflight_chunks == 0 and session.state in (
                SessionState.CANCELLED,
                SessionState.FAILED,
            ):
                self._discard_compression_session(key, session)

    "———————————————— depth: 3, src: _sender_worker() ————————————————"

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr,
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
            self.registered_layer_names,
            self.registered_layer_indices,
            self.registered_group_indices,
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
        )
        local_regions, remote_regions, align_err = _align_transfer_regions(
            local_regions, remote_regions
        )
        if align_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=align_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=local_regions,
            remote_regions=remote_regions,
            local_tp_size=self.tp_size,
            remote_tp_size=meta.remote_tp_size,
            # INLINED FROM _producer_cache_is_replicated().
            producer_cache_replicated=(self.transfer_topo.local_replicates_kv_cache),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
            ) = await self._build_transfer_params(
                ready_reqs,
                meta,
                local_regions,
                remote_regions,
            )
            err_req_set = set(err_reqs)
            ok_ready_reqs = [
                (d_req_id, send_meta)
                for d_req_id, send_meta in ready_reqs
                if d_req_id not in err_req_set
            ]

            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                    err_msg = (
                        transfer_err_msg
                        if err_msg is None
                        else f"{err_msg}; {transfer_err_msg}"
                    )
                    err_reqs = list(err_reqs)
                    for d_req_id, _ in ok_ready_reqs:
                        err_reqs.append(d_req_id)
                        err_req_set.add(d_req_id)
                    ok_ready_reqs = []

            for d_req_id, send_meta in ready_reqs:
                send_meta.sending -= 1

                if d_req_id in err_req_set:
                    continue

                send_meta.sent += 1
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    "———————————————— depth: 4, src: send_kv_to_decode() ————————————————"

    def _get_transfer_regions(
        self,
        base_addrs: list[int],
        block_lens: list[int],
        kv_block_lens: list[int],
        layer_names: list[str],
        layer_indices: list[int],
        group_indices: list[int] | None = None,
    ) -> list[TransferRegion]:
        if not group_indices:
            group_indices = [
                self._layer_group_indices.get(layer_name, 0)
                for layer_name in layer_names
            ]
        split_kv_regions = None
        if self.transfer_topo.virtually_split_kv_in_blocks:
            split_kv_regions = [
                not isinstance(
                    self._layer_specs[layer_name],
                    (MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec),
                )
                for layer_name in layer_names
            ]
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            kv_block_lens=kv_block_lens,
            layer_names=layer_names,
            layer_indices=layer_indices,
            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks,
            group_indices=group_indices,
            split_kv_regions=split_kv_regions,
        )

    "———————————————— depth: 4, src: send_kv_to_decode() ————————————————"

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]  # ids per group

            if not remote_block_ids or all(len(g) == 0 for g in remote_block_ids):
                continue

            if len(send_meta.local_block_ids) != len(remote_block_ids):
                logger.error(
                    "req %s: KV group count mismatch: local=%d, remote=%d",
                    d_req_id,
                    len(send_meta.local_block_ids),
                    len(remote_block_ids),
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "KV group count mismatch"
                continue

            # Keep KV-cache group identity. Hybrid/HMA groups can carry
            # different semantics (e.g. full-attention KV pages vs GDN/Mamba
            # inner-state slots), so their block IDs must not be flattened and
            # reused for every registered region.
            local_block_ids_by_group: list[list[int]] = []
            remote_block_ids_by_group: list[list[int]] = []
            has_block_error = False
            group_specs = self.kv_cache_config.kv_cache_groups
            for group_index, (local_group, remote_group) in enumerate(
                zip(send_meta.local_block_ids, remote_block_ids)
            ):
                is_mamba_group = isinstance(
                    group_specs[group_index].kv_cache_spec,
                    MambaSpec,
                )
                if is_mamba_group:
                    # Mamba/GDN prefix caching can use null blocks only as
                    # align-mode placeholders. They do not carry transferable
                    # state, so skip them on both producer and consumer sides.
                    local_group = [
                        block_id
                        for block_id in local_group
                        if block_id != NULL_BLOCK_ID
                    ]
                    remote_group = [
                        block_id
                        for block_id in remote_group
                        if block_id != NULL_BLOCK_ID
                    ]

                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    logger.error(
                        "req %s: local blocks(%d) < remote blocks(%d) "
                        "in a KV cache group (is_mamba_group=%s)",
                        d_req_id,
                        n_local,
                        n_remote,
                        is_mamba_group,
                    )
                    has_block_error = True
                    break
                elif n_local > n_remote:
                    # Partial prefix cache hit: just read uncomputed blocks.
                    local_group = local_group[-n_remote:] if n_remote > 0 else []
                local_block_ids_by_group.append(local_group)
                remote_block_ids_by_group.append(remote_group)

            if has_block_error:
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "P num blocks less than D"
                continue

            if not any(local_block_ids_by_group):
                continue

            local_block_ids_by_group = self._logical_to_kernel_block_ids(
                local_block_ids_by_group
            )
            remote_block_ids_by_group = self._logical_to_kernel_block_ids(
                remote_block_ids_by_group
            )

            for local_region, remote_region in zip(local_regions, remote_regions):
                assert local_region.group_index == remote_region.group_index, (
                    "Aligned Mooncake transfer regions must belong to the same "
                    "KV group."
                )
                group_index = local_region.group_index
                assert group_index < len(local_block_ids_by_group), (
                    "Transfer region references a missing KV group."
                )
                local_block_ids = local_block_ids_by_group[group_index]
                remote_block_ids = remote_block_ids_by_group[group_index]
                if not local_block_ids:
                    continue

                # Group by indices within this region's KV-cache group only.
                # INLINED FROM group_concurrent_contiguous().
                breaks = (
                    np.where(
                        (np.diff(local_block_ids) != 1)
                        | (np.diff(remote_block_ids) != 1)
                    )[0]
                    + 1
                )
                group_local_block_ids = [
                    group.tolist() for group in np.split(local_block_ids, breaks)
                ]
                group_remote_block_ids = [
                    group.tolist() for group in np.split(remote_block_ids, breaks)
                ]

                (
                    should_transfer,
                    src_region_offset,
                    dst_region_offset,
                    transfer_len,
                    # INLINED FROM _get_sender_transfer_plan().
                ) = _compute_sender_transfer_plan(
                    local_tp_rank=self.tp_rank,
                    local_tp_size=self.tp_size,
                    local_kv_block_len=local_region.kv_block_len,
                    remote_kv_block_len=remote_region.kv_block_len,
                    remote_tp_rank=agent_meta.remote_tp_rank,
                    remote_tp_size=agent_meta.remote_tp_size,
                    # INLINED FROM _producer_cache_is_replicated().
                    producer_cache_replicated=(
                        self.transfer_topo.local_replicates_kv_cache
                    ),
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Destination transfer region exceeds remote KV block size."
                )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                # INLINED FROM _can_coalesce_block_transfers().
                can_coalesce = (
                    src_region_offset == 0
                    and dst_region_offset == 0
                    and transfer_len == local_region.block_len
                    and transfer_len == remote_region.block_len
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len * len(group_local_block_id))
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                sum(len(group) for group in local_block_ids_by_group),
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg

    "———————————————— depth: 4, src: send_kv_to_decode() ————————————————"

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        phase: str | None = None,
    ) -> int:
        if not (len(src_ptrs) == len(dst_ptrs) == len(lengths)):
            raise ValueError(
                "Mooncake raw transfer requires matching source, destination, "
                "and length descriptor counts."
            )

        start_time = time.perf_counter()
        ret_value = 0
        batch_start = 0
        total_descriptors = len(lengths)
        completed_batches = 0
        while batch_start < total_descriptors:
            batch_end = batch_start
            batch_bytes = 0
            while batch_end < total_descriptors:
                length = lengths[batch_end]
                if length < 0:
                    raise ValueError("Mooncake raw transfer lengths cannot be negative.")
                exceeds_descriptors = (
                    batch_end - batch_start >= _RAW_TRANSFER_MAX_DESCRIPTORS
                )
                exceeds_bytes = (
                    batch_end > batch_start
                    and batch_bytes + length > _RAW_TRANSFER_MAX_BYTES
                )
                if exceeds_descriptors or exceeds_bytes:
                    break
                batch_bytes += length
                batch_end += 1

            ret_value = self.engine.batch_transfer_sync_write(
                remote_session,
                src_ptrs[batch_start:batch_end],
                dst_ptrs[batch_start:batch_end],
                lengths[batch_start:batch_end],
            )
            completed_batches += 1
            if ret_value != 0:
                break
            batch_start = batch_end

        duration = time.perf_counter() - start_time
        if phase is not None and getattr(self, "_timing_enabled", False):
            self.xfer_stats.record_phase(phase, duration)
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=sum(lengths),
                num_descs=len(src_ptrs),
            )
            logger.debug("Sending to %s done, took %s", remote_session, duration)
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Sending to %s failed (ret=%s) after %s "
                "(failed batch=%d, %d descriptors, %d bytes)",
                remote_session,
                ret_value,
                duration,
                completed_batches,
                len(src_ptrs),
                sum(lengths),
            )
        return ret_value

    "———————————————— depth: 5, src: _build_transfer_params() ————————————————"

    def _logical_to_kernel_block_ids(
        self, block_ids: list[list[int]]
    ) -> list[list[int]]:
        # For example, if a 544-token logical block is served by 32-token
        # FA kernel blocks, FA block id k expands to [17k, ..., 17k + 16],
        # while the matching Mamba/GDN state block remains k. Only attention
        # groups need logical block ids expanded to kernel block ids; Mamba/GDN
        # state block ids stay in the logical/page-id space.
        if self._physical_blocks_per_logical_kv_block == 1:
            return block_ids

        block_arange = np.arange(self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical_kv_block,
                block_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]

    r"""
    ———————————————————————————————————————————————————————————————————————————————————————————————————
                                            VLLM CONNECTOR API
                           _             _      _                 _    _
                       ___| |_ __ _ _ __| |_   | | ___   __ _  __| |  | | ____   __
                      / __| __/ _` | '__| __|  | |/ _ \ / _` |/ _` |  | |/ /\ \ / /
                      \__ \ || (_| | |  | |_   | | (_) | (_| | (_| |  |   <  \ V /
                      |___/\__\__,_|_|   \__|  |_|\___/ \__,_|\__,_|  |_|\_\  \_/

    ———————————————————————————————————————————————————————————————————————————————————————————————————
    """

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    "———————————————— depth: 1, src: start_load_kv() ————————————————"

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    "———————————————— depth: 1, src: start_load_kv() ————————————————"

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    "———————————————— depth: 2, src: _start_load_kv() ————————————————"

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    "———————————————— depth: 3(2), src: _start_load_kv() / handle_new_engine_id() ————————————————"

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )
        worker_addrs: list[str] = []
        selected_remote_pp: dict[int, list[int]] = {}
        for remote_tp_rank in remote_tp_ranks:
            pp_to_addr = self._remote_agents[remote_engine_id][remote_tp_rank]
            if self.pp_size == len(pp_to_addr) and self.pp_rank in pp_to_addr:
                pp_ranks = [self.pp_rank]
            else:
                pp_ranks = sorted(pp_to_addr)
            selected_remote_pp[remote_tp_rank] = pp_ranks
            worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)

        count = len(worker_addrs)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s "
            "and PP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
            selected_remote_pp,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        logger.info("Mooncake receive_kv: worker_addrs=%s", worker_addrs)
        for worker_addr in worker_addrs:
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    "———————————————— depth: 3, src: handle_new_engine_id() ————————————————"

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    "———————————————— depth: 4, src: receive_kv() ————————————————"

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        if self.compression is not None:
            await self._receive_compressed_kv_from_single_worker(
                worker_addr, pull_metas
            )
            return
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            registered_layer_names=self.registered_layer_names,
            registered_layer_indices=self.registered_layer_indices,
            registered_group_indices=self.registered_group_indices,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        self.xfer_stats.record_failed_recv()
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self.xfer_stats.record_failed_recv()
            return

    async def _receive_compressed_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ) -> None:
        for pull_meta in pull_metas.values():
            await self._receive_compressed_request(worker_addr, pull_meta)

    @staticmethod
    def _wait_compression_event(event: torch.cuda.Event, device_id: int) -> None:
        current_platform.set_device(device_id)
        event.synchronize()

    async def _receive_compressed_request(
        self,
        worker_addr: str,
        pull_meta: PullReqMeta,
    ) -> None:
        assert self.compression is not None
        session_started = time.perf_counter()
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                pull_meta.d_req_id: (
                    pull_meta.transfer_id,
                    pull_meta.local_block_ids,
                )
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            registered_layer_names=self.registered_layer_names,
            registered_layer_indices=self.registered_layer_indices,
            registered_group_indices=self.registered_group_indices,
        )
        open_request = CompressionSessionOpen(
            metadata=metadata,
            handshake=self.compression.handshake(),
        )
        logger.info(
            "Mooncake compression receiver opening session: d_req_id=%s "
            "transfer_id=%s producer=%s block_counts=%s",
            pull_meta.d_req_id,
            pull_meta.transfer_id,
            worker_addr,
            [len(group) for group in pull_meta.local_block_ids],
        )
        session: ConsumerSession | None = None
        network_inflight: dict[int, int] = {}
        writeback_tasks: dict[asyncio.Task, tuple[int, int, float]] = {}
        recv_task: asyncio.Task | None = None

        async def cleanup_consumer_resources() -> None:
            """Drain async/GPU work and return every slot owned by this session."""
            nonlocal recv_task
            cleanup_started = time.perf_counter()
            if recv_task is not None:
                recv_task.cancel()
                await asyncio.gather(recv_task, return_exceptions=True)
                recv_task = None

            # A writeback task wraps a CUDA event wait.  Do not cancel it and
            # immediately recycle its slot: the underlying GPU write may still
            # be in flight.  Waiting here makes slot reuse safe on failure.
            if writeback_tasks:
                await asyncio.gather(*writeback_tasks.keys(), return_exceptions=True)

            released: list[int] = []
            if session is not None and self._compression_receiver_slots is not None:
                for slot in session.slots:
                    # session.slots contains per-session state objects for all
                    # slot IDs; only non-FREE entries are owned by this session.
                    if slot.state is ReceiveSlotState.FREE and slot.chunk_id is None:
                        continue
                    slot.state = ReceiveSlotState.FREE
                    slot.chunk_id = None
                    self._compression_receiver_slots.put_nowait(slot.slot_id)
                    released.append(slot.slot_id)
            network_inflight.clear()
            writeback_tasks.clear()
            if released:
                logger.debug(
                    "Mooncake compression receiver cleanup: d_req_id=%s "
                    "transfer_id=%s released_slots=%s receiver_slots=%d",
                    getattr(session, "d_req_id", None),
                    getattr(session, "transfer_id", None),
                    released,
                    self._compression_receiver_slots.qsize()
                    if self._compression_receiver_slots is not None
                    else -1,
                )
            self._record_phase("cleanup_duration", cleanup_started)

        try:
            with make_zmq_socket(
                self.async_zmq_ctx,
                worker_addr,
                zmq.DEALER,
                bind=False,
                identity=(
                    f"vllm-mooncake-{self.tp_rank}-{time.monotonic_ns()}".encode()
                ),
                linger=0,
            ) as sock:
                socket_identity = sock.getsockopt(zmq.IDENTITY).hex()
                logger.debug(
                    "Mooncake compression D socket opened: d_req_id=%s "
                    "transfer_id=%s peer=%s identity=%s",
                    pull_meta.d_req_id,
                    pull_meta.transfer_id,
                    worker_addr,
                    socket_identity,
                )
                sock.setsockopt(
                    zmq.RCVTIMEO,
                    (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000,
                )
                logger.debug(
                    "Mooncake compression SessionOpen send: d_req_id=%s "
                    "transfer_id=%s identity=%s",
                    pull_meta.d_req_id,
                    pull_meta.transfer_id,
                    socket_identity,
                )
                await sock.send(self._encoder.encode(open_request))
                opened = self._compression_resp_decoder.decode(await sock.recv())
                logger.debug(
                    "Mooncake compression SessionOpen recv: d_req_id=%s "
                    "transfer_id=%s identity=%s response=%s error=%s",
                    pull_meta.d_req_id,
                    pull_meta.transfer_id,
                    socket_identity,
                    type(opened).__name__,
                    getattr(opened, "error", None),
                )
                if not isinstance(opened, CompressionSessionOpened):
                    raise RuntimeError("Expected CompressionSessionOpened response.")
                if opened.error:
                    raise RuntimeError(opened.error)
                if not opened.raw_complete:
                    raise RuntimeError("Raw GDN transfer did not complete.")
                expected_chunks = _compression_total_chunks(
                    opened.logical_block_count,
                    self.compression.config.logical_blocks_per_chunk,
                )
                if opened.total_chunks != expected_chunks:
                    raise RuntimeError(
                        "Producer returned an invalid compression chunk count."
                    )
                logger.info(
                    "Mooncake compression session OPEN: d_req_id=%s "
                    "transfer_id=%s logical_blocks=%d total_chunks=%d raw_complete=%s",
                    pull_meta.d_req_id,
                    pull_meta.transfer_id,
                    opened.logical_block_count,
                    opened.total_chunks,
                    opened.raw_complete,
                )

                session = ConsumerSession(
                    transfer_id=pull_meta.transfer_id,
                    d_req_id=pull_meta.d_req_id,
                    block_ids=pull_meta.local_block_ids,
                    logical_block_count=opened.logical_block_count,
                    total_chunks=opened.total_chunks,
                    state=SessionState.ACTIVE,
                    slots=[
                        ReceiveSlot(slot_id=index)
                        for index in range(self.compression.config.slot_count)
                    ],
                )
                next_chunk = 0

                async def recv_response() -> bytes:
                    return await sock.recv()

                async def issue_chunk(chunk_id: int) -> None:
                    assert self._compression_receiver_slots is not None
                    slot_id = await self._compression_receiver_slots.get()
                    slot = session.slots[slot_id]
                    logical_start, logical_block_count, kv_part = (
                        _compression_chunk_window(
                            chunk_id,
                            session.logical_block_count,
                            self.compression.config.logical_blocks_per_chunk,
                        )
                    )
                    payload_addr, payload_capacity, aux_addr, aux_capacity = (
                        self.compression.slot_addresses(slot_id)
                    )
                    try:
                        slot.transition(ReceiveSlotState.FREE, ReceiveSlotState.GRANTED)
                        slot.chunk_id = chunk_id
                        request = CompressionChunkPull(
                            transfer_id=session.transfer_id,
                            d_req_id=session.d_req_id,
                            chunk_id=chunk_id,
                            kv_part=kv_part,
                            logical_start=logical_start,
                            logical_block_count=logical_block_count,
                            slot_id=slot_id,
                            payload_addr=payload_addr,
                            payload_capacity=payload_capacity,
                            aux_addr=aux_addr,
                            aux_capacity=aux_capacity,
                        )
                        logger.debug(
                            "Mooncake compression ChunkPull send: d_req_id=%s "
                            "transfer_id=%s identity=%s chunk=%d slot=%d",
                            session.d_req_id,
                            session.transfer_id,
                            socket_identity,
                            chunk_id,
                            slot_id,
                        )
                        await sock.send(self._encoder.encode(request))
                        network_inflight[chunk_id] = slot_id
                        logger.debug(
                            "Mooncake compression chunk pull: d_req_id=%s "
                            "transfer_id=%s chunk=%d kv_part=%d slot=%d "
                            "payload_capacity=%d aux_capacity=%d",
                            session.d_req_id,
                            session.transfer_id,
                            chunk_id,
                            kv_part,
                            slot_id,
                            payload_capacity,
                            aux_capacity,
                        )
                    except BaseException:
                        slot.state = ReceiveSlotState.FREE
                        slot.chunk_id = None
                        self._compression_receiver_slots.put_nowait(slot_id)
                        raise

                # Always enter the receive loop after the first chunk.  With
                # several sessions sharing the process-wide slot queue, waiting
                # synchronously for the whole initial window can deadlock: two
                # sessions may each hold one slot while both await the other.
                if next_chunk < session.total_chunks:
                    await issue_chunk(next_chunk)
                    next_chunk += 1
                for _ in range(1, len(session.slots)):
                    if next_chunk >= session.total_chunks:
                        break
                    assert self._compression_receiver_slots is not None
                    if self._compression_receiver_slots.empty():
                        logger.debug(
                            "Mooncake compression initial window limited: "
                            "d_req_id=%s transfer_id=%s next_chunk=%d "
                            "global_slots=0",
                            session.d_req_id,
                            session.transfer_id,
                            next_chunk,
                        )
                        break
                    await issue_chunk(next_chunk)
                    next_chunk += 1

                while session.completed_chunks < session.total_chunks:
                    if network_inflight and recv_task is None:
                        recv_task = asyncio.create_task(recv_response())
                    wait_set: set[asyncio.Task] = set(writeback_tasks)
                    if recv_task is not None:
                        wait_set.add(recv_task)
                    if not wait_set:
                        raise RuntimeError("Compression session made no progress.")
                    done, _ = await asyncio.wait(
                        wait_set, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        if task is recv_task:
                            recv_task = None
                            ready = self._compression_resp_decoder.decode(task.result())
                            logger.debug(
                                "Mooncake compression socket recv: d_req_id=%s "
                                "transfer_id=%s identity=%s response=%s "
                                "chunk=%s slot=%s",
                                session.d_req_id,
                                session.transfer_id,
                                socket_identity,
                                type(ready).__name__,
                                getattr(ready, "chunk_id", None),
                                getattr(ready, "slot_id", None),
                            )
                            logger.debug(
                                "Mooncake compression response received: "
                                "d_req_id=%s transfer_id=%s type=%s "
                                "chunk=%s slot=%s error=%s",
                                session.d_req_id,
                                session.transfer_id,
                                type(ready).__name__,
                                getattr(ready, "chunk_id", None),
                                getattr(ready, "slot_id", None),
                                getattr(ready, "error", None),
                            )
                            if not isinstance(ready, CompressionChunkReady):
                                raise RuntimeError(
                                    "Expected CompressionChunkReady response."
                                )
                            if ready.error:
                                raise RuntimeError(ready.error)
                            if (
                                ready.transfer_id != session.transfer_id
                                or ready.d_req_id != session.d_req_id
                            ):
                                raise RuntimeError(
                                    "Compression response session mismatch."
                                )
                            expected_slot = network_inflight.pop(ready.chunk_id)
                            if ready.slot_id != expected_slot:
                                raise RuntimeError(
                                    "Compression response slot mismatch."
                                )
                            if ready.kv_part != ready.chunk_id % 2:
                                raise RuntimeError(
                                    "Compression response KV part mismatch."
                                )
                            logical_start, logical_block_count, _ = (
                                _compression_chunk_window(
                                    ready.chunk_id,
                                    session.logical_block_count,
                                    self.compression.config.logical_blocks_per_chunk,
                                )
                            )
                            if ready.logical_block_count != logical_block_count:
                                raise RuntimeError(
                                    "Compression response block count mismatch."
                                )
                            expected_aux = self.compression.layout.aux_bytes_by_part[
                                ready.kv_part
                            ]
                            if ready.aux_bytes != expected_aux:
                                raise RuntimeError(
                                    "Compression response Aux length mismatch."
                                )
                            if not (
                                0
                                < ready.codec_bytes
                                <= self.compression.layout.arena_bytes
                            ):
                                raise RuntimeError(
                                    "Compression response codec length is invalid."
                                )
                            slot = session.slots[ready.slot_id]
                            slot.transition(
                                ReceiveSlotState.GRANTED,
                                ReceiveSlotState.RECEIVED,
                            )
                            if _compression_trace_enabled():
                                receiver_slot = self.compression.slots[ready.slot_id]
                                logger.debug(
                                    "Mooncake compression payload received: "
                                    "chunk=%d kv_part=%d codec_bytes=%d "
                                    "payload_sum=%d aux_sum=%d aux_sha256=%s",
                                    ready.chunk_id,
                                    ready.kv_part,
                                    ready.codec_bytes,
                                    int(
                                        receiver_slot.arena_a[: ready.codec_bytes]
                                        .sum()
                                        .item()
                                    ),
                                    int(
                                        receiver_slot.aux[: ready.aux_bytes]
                                        .sum()
                                        .item()
                                    ),
                                    _compression_trace_digest(
                                        receiver_slot.aux, ready.aux_bytes
                                    ),
                                )
                            logical_ids_by_group = [
                                group[
                                    logical_start : logical_start
                                    + ready.logical_block_count
                                ]
                                if index in self.compression.fa_group_indices
                                else []
                                for index, group in enumerate(session.block_ids)
                            ]
                            slot.transition(
                                ReceiveSlotState.RECEIVED,
                                ReceiveSlotState.DECODING,
                            )
                            decode_started = time.perf_counter()
                            event = self.compression.decode(
                                ready.slot_id,
                                logical_ids_by_group,
                                ready.kv_part,
                                ready.codec_bytes,
                                ready.u8_bytes,
                            )
                            slot.transition(
                                ReceiveSlotState.DECODING,
                                ReceiveSlotState.WRITEBACK,
                            )
                            writeback = asyncio.create_task(
                                asyncio.to_thread(
                                    self._wait_compression_event,
                                    event,
                                    self.device_id,
                                )
                            )
                            writeback_tasks[writeback] = (
                                ready.slot_id,
                                ready.chunk_id,
                                decode_started,
                            )
                            logger.debug(
                                "Mooncake compression chunk received: "
                                "d_req_id=%s transfer_id=%s chunk=%d slot=%d "
                                "codec_bytes=%d aux_bytes=%d",
                                session.d_req_id,
                                session.transfer_id,
                                ready.chunk_id,
                                ready.slot_id,
                                ready.codec_bytes,
                                ready.aux_bytes,
                            )
                        else:
                            slot_id, chunk_id, decode_started = writeback_tasks.pop(task)
                            task.result()
                            # The decode event is recorded after inverse
                            # transform and KV-cache writes, so this duration
                            # intentionally represents the complete GPU
                            # decode/writeback phase for the current pipeline.
                            self._record_phase(
                                "compression_decode_duration", decode_started
                            )
                            slot = session.slots[slot_id]
                            slot.transition(
                                ReceiveSlotState.WRITEBACK,
                                ReceiveSlotState.FREE,
                            )
                            slot.chunk_id = None
                            assert self._compression_receiver_slots is not None
                            self._compression_receiver_slots.put_nowait(slot_id)
                            session.completed_chunks += 1
                            logger.debug(
                                "Mooncake compression writeback complete: "
                                "d_req_id=%s transfer_id=%s chunk=%d "
                                "completed=%d/%d",
                                session.d_req_id,
                                session.transfer_id,
                                chunk_id,
                                session.completed_chunks,
                                session.total_chunks,
                            )
                            if next_chunk < session.total_chunks:
                                await issue_chunk(next_chunk)
                                next_chunk += 1

                session.state = SessionState.COMPLETE
                logger.info(
                    "Mooncake compression session COMPLETE: d_req_id=%s "
                    "transfer_id=%s chunks=%d",
                    session.d_req_id,
                    session.transfer_id,
                    session.completed_chunks,
                )
                pull_meta.pull_tasks_count -= 1
                if pull_meta.pull_tasks_count == 0:
                    self.finished_recving_reqs.add(pull_meta.d_req_id)
        except asyncio.CancelledError:
            if session is not None:
                session.state = SessionState.CANCELLED
            await cleanup_consumer_resources()
            if session is not None:
                await self._send_compression_abort(worker_addr, session)
            raise
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated during compressed receive.")
        except Exception as exc:
            if session is not None:
                session.state = SessionState.FAILED
                session.error = str(exc)
                await cleanup_consumer_resources()
                await self._send_compression_abort(worker_addr, session)
            self.compression.fatal_error = exc
            self.xfer_stats.record_failed_recv()
            logger.error(
                "Mooncake compressed receive failed for %s: %s",
                pull_meta.d_req_id,
                exc,
            )
        finally:
            self._record_phase("session_duration", session_started)

    async def _send_compression_abort(
        self, worker_addr: str, session: ConsumerSession
    ) -> None:
        request = CompressionAbort(
            transfer_id=session.transfer_id,
            d_req_id=session.d_req_id,
            reason=session.error or session.state.name,
        )
        try:
            with make_zmq_socket(
                self.async_zmq_ctx,
                worker_addr,
                zmq.DEALER,
                bind=False,
                linger=0,
            ) as sock:
                await sock.send(self._encoder.encode(request))
        except (zmq.ContextTerminated, zmq.ZMQError):
            pass

    "———————————————— depth: 5, src: receive_kv_from_single_worker() ————————————————"

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    r"""
    ————————————————————————————————————————————————————————————————————————————————————————————
                                         VLLM CONNECTOR API
                                _       ___   _         _      _               _
                      __ _  ___| |_    /  _| (_) _ __  (_) ___| |__   ___   __| |
                     / _` |/ _ \ __|   | |_  | || '_ \ | |/ __| '_ \ / _ \ / _` |
                    | (_| |  __/ |_    |  _| | || | | || |\__ \ | | |  __/| (_| |
                     \__, |\___|\__|   |_|   |_||_| |_||_||___/_| |_|\___| \__,_|
                     |___/
    ————————————————————————————————————————————————————————————————————————————————————————————
    """

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        if self.compression is not None:
            self.compression.raise_if_failed()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Mooncake compression health: tp_rank=%d producer_sessions=%d "
                    "reqs_need_send=%d sender_slots=%s receiver_slots=%s "
                    "finished_send=%d finished_recv=%d",
                    self.tp_rank,
                    len(self._compression_producer_sessions),
                    len(self.reqs_need_send),
                    self._compression_sender_slots.qsize()
                    if self._compression_sender_slots is not None
                    else None,
                    self._compression_receiver_slots.qsize()
                    if self._compression_receiver_slots is not None
                    else None,
                    len(self.finished_sending_reqs),
                    len(self.finished_recving_reqs),
                )
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )
        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )
        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()
        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )
        return finished_sending_reqs or None, finished_recving_reqs or None

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    r"""
    ——————————————————————————————————————————————————————————————————————————————————————————————————————————
                                                VLLM CONNECTOR API
                 _       _                                              _                    _        _
       __ _  ___| |_    | | ____   __    ___ ___  _ __  _ __   ___  ___| |_ ___  _ __    ___| |_ __ _| |_ ___
      / _` |/ _ \ __|   | |/ /\ \ / /   / __/ _ \| '_ \| '_ \ / _ \/ __| __/ _ \| '__|  / __| __/ _` | __/ __|
     | (_| |  __/ |_    |   <  \ V /   | (_| (_) | | | | | | |  __/ (__| || (_) | |     \__ \ || (_| | |_\__ \
      \__, |\___|\__|   |_|\_\  \_/     \___\___/|_| |_|_| |_|\___|\___|\__\___/|_|     |___/\__\__,_|\__|___/
      |___/
    ——————————————————————————————————————————————————————————————————————————————————————————————————————————
    """

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return transfer stats collected since the last call, or None
        if nothing has been recorded in this interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()


# customization by pogi
