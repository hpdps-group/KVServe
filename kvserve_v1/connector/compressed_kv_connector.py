"""CompressedKVConnector — KVConnectorBase_V1 implementation for PD separation.

Phase 1: NCCL transport, identity compression (no-op).
Phase 2: Real compression via KVCompressionAdapter (kvserve pipelines).

Key design decisions (learned from conversation log):
- LLM (sync) not AsyncLLM.
- get_num_new_matched_tokens returns (n, False) — no WAITING_FOR_REMOTE_KVS.
- start_load_kv proactively drains transport; WORKER has its own _worker_received_kv.
- NCCL recv is only called on-demand (after ZMQ signal) — safe for CUDA graph capture.
- build_connector_meta must not drop pending decode loads across steps.
- Compression: GPU-resident CompressedWire bundles sent through NCCL.
  Compressed messages are identified by a "__compressed__" sentinel in layer_names.
"""

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from kvserve_v1.compression.manager import (
    add_sentinel, is_compressed_layer_names, strip_sentinel)
from kvserve_v1.compression.wire import (
    CompressedWire, build_wire, restore_from_wire)
from kvserve_v1.utils.kv_utils import (
    extract_kv_from_layer_by_blocks, inject_kv_into_layer_by_blocks,
    make_slot_mapping)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

try:
    _LOAD_TIMEOUT_S = float(os.environ.get("KVSERVE_LOAD_TIMEOUT_S", "60"))
except ValueError:
    _LOAD_TIMEOUT_S = 60.0
_DEFAULT_MAX_NCCL_CHUNK_BYTES = 512 * 1024 * 1024


def _sorted_rids(rids: list[str] | set[str]) -> list[str]:
    return sorted(rids)


def _write_compression_stats(
    request_id: str,
    transfer_id: str,
    original_bytes: int,
    compressed_bytes: int,
) -> None:
    stats_path = os.environ.get("KVSERVE_COMPRESSION_STATS_PATH")
    if not stats_path or original_bytes <= 0 or compressed_bytes <= 0:
        return
    row = {
        "request_id": request_id,
        "transfer_id": transfer_id,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
    }
    try:
        with open(stats_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as e:
        logger.warning_once(
            "[Connector] Failed to write compression stats to %s: %s",
            stats_path, e)


def _max_nccl_chunk_bytes() -> int:
    raw = os.environ.get("KVSERVE_MAX_NCCL_CHUNK_BYTES")
    if not raw:
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning_once(
            "Invalid KVSERVE_MAX_NCCL_CHUNK_BYTES=%r; using default %d",
            raw, _DEFAULT_MAX_NCCL_CHUNK_BYTES)
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    if value <= 0:
        logger.warning_once(
            "KVSERVE_MAX_NCCL_CHUNK_BYTES must be positive; using default %d",
            _DEFAULT_MAX_NCCL_CHUNK_BYTES)
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    return value


@dataclass
class ReqMeta:
    request_id: str
    transfer_id: str
    token_ids: list[int]
    block_ids: list[int]
    slot_mapping: torch.Tensor  # CPU LongTensor [num_tokens]


@dataclass
class CompressedKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, request_id: str, transfer_id: str, token_ids: list[int],
                    block_ids: list[int], block_size: int) -> None:
        self.requests.append(ReqMeta(
            request_id=request_id,
            transfer_id=transfer_id,
            token_ids=token_ids,
            block_ids=block_ids,
            slot_mapping=make_slot_mapping(token_ids, block_ids, block_size),
        ))


class CompressedKVConnector(KVConnectorBase_V1):
    """
    KVConnectorBase_V1 implementation.
    Transport: NcclTransport (ZMQ control + NCCL data).
    Compression: configurable via kv_connector_extra_config["compression"].
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        cfg = vllm_config.kv_transfer_config
        self.is_producer = cfg.is_kv_producer
        self._block_size = vllm_config.cache_config.block_size
        self._async_send = (
            os.environ.get("KVSERVE_ASYNC_SEND", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        try:
            self._max_inflight_bytes = int(
                os.environ.get(
                    "KVSERVE_MAX_INFLIGHT_BYTES",
                    str(4 * 1024 * 1024 * 1024),
                )
            )
        except ValueError:
            self._max_inflight_bytes = 4 * 1024 * 1024 * 1024

        # SCHEDULER side: consumer tracks which requests need KV load
        self._requests_need_load: dict[str, tuple["Request", list[int]]] = {}
        # request_id -> transfer_id (shared between P/D). If not provided by
        # upstream router, falls back to request_id.
        self._request_transfer_ids: dict[str, str] = {}
        # SCHEDULER side: producer chunked-prefill accumulation state.
        # req_id -> (accumulated block_ids, full prompt_token_ids)
        self.chunked_prefill: dict[str, tuple[list[int], list[int]]] = {}

        # WORKER side: received buffers keyed by transfer_id.
        self._worker_received_kv: dict[
            str, deque[tuple[list[str], Any]]
        ] = defaultdict(deque)

        # WORKER side: producer accumulates per-layer KV before sending
        self._layer_buffers: dict[str, dict[str, torch.Tensor]] = {}

        # KV shape info — read from vllm_config so both producer and consumer have it
        self._num_kv_heads: int = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self._head_size: int = vllm_config.model_config.get_head_size()
        # Basename of the model path (e.g. "Qwen2.5-7B-Instruct") used to patch
        # the quantizer's model_name when "default" compression mode is selected.
        self._model_name: str = os.path.basename(
            vllm_config.model_config.model.rstrip("/"))
        self._tp_rank: int = 0
        self._tp_size: int = 1

        # Compression spec: None | "default" | custom-dict | controller-dict
        # Built lazily on first use to avoid import overhead at init time.
        self._compressor: Optional[Any] = None
        self._compression_cfg = cfg.kv_connector_extra_config.get("compression")

        if role == KVConnectorRole.WORKER:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
                get_world_group,
            )
            local_rank = get_world_group().local_rank
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            self._tp_rank = tp_rank
            self._tp_size = tp_size
            self._transport = self._build_transport(
                cfg, local_rank=local_rank, channel_rank=tp_rank)
            logger.info(
                "[CompressedKVConnector] WORKER init: is_producer=%s "
                "local_rank=%d tp_rank=%d tp_size=%d compression=%s",
                self.is_producer, local_rank, tp_rank, tp_size,
                "enabled" if self._compression_cfg else "disabled")
            if self.is_producer and self._async_send:
                logger.info(
                    "[CompressedKVConnector] bounded async send enabled: "
                    "max_inflight_bytes=%d",
                    self._max_inflight_bytes)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default KV cache layout.")
            return None
        if vllm_config.model_config.use_mla:
            logger.warning_once(
                "CompressedKVConnector has not validated MLA KV cache layout; "
                "falling back to vLLM default layout.")
            return None
        logger.info_once(
            "CompressedKVConnector setting KV cache layout to NHD "
            "for the validated compressed KV transfer path.")
        return "NHD"

    # ── Worker-side ────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.is_producer:
            return

        newly_recv = self._transport.drain_received()
        if newly_recv:
            logger.info(
                "[Connector][RID][RECV] drained from transport: %s",
                _sorted_rids(list(newly_recv.keys())),
            )
            for request_id, payloads in newly_recv.items():
                for payload in payloads:
                    self._worker_received_kv[request_id].append(payload)

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        expected_rids = [req_meta.request_id for req_meta in meta.requests]
        logger.info(
            "[Connector][RID][RECV] expected this step: %s; buffered: %s",
            _sorted_rids(expected_rids),
            _sorted_rids(list(self._worker_received_kv.keys())),
        )

        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_id = req_meta.transfer_id

            deadline = time.monotonic() + _LOAD_TIMEOUT_S
            while not self._worker_received_kv.get(transfer_id):
                newly_recv = self._transport.drain_received()
                if newly_recv:
                    logger.info(
                        "[Connector][RID][RECV] drained while waiting: %s",
                        _sorted_rids(list(newly_recv.keys())),
                    )
                    for key, payloads in newly_recv.items():
                        for payload in payloads:
                            self._worker_received_kv[key].append(payload)
                if self._worker_received_kv.get(transfer_id):
                    break
                if time.monotonic() > deadline:
                    logger.warning(
                        "[Connector] Timeout waiting KV for rid=%s transfer_id=%s",
                        rid, transfer_id)
                    break
                time.sleep(0.005)

            if not self._worker_received_kv.get(transfer_id):
                continue

            layer_names, payload = self._worker_received_kv[transfer_id].popleft()
            if not self._worker_received_kv[transfer_id]:
                self._worker_received_kv.pop(transfer_id, None)

            # Consumer-side failure marker (OOM, pre-INIT, etc.).
            if payload is None:
                logger.error(
                    "[Connector][RID][RECV] transport reported failure for "
                    "rid=%s transfer_id=%s; skipping injection", rid,
                    transfer_id)
                continue

            # Decompress if needed
            if is_compressed_layer_names(layer_names):
                layer_names = strip_sentinel(layer_names)
                compressor = self._get_compressor()
                if compressor is not None:
                    if not isinstance(payload, dict) or not payload.get("__bundle__"):
                        logger.error(
                            "[Connector] Invalid compressed payload for %s", rid)
                        continue
                    wire = CompressedWire(
                        meta=payload["meta"],
                        body_chunks=payload["body_chunks"],
                        aux_tensors=payload["aux_tensors"],
                    )
                    compressed = restore_from_wire(wire)
                    stacked_kv = compressor.decompress(compressed)
                    if stacked_kv is None:
                        logger.error(
                            "[Connector] Decompression failed for %s", rid)
                        continue
                else:
                    logger.error(
                        "[Connector] Received compressed KV but no compressor "
                        "configured for %s", rid)
                    continue
            else:
                stacked_kv = payload  # GPU tensor from NCCL

            for i, layer_name in enumerate(layer_names):
                layer = forward_context.no_compile_layers.get(layer_name)
                if layer is None:
                    continue
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue
                kv_cache_layer = kv_cache[forward_context.virtual_engine]
                inject_kv_into_layer_by_blocks(
                    kv_cache_layer, stacked_kv[i], req_meta.block_ids, rid)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        for req_meta in meta.requests:
            rid = req_meta.request_id
            if rid not in self._layer_buffers:
                self._layer_buffers[rid] = {}
            extracted = extract_kv_from_layer_by_blocks(kv_layer, req_meta.block_ids)
            self._layer_buffers[rid][layer_name] = extracted

    def wait_for_save(self) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        current_rids = {req_meta.request_id for req_meta in meta.requests}
        logger.info(
            "[Connector][RID][SEND] scheduled this step: %s",
            _sorted_rids(current_rids),
        )

        stale = set(self._layer_buffers) - current_rids
        if stale:
            logger.warning("[Connector] Dropping stale layer buffers: %s", stale)
            for rid in stale:
                self._layer_buffers.pop(rid, None)

        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_id = req_meta.transfer_id
            if rid not in self._layer_buffers:
                continue
            layer_kv = self._layer_buffers.pop(rid)
            if not layer_kv:
                continue

            layer_names = sorted(layer_kv.keys())
            stacked = torch.stack([layer_kv[n] for n in layer_names], dim=0)
            # stacked: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]

            compressor = self._get_compressor()
            if compressor is not None:
                t0 = time.monotonic()
                compressed = compressor.compress(stacked, rid)
                if compressed is not None:
                    wire = build_wire(compressed, _max_nccl_chunk_bytes())
                    send_names = add_sentinel(layer_names)
                    logger.info(
                        "[Connector][RID][SEND] sending compressed rid=%s transfer_id=%s "
                        "layers=%d body_chunks=%d aux_tensors=%d payload_bytes=%d",
                        rid, transfer_id, len(layer_names), len(wire.body_chunks),
                        len(wire.aux_tensors), wire.nbytes,
                    )
                    self._transport.send_bundle(
                        transfer_id, send_names, wire.meta, wire.body_chunks,
                        wire.aux_tensors)
                    _write_compression_stats(
                        request_id=rid,
                        transfer_id=transfer_id,
                        original_bytes=stacked.numel() * stacked.element_size(),
                        compressed_bytes=wire.nbytes,
                    )
                    elapsed_ms = (time.monotonic() - t0) * 1e3
                    compressor.update_controller(rid, elapsed_ms)
                    logger.debug(
                        "[Connector] Sent compressed KV for %s "
                        "(%d layers, %.2f MB → %.2f MB)",
                        rid, len(layer_names),
                        stacked.numel() * stacked.element_size() / 1e6,
                        wire.nbytes / 1e6)
                    continue
                logger.warning(
                    "[Connector] Compression returned None for %s, "
                    "falling back to raw send", rid)

            logger.info(
                "[Connector][RID][SEND] sending raw rid=%s transfer_id=%s layers=%d shape=%s",
                rid, transfer_id, len(layer_names), list(stacked.shape),
            )
            self._transport.send(transfer_id, layer_names, stacked)
            logger.debug("[Connector] Sent raw KV for %s (%d layers)",
                         rid, len(layer_names))

        if self._async_send:
            wait_s = self._transport.wait_for_below(
                self._max_inflight_bytes)
            if wait_s > 0.001:
                logger.info(
                    "[Connector] async send backpressure %.3f ms "
                    "(pending_bytes=%d limit=%d)",
                    wait_s * 1e3,
                    self._transport.pending_bytes(),
                    self._max_inflight_bytes)
        else:
            self._transport.wait_for_sent()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def shutdown(self) -> None:
        transport = getattr(self, "_transport", None)
        if self.is_producer and transport is not None:
            transport.wait_for_sent()

    # ── Scheduler-side ─────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            return 0, False
        # vLLM requires at least one local token to be scheduled for the
        # request; external KV can cover the prompt prefix before that token.
        num_external = (len(request.prompt_token_ids) - 1
                        - num_computed_tokens)
        if num_external <= 0:
            return 0, False
        return num_external, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        # Capture transfer_id as early as possible for both producer/consumer.
        self._resolve_transfer_id(request.request_id, request)
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = CompressedKVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                    new_req.req_id
                ]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                prompt_token_ids = new_req.prompt_token_ids or []
                # Chunked prefill: defer transfer until full prompt KV exists.
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], prompt_token_ids
                    )
                    continue
                meta.add_request(
                    request_id=new_req.req_id,
                    transfer_id=self._resolve_transfer_id(new_req.req_id),
                    token_ids=prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
            else:
                if new_req.req_id in self._requests_need_load:
                    meta.add_request(
                        request_id=new_req.req_id,
                        transfer_id=self._resolve_transfer_id(new_req.req_id),
                        token_ids=new_req.prompt_token_ids or [],
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                    )
                    self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                assert req_id in self.chunked_prefill
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = self.chunked_prefill[req_id][0] + block_ids
                prompt_token_ids = self.chunked_prefill[req_id][1]

                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue

                meta.add_request(
                    request_id=req_id,
                    transfer_id=self._resolve_transfer_id(req_id),
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # Resumed preempted requests are first N in cached_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                meta.add_request(
                    request_id=req_id,
                    transfer_id=self._resolve_transfer_id(req_id, request),
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        self.chunked_prefill.pop(request.request_id, None)
        self._request_transfer_ids.pop(request.request_id, None)
        if not self.is_producer:
            self._requests_need_load.pop(request.request_id, None)
        return False, None

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_compressor(self):
        """Lazily build KVCompressionAdapter on first use."""
        if self._compression_cfg is None:
            return None
        if self._compressor is not None:
            return self._compressor

        from kvserve_v1.compression.manager import KVCompressionAdapter
        self._compressor = KVCompressionAdapter(
            compression_spec=self._compression_cfg,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_size,
            model_name=self._model_name,
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
        )
        spec = self._compression_cfg
        if spec == "default":
            mode_desc = "default"
        elif isinstance(spec, dict):
            mode_desc = spec.get("mode", "custom")
        else:
            mode_desc = str(type(spec).__name__)
        logger.info(
            "[Connector] KVCompressionAdapter built: mode=%s heads=%d head_size=%d",
            mode_desc, self._num_kv_heads, self._head_size)
        return self._compressor

    def _resolve_transfer_id(
        self, request_id: str, request: "Request | None" = None
    ) -> str:
        if request is not None:
            params = getattr(request, "kv_transfer_params", None)
            if params and params.get("transfer_id"):
                self._request_transfer_ids[request_id] = str(
                    params["transfer_id"])
            else:
                logger.warning_once(
                    "Missing transfer_id in kv_transfer_params from router; "
                    "falling back to local request_id. This is only safe for "
                    "single-process tests and should not be used in PD serving.")
        return self._request_transfer_ids.get(request_id, request_id)

    @staticmethod
    def _build_transport(cfg, local_rank: int = 0, channel_rank: int = 0):
        from kvserve_v1.transport.nccl_transport import NcclTransport
        return NcclTransport(
            is_sender=cfg.is_kv_producer,
            host=cfg.kv_ip,
            port=cfg.kv_port,
            local_rank=local_rank,
            channel_rank=channel_rank,
        )
