"""NCCL data transport with ZMQ control signaling.

Architecture (mirrors P2pNcclEngine's PUT_ASYNC approach):
  Consumer (receiver): ZMQ ROUTER binds on kv_port + tp_rank
  Producer (sender):   ZMQ DEALER connects to kv_port + tp_rank

Protocol:
  Init (startup):
    Producer  → INIT{unique_id} → Consumer
    Both call ncclCommInitRank (producer=rank 0, consumer=rank 1)
    ncclCommInitRank is collective → blocks until both sides call it.

  Per-transfer (PUT mode):
    Producer  → PUT{request_id, layer_names, shape, dtype}
    Consumer allocates GPU tensor
    Consumer  → ACK b"0"
    Producer  ncclSend on send_stream  (background thread, PUT_ASYNC)
    Consumer  ncclRecv on recv_stream  (listener thread, only on-demand)
    Consumer stores tensor in _received

Key: ncclRecv is NEVER blocking in the background — it only runs after a ZMQ
signal arrives.  Between requests the recv_stream has no pending ops, so
torch.cuda.synchronize() during CUDA graph capture is not blocked.
"""

import ctypes
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

import msgpack
import torch
import zmq

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclDataTypeEnum, ncclUniqueId)
from vllm.logger import init_logger

logger = init_logger(__name__)

_NCCL_ACK_OK = b"0"
_NCCL_ACK_OOM = b"1"

# Consumer-side failure marker placed into _received when a PUT/PUT_BUNDLE is
# rejected (OOM, pre-INIT, etc.). Connector treats payload=None as "this
# transfer failed, skip without waiting for the timeout".
FAILED_PAYLOAD: Any = None
FAILED_LAYER_NAMES: list[str] = []


class NcclTransport:
    """ZMQ signaling + NCCL data transfer.

    Producer (is_sender=True): DEALER connects, ncclSend in background thread.
    Consumer (is_sender=False): ROUTER binds, listener thread handles all msgs.
    """

    def __init__(self, is_sender: bool, host: str, port: int,
                 local_rank: int = 0, channel_rank: int = 0):
        self.is_sender = is_sender
        self.local_rank = local_rank
        self.channel_rank = channel_rank
        self.port = port + channel_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.nccl = NCCLLibrary()

        self._ctx = zmq.Context()

        if is_sender:
            self._sock = self._ctx.socket(zmq.DEALER)
            self._sock.setsockopt_string(zmq.IDENTITY, f"{host}:{self.port}")
            self._sock.connect(f"tcp://{host}:{self.port}")
            logger.info(
                "[NcclTransport] Producer DEALER connected to %s:%d "
                "(local_rank=%d channel_rank=%d)",
                host, self.port, local_rank, channel_rank)

            # Get unique_id, send to consumer, then both call ncclCommInitRank
            unique_id = self.nccl.ncclGetUniqueId()
            self._sock.send(msgpack.dumps(
                {"cmd": "INIT", "unique_id": bytes(unique_id.internal)},
                use_bin_type=True))

            with torch.cuda.device(self.device):
                self._comm = self.nccl.ncclCommInitRank(2, unique_id, 0)
            logger.info("[NcclTransport] Producer NCCL comm established")

            self._send_stream = torch.cuda.Stream(device=self.device)

            # PUT_ASYNC: background thread does actual ncclSend
            self._send_queue_cv = threading.Condition()
            self._send_queue: deque = deque()
            self._send_inflight = 0
            self._send_thread = threading.Thread(
                target=self._send_loop, daemon=True, name="nccl-send")
            self._send_thread.start()

        else:
            self._sock = self._ctx.socket(zmq.ROUTER)
            self._sock.bind(f"tcp://*:{self.port}")
            logger.info(
                "[NcclTransport] Consumer ROUTER bound on port %d "
                "(local_rank=%d channel_rank=%d)",
                self.port, local_rank, channel_rank)

            self._lock = threading.Lock()
            # ZMQ "request_id" is the connector wire key (transfer_key from connector).
            # Multiple PUTs for the same key append in order; deque preserves FIFO.
            self._received: dict[
                str, deque[tuple[list[str], Any, dict[str, Any] | None]]
            ] = (
                defaultdict(deque))
            self._recv_stream = torch.cuda.Stream(device=self.device)

            # Listener handles both INIT and DATA messages
            self._listener = threading.Thread(
                target=self._listen_loop, daemon=True, name="nccl-listen")
            self._listener.start()

    # ── Producer-side ──────────────────────────────────────────────────────

    def send(
        self,
        request_id: str,
        layer_names: list[str],
        stacked_kv: torch.Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Queue KV tensor for NCCL send (PUT_ASYNC: non-blocking)."""
        assert self.is_sender
        with torch.cuda.device(self.device):
            tensor = stacked_kv.to(self.device).contiguous()
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        with self._send_queue_cv:
            self._send_queue.append(
                ("tensor", request_id, layer_names, tensor, ready_event, meta)
            )
            self._send_queue_cv.notify()

    def send_bundle(
        self,
        request_id: str,
        layer_names: list[str],
        meta: dict[str, Any],
        body_chunks: list[torch.Tensor],
        aux_tensors: list[torch.Tensor],
        transfer_meta: dict[str, Any] | None = None,
    ) -> None:
        """Queue a compressed KV bundle for GPU-resident NCCL transfer."""
        assert self.is_sender
        with torch.cuda.device(self.device):
            body = [t.to(self.device).contiguous() for t in body_chunks]
            aux = [t.to(self.device).contiguous() for t in aux_tensors]
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        with self._send_queue_cv:
            self._send_queue.append(
                ("bundle", request_id, layer_names, meta, body, aux, ready_event,
                 transfer_meta)
            )
            self._send_queue_cv.notify()

    def wait_for_sent(self) -> None:
        """Block until all queued and in-flight sends have completed."""
        assert self.is_sender
        with self._send_queue_cv:
            while self._send_queue or self._send_inflight:
                self._send_queue_cv.wait()

    def _send_loop(self) -> None:
        while True:
            with self._send_queue_cv:
                while not self._send_queue:
                    self._send_queue_cv.wait()
                item = self._send_queue.popleft()
                self._send_inflight += 1
            try:
                kind = item[0]
                if kind == "tensor":
                    self._send_one(*item[1:])
                elif kind == "bundle":
                    self._send_bundle_one(*item[1:])
                else:
                    logger.error("[NcclTransport] Unknown send item kind: %s", kind)
            finally:
                with self._send_queue_cv:
                    self._send_inflight -= 1
                    if not self._send_queue and not self._send_inflight:
                        self._send_queue_cv.notify_all()

    def _send_one(
        self,
        request_id: str,
        layer_names: list[str],
        tensor: torch.Tensor,
        ready_event: torch.cuda.Event,
        transfer_meta: dict[str, Any] | None = None,
    ) -> None:
        msg_meta = dict(transfer_meta or {})
        msg_meta["send_ts_ns"] = time.time_ns()
        header = {
            "cmd": "PUT",
            "request_id": request_id,
            "layer_names": layer_names,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        if msg_meta:
            header["meta"] = msg_meta
        self._sock.send(msgpack.dumps(header, use_bin_type=True))

        ack = self._sock.recv()
        if ack != _NCCL_ACK_OK:
            logger.error("[NcclTransport] Consumer OOM for %s", request_id)
            return

        with torch.cuda.stream(self._send_stream):
            self._send_stream.wait_event(ready_event)
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                1,  # consumer is rank 1
                self._comm,
                cudaStream_t(self._send_stream.cuda_stream),
            )
        self._send_stream.synchronize()
        logger.debug("[NcclTransport][RID][SEND] sent rid=%s shape=%s",
                     request_id, list(tensor.shape))

    def _send_bundle_one(
        self,
        request_id: str,
        layer_names: list[str],
        meta: dict[str, Any],
        body_chunks: list[torch.Tensor],
        aux_tensors: list[torch.Tensor],
        ready_event: torch.cuda.Event,
        transfer_meta: dict[str, Any] | None = None,
    ) -> None:
        msg_transfer_meta = dict(transfer_meta or {})
        msg_transfer_meta["send_ts_ns"] = time.time_ns()
        body_specs = [_tensor_spec(t) for t in body_chunks]
        aux_specs = [_tensor_spec(t) for t in aux_tensors]
        msg = {
            "cmd": "PUT_BUNDLE",
            "request_id": request_id,
            "layer_names": layer_names,
            "meta": meta,
            "body_specs": body_specs,
            "aux_specs": aux_specs,
        }
        if msg_transfer_meta:
            msg["transport_meta"] = msg_transfer_meta
        self._sock.send(msgpack.dumps(msg, use_bin_type=True))

        ack = self._sock.recv()
        if ack != _NCCL_ACK_OK:
            logger.error("[NcclTransport] Consumer OOM for bundle %s", request_id)
            return

        with torch.cuda.stream(self._send_stream):
            self._send_stream.wait_event(ready_event)
            for tensor in body_chunks + aux_tensors:
                self._nccl_send_tensor(tensor)
        self._send_stream.synchronize()
        logger.debug(
            "[NcclTransport][RID][SEND] sent bundle rid=%s body=%d aux=%d",
            request_id, len(body_chunks), len(aux_tensors))

    def _nccl_send_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            1,  # consumer is rank 1
            self._comm,
            cudaStream_t(self._send_stream.cuda_stream),
        )

    # ── Consumer-side ──────────────────────────────────────────────────────

    def drain_received(
        self,
    ) -> dict[str, list[tuple[list[str], Any, dict[str, Any] | None]]]:
        """Drain received payloads; each key maps to an ordered list (FIFO)."""
        assert not self.is_sender
        with self._lock:
            result = {k: list(v) for k, v in self._received.items() if v}
            self._received.clear()
        return result

    def _mark_failed(self, request_id: str) -> None:
        """Post a sentinel so the connector can fast-fail this transfer
        instead of blocking until the load timeout. Producer-side OOM is
        not observable here; only consumer-side failures use this path."""
        with self._lock:
            self._received[request_id].append(
                (FAILED_LAYER_NAMES, FAILED_PAYLOAD, None))

    def _listen_loop(self) -> None:
        """Handle INIT and PUT messages from the producer."""
        comm: Optional[object] = None

        while True:
            try:
                frames = self._sock.recv_multipart()
                # ROUTER frame layout: [identity, data]
                identity, raw = frames[0], frames[1]
                msg = msgpack.loads(raw, raw=False)
                cmd = msg["cmd"]

                if cmd == "INIT":
                    uid = ncclUniqueId()
                    uid_bytes = msg["unique_id"]
                    ctypes.memmove(uid.internal, uid_bytes, len(uid_bytes))
                    with torch.cuda.device(self.device):
                        comm = self.nccl.ncclCommInitRank(2, uid, 1)
                    logger.info(
                        "[NcclTransport] Consumer NCCL comm established")

                elif cmd == "PUT":
                    if comm is None:
                        logger.error("[NcclTransport] PUT before INIT")
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        self._mark_failed(msg["request_id"])
                        continue

                    shape = tuple(msg["shape"])
                    dtype = getattr(torch, msg["dtype"])
                    try:
                        tensor = torch.empty(shape, dtype=dtype,
                                             device=self.device)
                        self._sock.send_multipart([identity, _NCCL_ACK_OK])
                    except torch.cuda.OutOfMemoryError:
                        logger.error("[NcclTransport] OOM for %s",
                                     msg["request_id"])
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        self._mark_failed(msg["request_id"])
                        continue

                    recv_t0 = time.monotonic()
                    with torch.cuda.stream(self._recv_stream):
                        self.nccl.ncclRecv(
                            buffer_type(tensor.data_ptr()),
                            tensor.numel(),
                            ncclDataTypeEnum.from_torch(dtype),
                            0,  # producer is rank 0
                            comm,
                            cudaStream_t(self._recv_stream.cuda_stream),
                        )
                    self._recv_stream.synchronize()
                    transfer_ms = (time.monotonic() - recv_t0) * 1e3

                    rid = msg["request_id"]
                    layer_names = msg["layer_names"]
                    logger.debug(
                        "[NcclTransport][RID][RECV] received rid=%s shape=%s",
                        rid, list(tensor.shape))
                    put_meta = dict(msg.get("meta") or {})
                    put_meta.setdefault("transfer_ms", transfer_ms)
                    with self._lock:
                        self._received[rid].append((layer_names, tensor, put_meta))

                elif cmd == "PUT_BUNDLE":
                    if comm is None:
                        logger.error("[NcclTransport] PUT_BUNDLE before INIT")
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        self._mark_failed(msg["request_id"])
                        continue

                    try:
                        body_tensors = [
                            _allocate_from_spec(spec, self.device)
                            for spec in msg["body_specs"]
                        ]
                        aux_tensors = [
                            _allocate_from_spec(spec, self.device)
                            for spec in msg["aux_specs"]
                        ]
                        self._sock.send_multipart([identity, _NCCL_ACK_OK])
                    except torch.cuda.OutOfMemoryError:
                        logger.error("[NcclTransport] OOM for bundle %s",
                                     msg["request_id"])
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        self._mark_failed(msg["request_id"])
                        continue

                    recv_t0 = time.monotonic()
                    with torch.cuda.stream(self._recv_stream):
                        for tensor in body_tensors + aux_tensors:
                            if tensor.numel() == 0:
                                continue
                            self.nccl.ncclRecv(
                                buffer_type(tensor.data_ptr()),
                                tensor.numel(),
                                ncclDataTypeEnum.from_torch(tensor.dtype),
                                0,  # producer is rank 0
                                comm,
                                cudaStream_t(self._recv_stream.cuda_stream),
                            )
                    self._recv_stream.synchronize()
                    transfer_ms = (time.monotonic() - recv_t0) * 1e3

                    rid = msg["request_id"]
                    layer_names = msg["layer_names"]
                    payload = {
                        "__bundle__": True,
                        "meta": msg["meta"],
                        "body_chunks": body_tensors,
                        "aux_tensors": aux_tensors,
                    }
                    logger.debug(
                        "[NcclTransport][RID][RECV] received bundle rid=%s "
                        "body=%d aux=%d",
                        rid, len(body_tensors), len(aux_tensors))
                    with self._lock:
                        transport_meta = dict(msg.get("transport_meta") or {})
                        transport_meta.setdefault("transfer_ms", transfer_ms)
                        self._received[rid].append(
                            (layer_names, payload, transport_meta)
                        )

            except Exception as e:
                logger.error("[NcclTransport] listen_loop error: %s", e)


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def _allocate_from_spec(spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.empty(tuple(spec["shape"]), dtype=getattr(torch, spec["dtype"]),
                       device=device)
