# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the out-of-process Mooncake compression session helper."""

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import session_ipc as ipc
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.protocol import (
    CompressionAbort,
    CompressionAbortAck,
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionSessionOpen,
    CompressionSessionOpened,
    MooncakeXferMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.session_process import (
    SessionProcess,
)

_ENC = msgspec.msgpack.Encoder()
_REQ_DEC = msgspec.msgpack.Decoder(
    CompressionSessionOpen | CompressionChunkPull | CompressionAbort
)
_RESP_DEC = msgspec.msgpack.Decoder(
    CompressionSessionOpened | CompressionChunkReady | CompressionAbortAck
)


class _Stats:
    def __init__(self):
        self.expired = 0
        self.failed = 0

    def record_kv_expired_req(self):
        self.expired += 1

    def record_failed_recv(self):
        self.failed += 1


def _wait_queue(target: queue.SimpleQueue, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return target.get_nowait()
        except queue.Empty:
            time.sleep(0.02)
    return None


def _make_init(needs_sender: bool, needs_receiver: bool, slots: int = 2):
    return ipc.InitMessage(
        needs_sender=needs_sender,
        needs_receiver=needs_receiver,
        register_with_bootstrap=False,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        engine_id="A",
        dp_rank=0,
        hostname="127.0.0.1",
        rpc_port=1234,
        bootstrap_host="127.0.0.1",
        bootstrap_port=1,
        slot_addresses=[
            ipc.SlotAddresses(
                payload=1000 + 100 * index,
                aux=2000 + 100 * index,
                tail=3000 + 100 * index,
            )
            for index in range(slots)
        ],
        kv_caches_base_addr=[4096],
        block_lens=[16],
        kv_block_lens=[16],
        registered_layer_names=["l0"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )


def test_session_process_toggle(monkeypatch):
    assert ipc.session_process_enabled()
    monkeypatch.setenv("KVSERVE_SESSION_PROCESS", "0")
    assert not ipc.session_process_enabled()


def test_handshake_target_ranks():
    assert ipc.handshake_target_ranks(0, 4, 4) == [0]
    assert ipc.handshake_target_ranks(3, 4, 4) == [3]
    assert ipc.handshake_target_ranks(0, 4, 2) == [0]
    assert ipc.handshake_target_ranks(3, 4, 2) == [1]
    assert ipc.handshake_target_ranks(1, 2, 4) == [2, 3]


def test_ipc_roundtrip():
    messages = [
        ipc.EngineStop(),
        ipc.HelperReady(side_channel_port=1234),
        ipc.Stopped(),
        ipc.FinalEvent(kind="sending", req_id="r1"),
        ipc.RecvDone(d_req_id="r2", failed_blocks=[1, 2]),
        ipc.ProducerMetadata(
            announced=[
                ipc.ProducerAnnouncement(
                    p_req_id="p",
                    transfer_id="t",
                    local_block_ids=[[1, 2]],
                    num_tokens=32,
                )
            ],
            published=[ipc.ProducerPublish(p_req_id="q", transfer_id="u")],
            dropped=["v"],
        ),
        ipc.ConsumerMetadata(
            requests=[
                ipc.ConsumerRequest(
                    d_req_id="d",
                    transfer_id="t",
                    local_block_ids=[[3]],
                    num_tokens=64,
                    remote_engine_id="B",
                    remote_bootstrap_addr="http://host:1",
                )
            ]
        ),
        ipc.OpenProducerJob(
            job_id=1,
            transfer_id="t",
            d_req_id="d",
            nonce="n",
            remote_hostname="h",
            remote_port=1,
            remote_tp_size=1,
            remote_tp_rank=0,
            remote_pp_size=1,
            local_block_ids=[[1]],
            remote_block_ids=[[2]],
            num_tokens=32,
            kv_caches_base_addr=[1],
            block_lens=[16],
            kv_block_lens=[16],
            registered_layer_names=["l"],
            registered_layer_indices=[0],
            registered_group_indices=[0],
        ),
        ipc.EncodeChunkJob(
            job_id=2,
            transfer_id="t",
            d_req_id="d",
            nonce="n",
            chunk_id=3,
            slot_id=1,
            logical_block_count=2,
            local_block_ids=[[1]],
            valid_tokens_by_group={0: 32},
            remote_session="h:1",
            payload_addr=1,
            aux_addr=2,
            tail_addr=3,
        ),
        ipc.DecodeChunkJob(
            job_id=3,
            transfer_id="t",
            d_req_id="d",
            nonce="n",
            chunk_id=1,
            slot_id=0,
            logical_block_count=2,
            local_block_ids=[[1]],
            num_tokens=32,
            codec_bytes=10,
            u8_bytes=20,
        ),
        ipc.DrainSlotsJob(job_id=4, slot_ids=[0, 1]),
        ipc.JobDone(job_id=5, error=None, u8_bytes=1, codec_bytes=2, aux_bytes=3),
    ]
    helper_jobs = msgspec.msgpack.Decoder(ipc.HelperToEngine)
    engine_msgs = msgspec.msgpack.Decoder(ipc.EngineToHelper)
    for message in messages:
        data = _ENC.encode(message)
        engine_only = (
            ipc.ProducerMetadata,
            ipc.ConsumerMetadata,
            ipc.JobDone,
            ipc.EngineStop,
        )
        union = engine_msgs if isinstance(message, engine_only) else helper_jobs
        assert union.decode(data) == message


def test_session_process_start_stop():
    fs, fr, inv = (
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
    )
    process = SessionProcess(
        lambda job: ipc.JobDone(job_id=job.job_id),
        _make_init(False, False),
        fs,
        fr,
        inv,
        _Stats(),
        -1,
    )
    process.start()
    assert process.side_channel_port == 0
    process.shutdown()


def test_session_helper_producer_flow():
    jobs: list = []
    jobs_lock = threading.Lock()

    def handler(job):
        with jobs_lock:
            jobs.append(job)
        if isinstance(job, ipc.OpenProducerJob):
            assert job.local_block_ids == [[1, 2, 3, 4]]
            assert job.remote_block_ids == [[11, 12, 13, 14]]
            return ipc.JobDone(
                job_id=job.job_id,
                logical_block_count=2,
                total_chunks=4,
                local_block_ids=[[1, 2, 3, 4]],
                valid_tokens_by_group={0: 256},
            )
        if isinstance(job, ipc.EncodeChunkJob):
            assert job.valid_tokens_by_group == {0: 256}
            return ipc.JobDone(
                job_id=job.job_id,
                u8_bytes=100 + job.chunk_id,
                codec_bytes=50,
                aux_bytes=0,
            )
        if isinstance(job, ipc.DrainSlotsJob):
            return ipc.JobDone(job_id=job.job_id)
        raise AssertionError(type(job))

    fs, fr, inv = (
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
    )
    stats = _Stats()
    process = SessionProcess(
        handler, _make_init(True, False), fs, fr, inv, stats, -1
    )
    process.start()
    try:
        assert process.side_channel_port > 0
        process.publish(
            ipc.ProducerMetadata(
                announced=[
                    ipc.ProducerAnnouncement(
                        p_req_id="p1",
                        transfer_id="t1",
                        local_block_ids=[[1, 2, 3, 4]],
                        num_tokens=256,
                    )
                ],
            )
        )
        context = zmq.Context()
        try:
            dealer = context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.RCVTIMEO, 10000)
            dealer.connect(f"tcp://127.0.0.1:{process.side_channel_port}")
            metadata = MooncakeXferMetadata(
                remote_hostname="127.0.0.1",
                remote_port=9999,
                remote_tp_size=1,
                remote_tp_rank=0,
                req_blocks={"d1": ("t1", [[11, 12, 13, 14]])},
                kv_caches_base_addr=[4096],
                block_lens=[16],
                kv_block_lens=[16],
            )
            dealer.send(
                _ENC.encode(CompressionSessionOpen(metadata=metadata, nonce="n1"))
            )
            opened = _RESP_DEC.decode(dealer.recv())
            assert isinstance(opened, CompressionSessionOpened)
            assert opened.total_chunks == 4
            assert opened.logical_block_count == 2
            for chunk_id in range(4):
                dealer.send(
                    _ENC.encode(
                        CompressionChunkPull(
                            transfer_id="t1",
                            d_req_id="d1",
                            chunk_id=chunk_id,
                            slot_id=0,
                            payload_addr=7777,
                            aux_addr=8888,
                            tail_addr=9999,
                            nonce="n1",
                        )
                    )
                )
                ready = _RESP_DEC.decode(dealer.recv())
                assert isinstance(ready, CompressionChunkReady)
                assert ready.error is None
                assert ready.u8_bytes == 100 + chunk_id
            assert _wait_queue(fs) == "p1"
            assert _wait_queue(fr) is None
            # An abort after completion must be acknowledged.
            dealer.send(
                _ENC.encode(
                    CompressionAbort(
                        transfer_id="t1", d_req_id="d1", reason="x", nonce="n1"
                    )
                )
            )
            ack = _RESP_DEC.decode(dealer.recv())
            assert isinstance(ack, CompressionAbortAck)
            assert ack.error is None
            dealer.close(linger=0)
        finally:
            context.term()
        with jobs_lock:
            kinds = [type(job).__name__ for job in jobs]
        assert kinds.count("EncodeChunkJob") == 4
    finally:
        process.shutdown()


def test_session_helper_consumer_flow():
    jobs: list = []
    jobs_lock = threading.Lock()

    def handler(job):
        with jobs_lock:
            jobs.append(job)
        if isinstance(job, ipc.DecodeChunkJob):
            assert job.local_block_ids == [[1, 2, 3, 4]]
            assert job.num_tokens == 256
            assert job.chunk_id < 4
            return ipc.JobDone(job_id=job.job_id)
        if isinstance(job, ipc.DrainSlotsJob):
            return ipc.JobDone(job_id=job.job_id)
        raise AssertionError(type(job))

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router_port = router.bind_to_random_port("tcp://127.0.0.1")

    class QueryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {
                    "0": {
                        "engine_id": "B",
                        "worker_addr": {
                            "0": {"0": f"tcp://127.0.0.1:{router_port}"}
                        },
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), QueryHandler)
    http_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def producer_side():
        identity, raw = router.recv_multipart()
        request = _REQ_DEC.decode(raw)
        assert isinstance(request, CompressionSessionOpen)
        assert request.nonce
        router.send_multipart(
            (
                identity,
                _ENC.encode(
                    CompressionSessionOpened(
                        logical_block_count=2, total_chunks=4
                    )
                ),
            )
        )
        for _ in range(4):
            identity, raw = router.recv_multipart()
            pull = _REQ_DEC.decode(raw)
            router.send_multipart(
                (
                    identity,
                    _ENC.encode(
                        CompressionChunkReady(
                            chunk_id=pull.chunk_id,
                            data_bytes=100 + pull.chunk_id,
                            payload_bytes=50,
                            aux_bytes=0,
                        )
                    ),
                )
            )

    producer_thread = threading.Thread(target=producer_side, daemon=True)
    producer_thread.start()

    fs, fr, inv = (
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
    )
    stats = _Stats()
    process = SessionProcess(
        handler, _make_init(False, True), fs, fr, inv, stats, -1
    )
    process.start()
    try:
        process.publish(
            ipc.ConsumerMetadata(
                requests=[
                    ipc.ConsumerRequest(
                        d_req_id="d1",
                        transfer_id="t1",
                        local_block_ids=[[1, 2, 3, 4]],
                        num_tokens=256,
                        remote_engine_id="B",
                        remote_bootstrap_addr=f"http://127.0.0.1:{http_port}",
                    )
                ]
            )
        )
        assert _wait_queue(fr, timeout=15) == "d1"
        assert _wait_queue(inv, timeout=1) is None
        with jobs_lock:
            kinds = [type(job).__name__ for job in jobs]
        assert kinds.count("DecodeChunkJob") == 4
    finally:
        process.shutdown()
        httpd.shutdown()
        router.close(linger=0)
        context.term()
