# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Mooncake KV compression helpers.

These cover the CPU-side pieces: configuration parsing, layout/alignment,
protocol serialization, chunk window math, token accounting and head
selection. The encode/decode GPU pipeline is exercised by the integration
tests.
"""

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
    config as compression_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.config import (
    MooncakeCompressionConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.layout import (
    ALIGNMENT,
    build_compression_layout,
    round_up,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.memory import (
    build_memory_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.protocol import (
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionRequest,
    CompressionResponse,
    CompressionSessionOpen,
    CompressionSessionOpened,
    MooncakeXferMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.scores import (
    build_head_selection,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.tokens import (
    chunk_token_counts,
    suffix_token_counts,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _compression_chunk_window,
    _compression_total_chunks,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def _fa_spec(num_kv_heads: int = 2) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=8,
        dtype=torch.bfloat16,
    )


def test_disabled_config_ignores_other_fields():
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": False,
                "unknown-field": "ignored",
            }
        }
    )
    assert not config.available
    assert config.enabled is False


def test_config_defaults_and_pipeline_normalization():
    config = MooncakeCompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True}}
    )
    assert config.available
    assert config.transport == "compression"
    assert config.transformer == "hadamard"
    assert config.pipeline == ("transformer", "quantizer", "codec")
    assert config.slot_count == 2
    assert config.logical_blocks_per_chunk == 1

    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "pipeline": ["transformer", "quantizer"],
            }
        }
    )
    assert config.pipeline == ("transformer", "quantizer")


def test_no_raw_aggregate_key_keeps_disabled_and_compression_defaults():
    config = MooncakeCompressionConfig.from_extra_config({})
    assert not config.available
    assert config.transport == "raw"
    assert not config.uses_aggregate

    config = MooncakeCompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": False}}
    )
    assert not config.available
    assert config.transport == "raw"


def test_raw_aggregate_true_uses_documented_defaults():
    config = MooncakeCompressionConfig.from_extra_config({"raw_aggregate": True})
    assert config.available
    assert config.transport == "raw_aggregate"
    assert config.is_raw_aggregate
    assert config.uses_aggregate
    assert not config.uses_transformer
    assert not config.uses_quantizer
    assert not config.uses_codec
    assert config.pipeline == ("aggregate",)
    assert config.slot_count == 2
    assert config.logical_blocks_per_chunk == 160


def test_raw_aggregate_object_overrides_window():
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "raw_aggregate": {
                "slot_count": 3,
                "logical_blocks_per_chunk": 25,
            }
        }
    )
    assert config.transport == "raw_aggregate"
    assert config.slot_count == 3
    assert config.logical_blocks_per_chunk == 25


def test_raw_agg_alias_is_normalized():
    config = MooncakeCompressionConfig.from_extra_config(
        {"raw_agg": {"slot_count": 1, "logical_blocks_per_chunk": 4}}
    )
    assert config.transport == "raw_aggregate"
    assert config.slot_count == 1
    assert config.logical_blocks_per_chunk == 4

    # Canonical key wins when both aliases are present.
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "raw_aggregate": {"slot_count": 2},
            "raw_agg": {"slot_count": 7},
        }
    )
    assert config.slot_count == 2


def test_raw_aggregate_disabled_key_is_ignored():
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "raw_aggregate": False,
            "compression": {"enable-compression-pipeline": True},
        }
    )
    assert config.transport == "compression"
    assert not config.uses_aggregate


def test_raw_aggregate_validation_errors():
    with pytest.raises(ValueError, match="slot_count must be >= 1"):
        MooncakeCompressionConfig.from_extra_config(
            {"raw_aggregate": {"slot_count": 0}}
        )
    with pytest.raises(ValueError, match="logical_blocks_per_chunk must be >= 1"):
        MooncakeCompressionConfig.from_extra_config(
            {"raw_aggregate": {"logical_blocks_per_chunk": 0}}
        )
    with pytest.raises(ValueError, match="Unknown raw_aggregate parameters"):
        MooncakeCompressionConfig.from_extra_config(
            {"raw_aggregate": {"slot": 2}}
        )
    with pytest.raises(ValueError, match="must be true or an object"):
        MooncakeCompressionConfig.from_extra_config({"raw_aggregate": "yes"})


def test_raw_aggregate_takes_priority_and_warns_once(
    monkeypatch, caplog_vllm
):
    monkeypatch.setattr(
        compression_config_module, "_RAW_AGGREGATE_WARNING_EMITTED", False
    )
    caplog_vllm.set_level("WARNING")
    extra_config = {
        "raw_aggregate": {"slot_count": 2, "logical_blocks_per_chunk": 160},
        "compression": {
            "enable-compression-pipeline": True,
            "split_type": "head",
            "pipeline": ["transformer", "quantizer", "codec"],
            "slot_count": 1,
        },
    }
    first = MooncakeCompressionConfig.from_extra_config(extra_config)
    second = MooncakeCompressionConfig.from_extra_config(extra_config)
    for config in (first, second):
        assert config.transport == "raw_aggregate"
        assert config.uses_aggregate
        assert config.slot_count == 2
        assert config.logical_blocks_per_chunk == 160
    warnings = [
        record.getMessage()
        for record in caplog_vllm.records
        if record.levelno >= 30
        and "raw_aggregate transport selected" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "enable-compression-pipeline" in warnings[0]
    assert "split_type" in warnings[0]
    assert "pipeline" in warnings[0]
    assert "slot_count" in warnings[0]


def test_raw_aggregate_without_compression_does_not_warn(
    monkeypatch, caplog_vllm
):
    monkeypatch.setattr(
        compression_config_module, "_RAW_AGGREGATE_WARNING_EMITTED", False
    )
    caplog_vllm.set_level("WARNING")
    config = MooncakeCompressionConfig.from_extra_config({"raw_aggregate": True})
    assert config.transport == "raw_aggregate"
    assert not any(
        "raw_aggregate transport selected" in record.getMessage()
        for record in caplog_vllm.records
    )


def test_raw_aggregate_with_disabled_compression_does_not_warn(
    monkeypatch, caplog_vllm
):
    monkeypatch.setattr(
        compression_config_module, "_RAW_AGGREGATE_WARNING_EMITTED", False
    )
    caplog_vllm.set_level("WARNING")
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "raw_aggregate": True,
            "compression": {"enable-compression-pipeline": False},
        }
    )
    assert config.transport == "raw_aggregate"
    assert not any(
        "raw_aggregate transport selected" in record.getMessage()
        for record in caplog_vllm.records
    )


def test_disabled_memory_plan_does_not_require_cuda():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    assert build_memory_plan(config, {}) is None


def test_layout_only_contains_full_attention_and_is_aligned():
    specs = {
        "layer.0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=8,
            dtype=torch.bfloat16,
        ),
        "layer.1": MambaSpec(
            block_size=16,
            shapes=((4, 8),),
            dtypes=(torch.bfloat16,),
            num_heads=4,
        ),
    }
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "layer",
                "pipeline": ["transformer"],
            }
        }
    )
    layout = build_compression_layout(specs, config, {0: 0}, 2)
    assert {part.layer_name for part in layout.layer_parts} == {"layer.0"}
    assert layout.scratch_stride % ALIGNMENT == 0
    assert layout.u8_base % ALIGNMENT == 0


def test_layout_head_split_creates_low_and_high_sections():
    specs = {"model.layers.0.self_attn": _fa_spec()}
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "head",
            }
        }
    )
    layout = build_compression_layout(specs, config, {0: 1}, 1)
    key_part = layout.layer_parts[0]
    value_part = layout.layer_parts[1]
    assert [section.kv_part for section in key_part.sections] == [0, 0]
    assert [section.kv_part for section in value_part.sections] == [1, 1]
    assert [
        (section.precision, section.head_count, section.num_levels)
        for section in key_part.sections
    ] == [
        ("low", 1, config.low_key_num_levels),
        ("high", 1, config.high_key_num_levels),
    ]
    assert [
        (section.precision, section.head_count, section.num_levels)
        for section in value_part.sections
    ] == [
        ("low", 1, config.low_value_num_levels),
        ("high", 1, config.high_value_num_levels),
    ]
    for part in layout.layer_parts:
        for section in part.sections:
            assert section.aux_min_offset % ALIGNMENT == 0
            assert section.aux_scale_offset % ALIGNMENT == 0


def test_compression_protocol_roundtrip():
    metadata = MooncakeXferMetadata(
        remote_hostname="host",
        remote_port=1234,
        remote_tp_size=2,
        remote_tp_rank=1,
        req_blocks={"req": ("xfer", [[1, 2]])},
        kv_caches_base_addr=[4096],
        block_lens=[16],
        kv_block_lens=[16],
    )
    encoder = msgspec.msgpack.Encoder()
    request_decoder = msgspec.msgpack.Decoder(CompressionRequest)
    response_decoder = msgspec.msgpack.Decoder(CompressionResponse)

    requests = [
        CompressionSessionOpen(metadata=metadata, nonce="nonce"),
        CompressionChunkPull(
            transfer_id="xfer",
            d_req_id="req",
            chunk_id=3,
            slot_id=1,
            payload_addr=4096,
            aux_addr=8192,
            tail_addr=12288,
            nonce="nonce",
        ),
    ]
    for message in requests:
        assert request_decoder.decode(encoder.encode(message)) == message

    responses = [
        CompressionSessionOpened(logical_block_count=5, total_chunks=10),
        CompressionChunkReady(
            chunk_id=1, data_bytes=1024, payload_bytes=512, aux_bytes=64
        ),
    ]
    for message in responses:
        assert response_decoder.decode(encoder.encode(message)) == message

    ready = CompressionChunkReady(
        chunk_id=1, data_bytes=1024, payload_bytes=512, aux_bytes=64
    )
    assert ready.u8_bytes == 1024
    assert ready.codec_bytes == 512


def test_compression_chunk_window_alternates_kv_parts():
    assert _compression_total_chunks(0, 1) == 0
    assert _compression_total_chunks(1, 1) == 2
    assert _compression_total_chunks(5, 2) == 6
    assert _compression_total_chunks(5, 3) == 4

    assert _compression_chunk_window(0, 5, 2) == (0, 2, 0)
    assert _compression_chunk_window(1, 5, 2) == (0, 2, 1)
    assert _compression_chunk_window(2, 5, 2) == (2, 2, 0)
    assert _compression_chunk_window(4, 5, 2) == (4, 1, 0)
    assert _compression_chunk_window(5, 5, 2) == (4, 1, 1)

    # With lbc=1 every logical block gets its own K and V chunk.
    assert [_compression_chunk_window(chunk_id, 3, 1)[2] for chunk_id in range(6)] == [
        0,
        1,
        0,
        1,
        0,
        1,
    ]


def test_suffix_token_counts():
    counts = suffix_token_counts(1000, [[0, 1, 2], [7, 8, 9, 10]], {0: 16, 1: 32})
    assert counts == {0: 1000 - (63 - 3) * 16, 1: 1000 - (32 - 4) * 32}
    # A full prefix hit has no transferred blocks and therefore no suffix.
    assert suffix_token_counts(1000, [], {0: 16}) == {0: 0}


def test_chunk_token_counts():
    valid_tokens = {3: 1371}
    block_sizes = {3: 784}
    assert chunk_token_counts(valid_tokens, block_sizes, 0, 1) == {3: 784}
    assert chunk_token_counts(valid_tokens, block_sizes, 1, 1) == {3: 587}
    assert chunk_token_counts(valid_tokens, block_sizes, 0, 2) == {3: 1371}


def test_head_selection_layer_split_needs_no_scores():
    specs = {
        "model.layers.3.self_attn": _fa_spec(),
        "model.layers.7.self_attn": _fa_spec(),
    }
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "layer",
            }
        }
    )
    selection = build_head_selection(specs, config, tp_rank=1, tp_size=2)
    assert selection.low_counts == {3: 0, 7: 0}
    assert selection.local_orders == {3: (0, 1), 7: (0, 1)}


def test_head_selection_uses_scores_and_tp_slice():
    specs = {
        "model.layers.3.self_attn": _fa_spec(),
        "model.layers.7.self_attn": _fa_spec(),
    }
    scores = np.ones((64, 4), dtype=np.float32)
    scores[3, 3] = 0.0  # layer 3, global head 3 -> rank 1 local head 1
    scores[7, 2] = 0.0  # layer 7, global head 2 -> rank 1 local head 0
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "head",
                "hybrid_ratio": 0.2,
            }
        }
    )
    selection = build_head_selection(specs, config, tp_rank=1, tp_size=2, scores=scores)
    assert selection.low_counts == {3: 1, 7: 1}
    assert selection.local_orders == {3: (1, 0), 7: (0, 1)}


_STAGE_PIPELINES = [
    ["transformer"],
    ["quantizer"],
    ["transformer", "quantizer"],
    ["transformer", "codec"],
    ["quantizer", "codec"],
    ["codec"],
    ["transformer", "quantizer", "codec"],
    [],
]


@pytest.mark.parametrize("pipeline", _STAGE_PIPELINES)
def test_config_stage_flags(pipeline):
    config = MooncakeCompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True, "pipeline": pipeline}}
    )
    assert config.uses_aggregate is False
    assert config.uses_transformer == ("transformer" in pipeline)
    assert config.uses_quantizer == ("quantizer" in pipeline)
    assert config.uses_codec == ("codec" in pipeline)

    aggregate = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "pipeline": ["transformer", "quantizer", "codec", "aggregate"],
            }
        }
    )
    assert aggregate.uses_aggregate is True
    assert not aggregate.uses_transformer
    assert not aggregate.uses_quantizer
    assert not aggregate.uses_codec


def _memory_plan_env(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.utils.get_current_attn_backends",
        lambda config: [],
    )
    monkeypatch.setattr(
        "vllm.v1.worker.utils.select_common_block_size",
        lambda block_size, backends: block_size,
    )
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "compression": {
                    "enable-compression-pipeline": True,
                    "split_type": "layer",
                    "slot_count": 2,
                    "logical_blocks_per_chunk": 2,
                }
            }
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=1)),
    )


def _build_plan(monkeypatch, pipeline, probe):
    vllm_config = _memory_plan_env(monkeypatch)
    vllm_config.kv_transfer_config.kv_connector_extra_config["compression"][
        "pipeline"
    ] = pipeline
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.memory._probe_operators",
        probe,
    )
    specs = {"model.layers.0.self_attn": _fa_spec()}
    plan = build_memory_plan(vllm_config, specs)
    assert plan is not None
    return plan


def _recording_probe(calls):
    def probe(layout, device, sizes, warm_hadamard):
        calls.append((tuple(sizes), warm_hadamard))
        return (1024, (8192, 8192), 2048)

    return probe


def _forbidden_probe(layout, device, sizes, warm_hadamard):
    raise AssertionError("probe must not run without codec")


@pytest.mark.parametrize(
    "pipeline",
    [["transformer"], ["quantizer"], ["transformer", "quantizer"], [], ["aggregate"]],
)
def test_memory_plan_skips_codec_without_codec_stage(monkeypatch, pipeline):
    plan = _build_plan(monkeypatch, pipeline, _forbidden_probe)
    assert plan.codec_workspace_bytes == 0
    assert plan.codec_max_output_bytes == (0, 0)
    assert plan.retained_driver_bytes == 0


@pytest.mark.parametrize(
    "pipeline, warm_hadamard",
    [
        (["transformer", "codec"], True),
        (["transformer", "quantizer", "codec"], True),
        (["codec"], False),
        (["quantizer", "codec"], False),
    ],
)
def test_memory_plan_probes_only_packed_sizes(monkeypatch, pipeline, warm_hadamard):
    calls: list[tuple[tuple[int, ...], bool]] = []
    plan = _build_plan(monkeypatch, pipeline, _recording_probe(calls))
    quantized = "quantizer" in pipeline
    packed = (
        plan.layout.u8_bytes_by_part if quantized else plan.layout.raw_bytes_by_part
    )
    assert calls == [(tuple(packed), warm_hadamard)]
    assert plan.codec_max_output_bytes == (8192, 8192)
    assert plan.codec_workspace_bytes == 1024 * plan.config.slot_count
    assert plan.retained_driver_bytes == 2048 * plan.config.slot_count


@pytest.mark.parametrize("pipeline", _STAGE_PIPELINES)
def test_memory_plan_arena_is_minimal(monkeypatch, pipeline):
    plan = _build_plan(monkeypatch, pipeline, _recording_probe([]))
    layout = plan.layout
    base = max(
        2 * layout.scratch_stride,
        layout.u8_base + max(layout.packed_bytes_by_part),
    )
    output = 8192 if "codec" in pipeline else 0
    assert layout.arena_bytes == round_up(max(base, output), ALIGNMENT)
    assert (layout.aux_arena_bytes > 0) == ("quantizer" in pipeline)
    assert layout.packed_bytes_by_part == (
        layout.u8_bytes_by_part if "quantizer" in pipeline else layout.raw_bytes_by_part
    )


def test_memory_plan_aggregate_has_no_staging(monkeypatch):
    plan = _build_plan(monkeypatch, ["aggregate"], _forbidden_probe)
    layout = plan.layout
    assert layout.aggregate
    assert layout.arena_bytes == round_up(max(layout.raw_bytes_by_part), ALIGNMENT)
    assert layout.aux_arena_bytes == 0
    assert layout.raw_tail_capacity == 0


def _bare_worker():
    """A MooncakeConnectorWorker with only what get_finished needs."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (  # noqa: E501
        MooncakeConnectorWorker,
    )

    worker = object.__new__(MooncakeConnectorWorker)
    worker.tp_rank = 0
    worker._finished_sending = queue.SimpleQueue()
    worker._finished_recving = queue.SimpleQueue()
    worker.reqs_need_send = {}
    worker.xfer_stats = SimpleNamespace(record_kv_expired_req=lambda: None)
    return worker


def test_get_finished_drains_queues_without_an_event_loop():
    worker = _bare_worker()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker._finished_sending.put, [f"s{i}" for i in range(200)]))
        list(pool.map(worker._finished_recving.put, ["r1", "r2", "r3"]))

    sending, recving = worker.get_finished()
    assert sending == {f"s{i}" for i in range(200)}
    assert recving == {"r1", "r2", "r3"}
    # Second call observes nothing and must be cheap/empty.
    assert worker.get_finished() == (None, None)


def test_expired_sends_are_published_to_the_finished_queue():
    worker = _bare_worker()
    now = time.perf_counter()
    worker.reqs_need_send = {
        "t1": SimpleNamespace(p_req_id="stale", expire_time=now - 1, sending=0),
        "t2": SimpleNamespace(p_req_id="fresh", expire_time=now + 60, sending=0),
        "t3": SimpleNamespace(p_req_id="inflight", expire_time=now - 1, sending=1),
    }
    worker._sweep_expired_sends()
    assert set(worker.reqs_need_send) == {"t2", "t3"}
    assert worker.get_finished() == ({"stale"}, None)
