# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU round-trip tests for the Mooncake KV compression pipeline.

Each test builds a real compression memory plan (including the nvCOMP/humming
probe), runs the encode/decode pipeline on a single attention layer, and
compares the reconstructed KV cache against the source.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.layout import (
    ALIGNMENT,
    round_up,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.manager import (
    MooncakeCompressionManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.memory import (
    build_memory_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.pipeline import (
    CompressionPipeline,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _compression_chunk_window,
    _compression_total_chunks,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Mooncake compression requires CUDA"
)

_CPP_BACKEND = os.environ.get("KVSERVE_COMPRESSION_BACKEND", "python") == "cpp"

BLOCK_SIZE = 16
NUM_BLOCKS = 4
NUM_HEADS = 2
HEAD_SIZE = 8
LAYER_NAME = "model.layers.0.self_attn"


def _build_manager(
    monkeypatch,
    compression: dict,
    num_layers: int = 1,
    raw_aggregate: dict | bool | None = None,
):
    extra_config: dict = {"compression": compression}
    if raw_aggregate is not None:
        extra_config["raw_aggregate"] = raw_aggregate
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra_config
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_layers)
        ),
    )
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
    layer_names = [f"model.layers.{index}.self_attn" for index in range(num_layers)]
    specs = {
        name: FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        for name in layer_names
    }
    plan = build_memory_plan(vllm_config, specs)
    assert plan is not None
    manager = MooncakeCompressionManager(
        plan=plan,
        kv_cache_layout="LBHNC",
        blocks_per_logical=plan.physical_blocks_per_logical,
        tp_size=1,
        pp_size=1,
        device=torch.device("cuda", 0),
    )
    cache = torch.rand(
        NUM_BLOCKS,
        NUM_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.bind_kv_caches(
        dict.fromkeys(layer_names, cache), dict.fromkeys(layer_names, 0)
    )
    return manager, cache


@pytest.fixture
def make_manager(monkeypatch):
    created = []

    def factory(
        compression: dict,
        num_layers: int = 1,
        raw_aggregate: dict | bool | None = None,
    ):
        manager, cache = _build_manager(
            monkeypatch, compression, num_layers, raw_aggregate
        )
        created.append(manager)
        return manager, cache

    yield factory
    for manager in created:
        manager.shutdown()


def _roundtrip(manager, cache, kv_part: int, valid_tokens: int, slot_id: int = 0):
    logical_ids = [[0]]
    encoded = manager.encode(
        slot_id, logical_ids, kv_part, valid_tokens_by_group={0: valid_tokens}
    )
    encoded.event.synchronize()
    offset = kv_part * HEAD_SIZE
    source = cache[0, :, :, offset : offset + HEAD_SIZE].clone()
    cache.zero_()
    event = manager.decode(
        slot_id,
        logical_ids,
        kv_part,
        encoded.codec_bytes,
        encoded.u8_bytes,
        valid_tokens_by_group={0: valid_tokens},
    )
    event.synchronize()
    return source, cache[0, :, :, offset : offset + HEAD_SIZE], encoded


def test_compression_roundtrip_full_block(make_manager):
    manager, cache = make_manager(
        {"enable-compression-pipeline": True, "split_type": "layer"}
    )
    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        assert encoded.codec_bytes > 0
        assert encoded.raw_tail_bytes == 0
        torch.testing.assert_close(output, source, atol=0.3, rtol=0)


def test_compression_roundtrip_raw_tail_is_exact(make_manager):
    manager, cache = make_manager(
        {"enable-compression-pipeline": True, "split_type": "layer"}
    )
    valid_tokens = BLOCK_SIZE // 2
    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, valid_tokens)
        assert encoded.codec_bytes == 0
        assert encoded.raw_tail_bytes > 0
        assert torch.equal(output[:, :valid_tokens, :], source[:, :valid_tokens, :])


def test_compression_roundtrip_uses_each_slot(make_manager):
    manager, cache = make_manager(
        {"enable-compression-pipeline": True, "split_type": "layer"}
    )
    for slot_id in range(2):
        source, output, encoded = _roundtrip(
            manager, cache, 0, BLOCK_SIZE, slot_id=slot_id
        )
        assert encoded.codec_bytes > 0
        torch.testing.assert_close(output, source, atol=0.3, rtol=0)


def test_index_built_once_per_group(monkeypatch, make_manager):
    """One block-index build per KV group per encode call, not per layer."""
    manager, _ = make_manager(
        {"enable-compression-pipeline": True, "split_type": "layer"}, num_layers=3
    )
    calls = []
    original = CompressionPipeline._index_tensor

    def counting_index_tensor(self, slot, logical_ids, row=0):
        calls.append(row)
        return original(self, slot, logical_ids, row)

    monkeypatch.setattr(CompressionPipeline, "_index_tensor", counting_index_tensor)
    for kv_part in (0, 1):
        del calls[:]
        encoded = manager.encode(
            0, [[0]], kv_part, valid_tokens_by_group={0: BLOCK_SIZE}
        )
        encoded.event.synchronize()
        assert len(calls) == 1
        manager.decode(
            0,
            [[0]],
            kv_part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: BLOCK_SIZE},
        ).synchronize()


def test_compression_roundtrip_head_split(make_manager, tmp_path):
    scores = np.linspace(0.0, 1.0, 64 * NUM_HEADS, dtype=np.float32).reshape(
        64, NUM_HEADS
    )
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps(scores.tolist()))
    manager, cache = make_manager(
        {
            "enable-compression-pipeline": True,
            "split_type": "head",
            "score_path": str(score_path),
            "hybrid_ratio": 0.5,
        }
    )
    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        assert encoded.codec_bytes > 0
        torch.testing.assert_close(output, source, atol=0.7, rtol=0)


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


def _ans_capacity(size: int) -> int:
    from nvidia import nvcomp

    stream = torch.cuda.Stream()
    codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
    source = torch.empty(size, dtype=torch.uint8, device="cuda")
    array = nvcomp.as_array(source, cuda_stream=stream.cuda_stream)
    return codec.get_max_comp_buffer_size(array)


def _count_index_select_dims(monkeypatch) -> list[int]:
    dims: list[int] = []
    original = torch.index_select

    def counting(input, dim, index, *args, **kwargs):
        dims.append(dim)
        return original(input, dim, index, *args, **kwargs)

    monkeypatch.setattr(torch, "index_select", counting)
    return dims


@pytest.mark.skipif(
    _CPP_BACKEND, reason="asserts Python-path internals"
)
def test_layer_mode_skips_head_reorder_copy(monkeypatch, make_manager):
    manager, cache = make_manager(
        {"enable-compression-pipeline": True, "split_type": "layer"}
    )
    execution = manager.prepare()
    assert execution.identity_layers == frozenset({LAYER_NAME})
    dims = _count_index_select_dims(monkeypatch)
    for kv_part in (0, 1):
        source, output, _ = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        torch.testing.assert_close(output, source, atol=0.3, rtol=0)
    assert 0 in dims
    assert 2 not in dims


@pytest.mark.skipif(
    _CPP_BACKEND, reason="asserts Python-path internals"
)
def test_head_split_non_identity_order_roundtrips(monkeypatch, tmp_path, make_manager):
    scores = np.ones((64, NUM_HEADS), dtype=np.float32)
    scores[0, 1] = 0.0
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps(scores.tolist()))
    manager, cache = make_manager(
        {
            "enable-compression-pipeline": True,
            "split_type": "head",
            "score_path": str(score_path),
            "hybrid_ratio": 0.3,
        }
    )
    execution = manager.prepare()
    assert LAYER_NAME not in execution.identity_layers
    dims = _count_index_select_dims(monkeypatch)
    for kv_part in (0, 1):
        source, output, _ = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        torch.testing.assert_close(output, source, atol=0.7, rtol=0)
    assert 2 in dims


@pytest.mark.parametrize("pipeline", _STAGE_PIPELINES)
def test_stage_memory_is_on_demand(make_manager, pipeline):
    manager, cache = make_manager(
        {
            "enable-compression-pipeline": True,
            "split_type": "layer",
            "pipeline": pipeline,
        }
    )
    plan = manager.plan
    layout = plan.layout
    transformer = "transformer" in pipeline
    quantized = "quantizer" in pipeline
    codec = "codec" in pipeline
    packed = layout.u8_bytes_by_part if quantized else layout.raw_bytes_by_part
    staging = max(2 * layout.scratch_stride, layout.u8_base + max(packed))
    if codec:
        cap = max(_ans_capacity(size) for size in set(packed))
        assert plan.codec_max_output_bytes == (cap, cap)
        assert layout.arena_bytes == round_up(max(staging, cap), ALIGNMENT)
        assert plan.codec_workspace_bytes > 0
    else:
        assert plan.codec_max_output_bytes == (0, 0)
        assert plan.codec_workspace_bytes == 0
        assert plan.retained_driver_bytes == 0
        assert layout.arena_bytes == round_up(staging, ALIGNMENT)
    assert (layout.aux_arena_bytes > 0) == quantized
    assert (len(manager.signs) > 0) == transformer

    slot_bytes = (
        2 * layout.arena_bytes + layout.aux_arena_bytes + layout.raw_tail_capacity
    )
    expected_bulk = (
        plan.config.slot_count * slot_bytes
        + plan.codec_workspace_bytes
        + plan.metadata_bytes
    )
    assert manager.bulk.numel() == expected_bulk

    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        assert encoded.u8_bytes == packed[kv_part]
        if quantized:
            torch.testing.assert_close(output, source, atol=0.3, rtol=0)
        elif transformer:
            torch.testing.assert_close(output, source, atol=0.05, rtol=0)
        else:
            torch.testing.assert_close(output, source, atol=0, rtol=0)


def test_aggregate_memory_and_roundtrip(make_manager):
    manager, cache = make_manager(
        {
            "enable-compression-pipeline": True,
            "split_type": "layer",
            "pipeline": ["aggregate"],
        }
    )
    plan = manager.plan
    layout = plan.layout
    assert layout.aggregate
    assert layout.aux_arena_bytes == 0
    assert layout.raw_tail_capacity == 0
    assert plan.codec_workspace_bytes == 0
    assert plan.codec_max_output_bytes == (0, 0)
    assert plan.retained_driver_bytes == 0
    assert layout.arena_bytes == round_up(max(layout.raw_bytes_by_part), ALIGNMENT)
    slot_bytes = (
        2 * layout.arena_bytes + layout.aux_arena_bytes + layout.raw_tail_capacity
    )
    expected_bulk = (
        plan.config.slot_count * slot_bytes
        + plan.codec_workspace_bytes
        + plan.metadata_bytes
    )
    assert manager.bulk.numel() == expected_bulk

    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        assert encoded.raw_tail_bytes == 0
        torch.testing.assert_close(output, source, atol=0, rtol=0)


def test_raw_aggregate_config_beats_compression_and_roundtrips_exactly(make_manager):
    """Product ``raw_aggregate`` key wins over a full compression config."""
    manager, cache = make_manager(
        {
            "enable-compression-pipeline": True,
            "split_type": "layer",
            "pipeline": ["transformer", "quantizer", "codec"],
            "slot_count": 1,
            "logical_blocks_per_chunk": 4,
        },
        raw_aggregate={"slot_count": 2, "logical_blocks_per_chunk": 2},
    )
    plan = manager.plan
    layout = plan.layout
    assert plan.config.transport == "raw_aggregate"
    assert plan.config.uses_aggregate
    assert plan.config.slot_count == 2
    assert plan.config.logical_blocks_per_chunk == 2
    assert layout.aggregate
    assert layout.aux_arena_bytes == 0
    assert layout.raw_tail_capacity == 0
    assert plan.codec_workspace_bytes == 0
    assert plan.codec_max_output_bytes == (0, 0)
    assert plan.retained_driver_bytes == 0
    assert manager.bulk.numel() > 0

    single_block_bytes = BLOCK_SIZE * NUM_HEADS * HEAD_SIZE * 2
    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, BLOCK_SIZE)
        assert encoded.u8_bytes == single_block_bytes
        assert encoded.codec_bytes == single_block_bytes
        assert encoded.raw_tail_bytes == 0
        assert torch.equal(output, source)


def test_raw_aggregate_partial_block_valid_region_is_exact(make_manager):
    manager, cache = make_manager(
        {"enable-compression-pipeline": True},
        raw_aggregate={"slot_count": 2, "logical_blocks_per_chunk": 2},
    )
    valid_tokens = BLOCK_SIZE // 2
    for kv_part in (0, 1):
        source, output, encoded = _roundtrip(manager, cache, kv_part, valid_tokens)
        assert encoded.u8_bytes > 0
        assert encoded.raw_tail_bytes == 0
        assert torch.equal(output[:, :valid_tokens], source[:, :valid_tokens])


def test_raw_aggregate_multi_chunk_alternates_slots_byte_exact(make_manager):
    """Mirror the helper job flow: one chunk per logical block, K/V mixed."""
    manager, cache = make_manager(
        {"enable-compression-pipeline": True},
        raw_aggregate={"slot_count": 2, "logical_blocks_per_chunk": 1},
    )
    total_blocks = NUM_BLOCKS
    total_chunks = _compression_total_chunks(total_blocks, 1)
    assert total_chunks == 2 * total_blocks
    slot = 0
    for chunk_id in range(total_chunks):
        start, count, kv_part = _compression_chunk_window(chunk_id, total_blocks, 1)
        block_ids = list(range(start, start + count))
        valid_tokens = BLOCK_SIZE * count
        encoded = manager.encode(
            slot,
            [block_ids],
            kv_part,
            valid_tokens_by_group={0: valid_tokens},
        )
        encoded.event.synchronize()
        assert encoded.u8_bytes == encoded.codec_bytes > 0
        assert encoded.raw_tail_bytes == 0
        offset = kv_part * HEAD_SIZE
        region = slice(offset, offset + HEAD_SIZE)
        source = cache[block_ids, :, :, region].clone()
        cache[block_ids, :, :, region] = 0
        manager.decode(
            slot,
            [block_ids],
            kv_part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: valid_tokens},
        ).synchronize()
        assert torch.equal(cache[block_ids, :, :, region], source)
        slot = (slot + 1) % 2
