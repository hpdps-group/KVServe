"""Public runtime restoration, numerical and workspace contracts."""

import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from kvserve_v1.compression.runtime import (
    CacheSpec,
    CompressionConfig,
    CompressionRuntime,
    build_memory_plan,
    cpp_backend,
)
from kvserve_v1.compression.runtime.memory import FixedWorkspaceAllocator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
LAYER_NAME = "model.layers.0.self_attn"


def chunk_window(chunk_id, logical_block_count, chunk_blocks):
    first = chunk_id // 2 * chunk_blocks
    return first, min(chunk_blocks, logical_block_count - first), chunk_id % 2


@pytest.fixture
def make_manager(monkeypatch):
    torch.manual_seed(0)
    created = []

    def factory(
        compression=None,
        num_layers=1,
        raw_aggregate=None,
        blocks_per_logical=1,
        block_size=16,
        head_size=8,
    ):
        extra = {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "layer",
                **(compression or {}),
            }
        }
        if raw_aggregate is not None:
            extra["raw_aggregate"] = raw_aggregate
        config = CompressionConfig.from_extra_config(extra)
        dependencies = []
        if config.uses_codec:
            dependencies.append("nvidia.nvcomp")
        for dependency in dependencies:
            if os.environ.get("VLLM_TEST_REQUIRE_MOONCAKE_COMPRESSION") == "1":
                importlib.import_module(dependency)
            else:
                pytest.importorskip(dependency)
        specs = {
            f"model.layers.{i}.self_attn": CacheSpec(
                layer_index=i,
                block_size=block_size,
                num_kv_heads=2,
                head_size=head_size,
                dtype=torch.bfloat16,
            )
            for i in range(num_layers)
        }
        plan = build_memory_plan(
            config, specs, blocks_per_logical=blocks_per_logical, num_layers=num_layers
        )
        assert plan is not None
        manager = CompressionRuntime(
            plan, "LBHNC", blocks_per_logical, torch.device("cuda", 0)
        )
        created.append(manager)
        caches = {
            name: torch.rand(
                4 * blocks_per_logical,
                2,
                block_size // blocks_per_logical,
                2 * head_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for name in specs
        }
        manager.bind_kv_caches(caches, dict.fromkeys(specs, 0))
        return manager, caches[LAYER_NAME]

    yield factory
    for manager in reversed(created):
        manager.shutdown()


def _copy_wire(producer, consumer, encoded, part, tokens):
    source, target = producer.slots[part], consumer.slots[1 - part]
    target.arena_a[: encoded.codec_bytes].copy_(source.arena_a[: encoded.codec_bytes])
    aux = producer.aux_bytes(part, {0: tokens})
    target.aux[:aux].copy_(source.aux[:aux])
    target.raw_tail[: encoded.raw_tail_bytes].copy_(
        source.raw_tail[: encoded.raw_tail_bytes]
    )
    torch.accelerator.synchronize()


def _assert_restored(producer, consumer, part, tokens, atol=0.06):
    factor = producer.pipelines[0].blocks_per_logical
    block_size = producer.layout.layer_parts[0].block_size
    count = (tokens + block_size - 1) // block_size
    source_ids = [b * factor + i for b in [2, 0][:count] for i in range(factor)]
    target_ids = [b * factor + i for b in [1, 3][:count] for i in range(factor)]
    untouched = sorted(set(range(4 * factor)) - set(target_ids))
    for name, source in producer.pipelines[0].kv_caches.items():
        target = consumer.pipelines[0].kv_caches[name]
        head_size = source.shape[-1] // 2
        section = slice(part * head_size, (part + 1) * head_size)
        expected = source[source_ids, :, :, section].permute(0, 2, 1, 3).flatten(0, 1)
        actual = target[target_ids, :, :, section].permute(0, 2, 1, 3).flatten(0, 1)
        stages = producer.config.pipeline
        tolerance = atol if "quantizer" in stages or "transformer" in stages else 0
        torch.testing.assert_close(
            actual[:tokens], expected[:tokens], atol=tolerance, rtol=0
        )
        if "aggregate" not in stages:
            tail = tokens // block_size * block_size
            assert torch.equal(actual[tail:tokens], expected[tail:tokens])
        assert (target[untouched] == -9).all()


@pytest.mark.parametrize(
    "stages,tokens,factor,block_size",
    [
        (None, 16, 1, 16),
        (["aggregate"], 17, 2, 16),
        ([], 32, 1, 16),
        (["codec"], 32, 1, 16),
        (["transformer"], 32, 1, 16),
        (["quantizer"], 32, 1, 16),
        (["transformer", "codec"], 32, 1, 16),
        (["quantizer", "codec"], 31, 4, 16),
        (["transformer", "quantizer"], 31, 4, 16),
        (["transformer", "quantizer", "codec"], 1, 1, 16),
        (["transformer", "quantizer", "codec"], 15, 1, 16),
        (["transformer", "quantizer", "codec"], 16, 1, 16),
        (["transformer", "quantizer", "codec"], 17, 2, 16),
        (["transformer", "quantizer", "codec"], 31, 4, 16),
        (["transformer", "quantizer", "codec"], 32, 1, 16),
        (["transformer", "quantizer", "codec"], 257, 1, 256),
    ],
)
def test_peer_buffers_restore_layers_blocks_and_partial_tail(
    make_manager, stages, tokens, factor, block_size
):
    config = {"logical_blocks_per_chunk": 2}
    if stages is not None:
        config.update(
            pipeline=stages,
            **dict.fromkeys(
                (
                    "high_key_num_levels",
                    "low_key_num_levels",
                    "high_value_num_levels",
                    "low_value_num_levels",
                ),
                256,
            ),
        )
    num_layers = 1 if stages is None else 3
    producer, _ = make_manager(
        config, num_layers=num_layers, blocks_per_logical=factor, block_size=block_size
    )
    consumer, _ = make_manager(
        config, num_layers=num_layers, blocks_per_logical=factor, block_size=block_size
    )
    for target in consumer.pipelines[0].kv_caches.values():
        target.fill_(-9)
    torch.accelerator.synchronize()
    count = (tokens + block_size - 1) // block_size
    for part in (0, 1):
        encoded = producer.encode(
            part, [[2, 0][:count]], part, valid_tokens_by_group={0: tokens}
        )
        encoded.event.synchronize()
        _copy_wire(producer, consumer, encoded, part, tokens)
        consumer.decode(
            1 - part,
            [[1, 3][:count]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: tokens},
        ).synchronize()
        _assert_restored(
            producer, consumer, part, tokens, atol=0.3 if stages is None else 0.06
        )


def test_nonidentity_head_order_restores_original_heads(make_manager, tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps([[1.0, 0.0]]))
    config = {
        "split_type": "head",
        "score_path": str(path),
        "hybrid_ratio": 0.5,
        "logical_blocks_per_chunk": 2,
        **dict.fromkeys(
            (
                "high_key_num_levels",
                "low_key_num_levels",
                "high_value_num_levels",
                "low_value_num_levels",
            ),
            256,
        ),
    }
    producer, _ = make_manager(config)
    consumer, target = make_manager(config)
    assert producer.plan.head_selection.local_orders[0] == (1, 0)
    target.fill_(-9)
    torch.accelerator.synchronize()
    for part in (0, 1):
        encoded = producer.encode(part, [[2]], part, valid_tokens_by_group={0: 16})
        encoded.event.synchronize()
        _copy_wire(producer, consumer, encoded, part, 16)
        consumer.decode(
            1 - part,
            [[1]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 16},
        ).synchronize()
        _assert_restored(producer, consumer, part, 16)


def test_raw_aggregate_multiple_chunks_reuse_slots_exactly(make_manager):
    manager, cache = make_manager(
        raw_aggregate={"slot_count": 2, "logical_blocks_per_chunk": 1}
    )
    for chunk in range(8):
        block, _count, part = chunk_window(chunk, 4, 1)
        slot = chunk % 2
        section = slice(part * 8, (part + 1) * 8)
        source = cache[block, :, :, section].clone()
        encoded = manager.encode(slot, [[block]], part, valid_tokens_by_group={0: 16})
        encoded.event.synchronize()
        cache[block, :, :, section].zero_()
        torch.accelerator.synchronize()
        manager.decode(
            slot,
            [[block]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 16},
        ).synchronize()
        assert encoded.codec_bytes == encoded.u8_bytes and encoded.raw_tail_bytes == 0
        assert torch.equal(cache[block, :, :, section], source)


def test_aggregate_indices_use_slot_stream_and_sparse_group(make_manager):
    manager, cache = make_manager({"pipeline": ["aggregate"]})
    manager.bind_kv_caches({LAYER_NAME: cache}, {LAYER_NAME: 1})
    cache[0].fill_(1)
    cache[1].fill_(2)
    torch.accelerator.synchronize()
    manager.encode(0, [[], [0]], 0, valid_tokens_by_group={1: 16}).event.synchronize()
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        torch.cuda._sleep(20_000_000)
        encoded = manager.encode(0, [[], [1]], 0, valid_tokens_by_group={1: 16})
    encoded.event.synchronize()
    caller.synchronize()
    with torch.cuda.stream(caller):
        torch.cuda._sleep(20_000_000)
        event = manager.decode(
            0,
            [[], [3]],
            0,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={1: 16},
        )
    event.synchronize()
    caller.synchronize()
    assert (cache[3, :, :, :8] == 2).all()
    assert (cache[0] == 1).all()


def test_concurrent_slots_do_not_cross_request_writes(make_manager):
    manager, cache = make_manager({"pipeline": ["codec"], "slot_count": 2})
    expected = cache.clone()
    torch.accelerator.synchronize()

    def encode(slot):
        with torch.accelerator.device_index(0):
            return manager.encode(slot, [[slot]], 0, valid_tokens_by_group={0: 16})

    def decode(slot, encoded):
        with torch.accelerator.device_index(0):
            return manager.decode(
                slot,
                [[slot]],
                0,
                encoded.codec_bytes,
                encoded.u8_bytes,
                valid_tokens_by_group={0: 16},
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        chunks = list(pool.map(encode, (0, 1)))
        for chunk in chunks:
            chunk.event.synchronize()
        cache[:2, :, :, :8].zero_()
        torch.accelerator.synchronize()
        for event in pool.map(decode, (0, 1), chunks):
            event.synchronize()
    assert torch.equal(cache, expected)


def test_probe_and_shutdown_preserve_other_manager_workspace(make_manager, monkeypatch):
    first, first_cache = make_manager({"pipeline": ["codec"]})
    second, cache = make_manager({"pipeline": ["codec"]})
    used = set()
    allocate = FixedWorkspaceAllocator.__call__

    def tracked(self, nbytes, stream):
        result = allocate(self, nbytes, stream)
        assert (
            self.buffer.data_ptr()
            <= result.ptr
            < self.buffer.data_ptr() + self.buffer.numel()
        )
        used.add(self.buffer.data_ptr())
        return result

    monkeypatch.setattr(FixedWorkspaceAllocator, "__call__", tracked)
    for index, (manager, target) in enumerate(
        ((first, first_cache), (second, cache), (second, cache))
    ):
        if index == 2:
            first.shutdown()
        used.clear()
        source = target[0, :, :, :8].clone()
        encoded = manager.encode(0, [[0]], 0, valid_tokens_by_group={0: 16})
        encoded.event.synchronize()
        target[0, :, :, :8].zero_()
        torch.accelerator.synchronize()
        manager.decode(
            0,
            [[0]],
            0,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 16},
        ).synchronize()
        assert torch.equal(target[0, :, :, :8], source)
        assert used == {manager.workspace_allocators[0].buffer.data_ptr()}


def test_external_nvcomp_stream_survives_manager_shutdown(make_manager):
    from nvidia import nvcomp

    manager, _ = make_manager({"pipeline": ["codec"]})
    stream = torch.cuda.Stream()
    codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
    for active in (True, False):
        if not active:
            manager.shutdown()
        with torch.cuda.stream(stream):
            source = torch.randint(0, 256, (65536,), device="cuda", dtype=torch.uint8)
            encoded = codec.encode(
                nvcomp.as_array(source, cuda_stream=stream.cuda_stream)
            )
            decoded = torch.as_tensor(codec.decode(encoded), device="cuda").view(
                torch.uint8
            )
        stream.synchronize()
        assert torch.equal(decoded, source)


def test_transform_matches_independent_reference(make_manager):
    manager, cache = make_manager({"pipeline": ["transformer"]}, head_size=128)
    # Independent Sylvester matrix: two matching bugs must not pass a roundtrip.
    hadamard = torch.ones((1, 1), dtype=torch.float64)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat(
            (torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0
        )
    for part in (0, 1):
        source = cache[0, :, :, part * 128 : (part + 1) * 128].clone()
        encoded = manager.encode(0, [[0]], part, valid_tokens_by_group={0: 16})
        encoded.event.synchronize()
        signs = manager.signs[(LAYER_NAME, part)].cpu().double().reshape(2, 128)
        expected = (
            (source.permute(1, 0, 2).cpu().double() * signs) @ hadamard / 128**0.5
        )
        actual = (
            manager.slots[0]
            .arena_a[: encoded.codec_bytes]
            .view(torch.bfloat16)
            .reshape(16, 2, 128)
            .cpu()
            .double()
        )
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.03)


@pytest.mark.parametrize("axis", ["channel", "token", "tensor"])
def test_quantizer_matches_numpy_reference(make_manager, monkeypatch, axis):
    if axis == "tensor" and cpp_backend.enabled():
        monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    config = {
        "pipeline": ["quantizer"],
        "axis_key": axis,
        "axis_value": axis,
        **dict.fromkeys(
            (
                "high_key_num_levels",
                "low_key_num_levels",
                "high_value_num_levels",
                "low_value_num_levels",
            ),
            16,
        ),
    }
    manager, cache = make_manager(config)
    t = torch.arange(16, device="cuda").reshape(1, -1, 1)
    h = torch.arange(2, device="cuda").reshape(-1, 1, 1)
    for part in (0, 1):
        section = slice(part * 8, (part + 1) * 8)
        cache[0, :, :, section].mul_(1 + t).add_(h * 20)
        x = (
            cache[0, :, :, section]
            .permute(1, 0, 2)
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        axes = {"channel": (0,), "token": (1, 2), "tensor": (0, 1, 2)}[axis]
        low, high = x.min(axis=axes, keepdims=True), x.max(axis=axes, keepdims=True)
        step = np.maximum(high - low, 1e-5) / 15
        expected = np.clip(np.rint((x - low) / step), 0, 15)
        torch.accelerator.synchronize()
        encoded = manager.encode(0, [[0]], part, valid_tokens_by_group={0: 16})
        encoded.event.synchronize()
        actual = (
            manager.slots[0].arena_a[: encoded.u8_bytes].cpu().numpy().reshape(x.shape)
        )
        assert np.abs(actual.astype(np.float64) - expected).max() <= 1
        cache[1].zero_()
        torch.accelerator.synchronize()
        manager.decode(
            0,
            [[1]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 16},
        ).synchronize()
        result = cache[1, :, :, section].permute(1, 0, 2).float().cpu().numpy()
        bound = step * 0.65 + np.maximum(np.abs(low), np.abs(high)) * 0.02
        assert np.all(np.abs(result - x) <= bound)


@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize(
    "pattern,levels",
    [("random", 4), ("random", 6), ("constant", 12), ("tiny", 8), ("zeros", 16)],
)
def test_quantization_preserves_bf16_rounding_and_aux(
    make_manager, axis, pattern, levels
):
    config = {
        "pipeline": ["quantizer"],
        "logical_blocks_per_chunk": 2,
        "axis_key": axis,
        "axis_value": axis,
        **dict.fromkeys(
            (
                "high_key_num_levels",
                "low_key_num_levels",
                "high_value_num_levels",
                "low_value_num_levels",
            ),
            levels,
        ),
    }
    manager, cache = make_manager(config)
    cache.sub_(0.5)
    if pattern == "constant":
        cache.fill_(-3.125)
    elif pattern == "tiny":
        cache.mul_(1e-6)
    elif pattern == "zeros":
        cache.zero_()
        cache[..., ::2] = -0.0
    torch.accelerator.synchronize()
    order = manager.pipelines[0].head_orders[LAYER_NAME].cpu()
    for part in (0, 1):
        x = cache[:2, :, :, part * 8 : (part + 1) * 8].cpu()
        x = x.index_select(1, order).permute(0, 2, 1, 3)
        dims = (0, 1) if axis == "channel" else (2, 3)
        minimum = x.amin(dims, keepdim=True)
        maximum = x.amax(dims, keepdim=True)
        scale = ((maximum - minimum).clamp_min(1e-5).float() / (levels - 1)).to(x.dtype)
        inverse = scale.float().reciprocal().to(x.dtype)
        shifted = (x.float() - minimum.float()).to(x.dtype)
        expected = (shifted.float() * inverse.float()).to(x.dtype).float()
        expected = expected.round().clamp(0, levels - 1).to(torch.uint8)
        encoded = manager.encode(0, [[0, 1]], part, valid_tokens_by_group={0: 32})
        encoded.event.synchronize()
        actual = manager.slots[0].arena_a[: encoded.u8_bytes].cpu().reshape(x.shape)
        assert torch.equal(actual, expected)
        section = manager.layout.parts(part)[0].sections[0]
        for offset, value in (
            (section.aux_min_offset, minimum),
            (section.aux_scale_offset, scale),
        ):
            actual_aux = manager.slots[0].aux[offset : offset + value.numel() * 2].cpu()
            if pattern == "zeros" and offset == section.aux_min_offset:
                # The sign of a tied zero depends on CPU/GPU reduction order.
                # Both signs encode the same minimum; keep all other metadata
                # byte-exact, including the scale of an all-zero input.
                torch.testing.assert_close(
                    actual_aux.view(value.dtype), value.flatten(), rtol=0, atol=0
                )
            else:
                assert torch.equal(
                    actual_aux, value.contiguous().view(torch.uint8).flatten()
                )


@pytest.mark.parametrize(
    "field,value,compatible",
    [
        ("seed", 123, False),
        ("axis_key", "token", False),
        ("high_key_num_levels", 32, False),
        ("logical_blocks_per_chunk", 2, False),
        ("slot_count", 4, True),
    ],
)
def test_wire_profile_compares_representation_not_local_slots(
    make_manager, field, value, compatible
):
    producer, _ = make_manager({"slot_count": 1})
    consumer, _ = make_manager({"slot_count": 1, field: value})
    assert (producer.wire_profile == consumer.wire_profile) == compatible


@pytest.mark.parametrize(
    "encoder_backend,decoder_backend", [("python", "cpp"), ("cpp", "python")]
)
@pytest.mark.parametrize("axis", ["channel", "token"])
def test_cross_backend_wire_decode(
    make_manager, monkeypatch, encoder_backend, decoder_backend, axis
):
    config = {
        "axis_key": axis,
        "axis_value": axis,
        "logical_blocks_per_chunk": 2,
        **dict.fromkeys(
            (
                "high_key_num_levels",
                "low_key_num_levels",
                "high_value_num_levels",
                "low_value_num_levels",
            ),
            256,
        ),
    }
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", encoder_backend)
    producer, _ = make_manager(config)
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", decoder_backend)
    consumer, target = make_manager(config)
    target.fill_(-9)
    torch.accelerator.synchronize()
    for part in (0, 1):
        monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", encoder_backend)
        encoded = producer.encode(part, [[2, 0]], part, valid_tokens_by_group={0: 31})
        encoded.event.synchronize()
        _copy_wire(producer, consumer, encoded, part, 31)
        monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", decoder_backend)
        consumer.decode(
            1 - part,
            [[1, 3]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 31},
        ).synchronize()
        _assert_restored(producer, consumer, part, 31)


def test_cpp_sparse_fa_group_uses_cpp_without_fallback(make_manager, monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    manager, cache = make_manager({"pipeline": ["codec"]})
    manager.bind_kv_caches({LAYER_NAME: cache}, {LAYER_NAME: 1})
    expected = cache[0, :, :, :8].clone()
    calls = []
    original = cpp_backend.encode_chunk

    def encode(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(cpp_backend, "encode_chunk", encode)
    encoded = manager.encode(0, [[], [0]], 0, valid_tokens_by_group={1: 16})
    encoded.event.synchronize()
    cache[1].zero_()
    torch.accelerator.synchronize()
    manager.decode(
        0,
        [[], [1]],
        0,
        encoded.codec_bytes,
        encoded.u8_bytes,
        valid_tokens_by_group={1: 16},
    ).synchronize()
    assert calls == [True]
    assert torch.equal(cache[1, :, :, :8], expected)


def test_cpp_multiple_fa_groups_rejected_at_bind(make_manager, monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    manager, _ = make_manager(num_layers=2)
    caches = manager.pipelines[0].kv_caches
    with pytest.raises(ValueError, match="requires one FA cache group"):
        manager.bind_kv_caches(caches, {name: i for i, name in enumerate(caches)})
