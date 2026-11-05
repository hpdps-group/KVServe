"""Provider-only validation and deterministic TP head selection."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from kvserve_v1.compression.runtime import (
    CacheSpec,
    CompressionConfig,
    CompressionRuntime,
    cpp_backend,
)
from kvserve_v1.compression.runtime import pipeline as compression_pipeline
from kvserve_v1.compression.runtime.layout import build_compression_layout
from kvserve_v1.compression.runtime.scores import build_head_selection


def config(**kwargs):
    return CompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True, **kwargs}}
    )


def test_static_profile_and_lossless_stages():
    assert config().pipeline == ("transformer", "quantizer", "codec")
    assert config(pipeline=["codec"]).pipeline == ("codec",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline", ["codec", "quantizer"]),
        ("high_key_num_levels", 1),
        ("slot_count", 0),
        ("axis_key", "unknown"),
    ],
)
def test_invalid_profile_is_rejected(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


@pytest.mark.parametrize("rank", range(8))
def test_head_scores_map_replicated_kv_heads(rank):
    spec = CacheSpec(0, 16, 1, 8, torch.bfloat16)
    selected = build_head_selection(
        {"arbitrary.name": spec},
        config(split_type="head", hybrid_ratio=0.5),
        rank,
        8,
        np.arange(4, dtype=np.float32).reshape(1, 4),
    )
    assert selected.low_counts[0] == (1 if rank < 4 else 0)
    assert selected.local_orders[0] == (0,)


@pytest.mark.parametrize("axis", ["channel", "token", "tensor"])
def test_fixed_profile_auxiliary_capacity_covers_sections(axis):
    spec = CacheSpec(0, 16, 2, 8, torch.bfloat16)
    layout = build_compression_layout(
        {"arbitrary.name": spec},
        config(
            split_type="head",
            axis_key=axis,
            axis_value=axis,
            logical_blocks_per_chunk=2,
        ),
        {0: 1},
        1,
    )
    assert layout.aux_arena_bytes == 2 * max(layout.aux_bytes_by_part)
    for part in layout.layer_parts:
        for section in part.sections:
            assert section.aux_min_offset < layout.aux_arena_bytes // 2
            assert section.aux_scale_offset < layout.aux_arena_bytes // 2


def _wire_profile_for(config, specs=None, num_model_layers=1):
    """Exercise Runtime's actual profile construction without allocating CUDA buffers."""
    if specs is None:
        specs = {"layer.0": CacheSpec(0, 1, 4, 128, torch.bfloat16)}
    selection = build_head_selection(specs, config, 0, 1)
    layout = build_compression_layout(specs, config, selection.low_counts, num_model_layers)
    runtime = object.__new__(CompressionRuntime)
    runtime.config = config
    runtime.layout = layout
    runtime.plan = SimpleNamespace(
        head_selection=selection
    )
    runtime.pipelines = [
        SimpleNamespace(
            kv_cache_layout="HND", blocks_per_logical=1, bind_kv_caches=Mock()
        )
    ]
    runtime._external_execution = False
    runtime._metadata_initialized = True
    runtime.bind_kv_caches({}, {name: 0 for name in specs})
    return runtime.wire_profile


def test_wire_profile_ignores_local_slot_count():
    producer = config(logical_blocks_per_chunk=16, slot_count=2)
    consumer = config(logical_blocks_per_chunk=16, slot_count=8)
    assert _wire_profile_for(producer) == _wire_profile_for(consumer)


def test_sparse_global_attention_layers_use_full_model_size_and_profile():
    specs = {
        f"layer.{i}": CacheSpec(i, 1, 2, 256, torch.bfloat16)
        for i in (0, 5, 31)
    }
    producer = config(logical_blocks_per_chunk=64, slot_count=2)
    consumer = config(logical_blocks_per_chunk=64, slot_count=8)
    selection = build_head_selection(specs, producer, 0, 1)
    layout = build_compression_layout(specs, producer, selection.low_counts, 32)
    assert [part.layer_index for part in layout.parts(0)] == [0, 5, 31]
    assert [part.sections[0].num_levels for part in layout.parts(0)] == [12, 12, 6]
    assert [part.sections[0].num_levels for part in layout.parts(1)] == [8, 8, 4]
    assert _wire_profile_for(producer, specs, 32) == _wire_profile_for(consumer, specs, 32)
    assert _wire_profile_for(producer, specs, 32) != _wire_profile_for(consumer, specs, 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline", ["quantizer", "codec"]),
        ("axis_key", "token"),
        ("axis_value", "channel"),
        ("high_key_num_levels", 16),
        ("low_value_num_levels", 5),
        ("seed", 17),
        ("logical_blocks_per_chunk", 32),
    ],
)
def test_wire_profile_distinguishes_codec_configuration(field, value):
    original = config(logical_blocks_per_chunk=16)
    updated = config(**{"logical_blocks_per_chunk": 16, field: value})
    assert _wire_profile_for(original) != _wire_profile_for(updated)


def test_registered_memory_skips_empty_output_buffers():
    runtime = object.__new__(CompressionRuntime)
    runtime.bulk = torch.empty(128, dtype=torch.uint8)
    payload_a = torch.empty(64, dtype=torch.uint8)
    payload_b = torch.empty(64, dtype=torch.uint8)
    auxiliary = torch.empty(16, dtype=torch.uint8)
    empty = torch.empty(0, dtype=torch.uint8)
    # Codec-only output has no auxiliary data. Never pass a zero-sized
    # registration or a null address to the transport engine.
    runtime.output_slots = [(payload_a, empty), (empty, empty), (payload_b, auxiliary)]

    ptrs, sizes = runtime.registered_memory()

    assert ptrs == [
        runtime.bulk.untyped_storage().data_ptr(),
        payload_a.data_ptr(),
        payload_b.data_ptr(),
        auxiliary.data_ptr(),
    ]
    assert sizes == [128, 64, 64, 16]


def _pipeline_for(config, backend):
    specs = {"layer.0": CacheSpec(0, 1, 4, 128, torch.bfloat16)}
    layout = build_compression_layout(specs, config, {0: 0}, 1)
    return compression_pipeline.CompressionPipeline(
        layout,
        "HND",
        1,
        [],
        None,
        SimpleNamespace(device=torch.device("cpu")),
        {},
        {},
        stages=config.pipeline,
        pipeline_backend=backend,
    )


def test_pipeline_backend_is_instance_scoped(monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", Mock())
    before = dict(os.environ)
    selected = config(pipeline=["quantizer"])
    cpp = _pipeline_for(selected, "cpp")
    python = _pipeline_for(selected, "python")
    inherited = _pipeline_for(selected, None)
    cache = {"layer.0": tuple(
        torch.empty((4, 4, 1, 128), dtype=torch.bfloat16) for _ in range(2)
    )}
    for pipeline in (cpp, python, inherited):
        pipeline.bind_kv_caches(cache)
    assert cpp._use_cpp()
    assert not python._use_cpp()
    assert not inherited._use_cpp()
    assert dict(os.environ) == before

    # Changing the legacy process default must not change either explicit instance.
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    assert cpp._use_cpp()
    assert not python._use_cpp()
    assert not inherited._use_cpp()
    later = _pipeline_for(selected, None)
    later.bind_kv_caches(cache)
    assert later._use_cpp()


@pytest.mark.parametrize(
    "instance_backend,global_backend,rejected",
    [("cpp", "python", True), ("python", "cpp", False)],
)
def test_binding_multiple_groups_uses_instance_backend(
    monkeypatch, instance_backend, global_backend, rejected
):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", global_backend)
    load = Mock()
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", load)
    selected = config(pipeline=["quantizer"])
    specs = {
        f"layer.{i}": CacheSpec(i, 1, 2, 8, torch.bfloat16) for i in range(2)
    }
    groups = {"layer.0": 0, "layer.1": 1}
    layout = build_compression_layout(specs, selected, {0: 0, 1: 0}, 2, groups)
    pipeline = compression_pipeline.CompressionPipeline(
        layout,
        "HND",
        1,
        [],
        None,
        SimpleNamespace(device=torch.device("cpu")),
        {},
        {},
        stages=selected.pipeline,
        pipeline_backend=instance_backend,
    )
    caches = {
        name: tuple(torch.empty((4, 2, 1, 8), dtype=torch.bfloat16) for _ in range(2))
        for name in specs
    }

    if rejected:
        with pytest.raises(ValueError, match="requires one FA cache group"):
            pipeline.bind_kv_caches(caches)
    else:
        pipeline.bind_kv_caches(caches)
        assert set(pipeline.kv_caches) == set(caches)
    load.assert_not_called()


@pytest.mark.parametrize(
    "instance_backend,global_backend,prepare_backend,expect_load",
    [
        ("cpp", "python", True, True),
        ("python", "cpp", True, False),
        (None, "cpp", True, True),
        (None, "python", True, False),
        ("cpp", "python", False, False),
    ],
)
def test_binding_loads_selected_instance_backend_only(
    monkeypatch, instance_backend, global_backend, prepare_backend, expect_load
):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", global_backend)
    load = Mock()
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", load)
    # No transformer: the separate, unconditional Hadamard CUDA extension
    # would otherwise load even for the Python pipeline backend.
    pipeline = _pipeline_for(config(pipeline=["quantizer"]), instance_backend)
    caches = {
        "layer.0": tuple(
            torch.empty((4, 4, 1, 128), dtype=torch.bfloat16) for _ in range(2)
        )
    }
    pipeline.bind_kv_caches(caches, prepare_backend=prepare_backend)
    assert set(pipeline.kv_caches) == set(caches)
    if expect_load:
        load.assert_called_once_with(torch.device("cpu"))
    else:
        load.assert_not_called()


def test_cpp_tensor_axis_is_checked_only_when_quantizing(monkeypatch):
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", Mock())
    codec_only = config(pipeline=["codec"], axis_key="tensor", axis_value="tensor")
    pipeline = _pipeline_for(codec_only, "cpp")
    cache = {"layer.0": tuple(
        torch.empty((4, 4, 1, 128), dtype=torch.bfloat16) for _ in range(2)
    )}
    pipeline.bind_kv_caches(cache)
    assert pipeline._use_cpp()

    quantized = config(pipeline=["quantizer"], axis_key="tensor")
    with pytest.raises(ValueError, match="tensor-axis"):
        _pipeline_for(quantized, "cpp")


def test_unknown_explicit_backend_does_not_silently_use_python(monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpo")
    with pytest.raises(ValueError, match="Unknown KVSERVE_COMPRESSION_BACKEND"):
        cpp_backend.enabled()


def test_cpu_transform_without_humming(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "humming", None)
    monkeypatch.setitem(sys.modules, "humming.ops", None)
    matrix = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(3):
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    source = torch.randint(-1, 2, (33, 8)).to(torch.bfloat16)
    expected = ((source.double() @ matrix) / (8**0.5)).to(torch.bfloat16)
    output = torch.empty_like(source)
    compression_pipeline.CompressionPipeline._hadamard_transform(source, output, 8)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_cpp_jit_uses_worker_device_and_respects_explicit_arch(monkeypatch):
    device = torch.device("cuda", 3)
    capability = Mock(return_value=(12, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(cpp_backend, "load", Mock())
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cpp_backend, "_module", None)
    cpp_backend.load_kvs_compress(device)
    capability.assert_called_once_with(device)

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.0"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0;12.0")
    monkeypatch.setattr(cpp_backend, "_module", None)
    cpp_backend.load_kvs_compress(device)
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0;12.0"


@pytest.mark.parametrize(
    "backend,stages",
    [("cpp", ("quantizer",)), ("python", ("transformer", "quantizer"))],
)
def test_cache_binding_loads_backend_and_keeps_asymmetric_kv_views(
    monkeypatch, backend, stages
):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", backend)
    load = Mock()
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", load)
    parts = tuple(
        SimpleNamespace(
            layer_name="layer",
            kv_part=i,
            head_size=size,
            dtype=torch.bfloat16,
            num_heads=2,
            block_size=1,
            sections=(),
        )
        for i, size in enumerate((8, 16))
    )
    # This fixture uses a real CPU cache, so its stream must report CPU too.
    # The separate JIT test above checks propagation of a CUDA worker device.
    device = torch.device("cpu")
    pipeline = compression_pipeline.CompressionPipeline(
        SimpleNamespace(
            layer_parts=parts,
            fa_group_indices={0},
            parts=lambda kv: tuple(part for part in parts if part.kv_part == kv),
        ),
        "LBHNC",
        1,
        [],
        None,
        SimpleNamespace(device=device),
        {},
        {},
        stages=stages,
    )
    cache = torch.arange(96, dtype=torch.bfloat16).reshape(2, 2, 1, 24)
    pipeline.bind_kv_caches({"layer": cache})
    load.assert_called_once_with(device)
    for part, expected in zip(parts, (cache[..., :8], cache[..., 8:])):
        view = pipeline._cache_part(part)
        assert view.data_ptr() == expected.data_ptr()
        torch.testing.assert_close(view, expected)
    with pytest.raises(ValueError, match="Unsupported compression cache"):
        pipeline.bind_kv_caches({"layer": cache.squeeze(2)})

    # Cache rebinding must discard metadata containing old cache pointers.
    build = Mock(side_effect=[torch.tensor([1]), torch.tensor([2])])
    monkeypatch.setattr(cpp_backend, "build_meta", build)
    pipeline.slots.append(SimpleNamespace())
    original = pipeline._cpp_meta(0, 0)
    assert pipeline._cpp_meta(0, 0) is original
    assert build.call_count == 1
    pipeline.bind_kv_caches({"layer": cache.clone()})
    assert pipeline._cpp_meta(0, 0) is not original
    assert build.call_count == 2
