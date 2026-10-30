"""Provider-only validation and deterministic TP head selection."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from kvserve_v1.compression.runtime import CacheSpec, CompressionConfig, cpp_backend
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
    device = torch.device("cuda", 3)
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
