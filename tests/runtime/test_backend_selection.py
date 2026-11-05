# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Backend selection preserves supported fast paths and explicit overrides."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kvserve_v1.compression.runtime import CacheSpec, CompressionConfig
from kvserve_v1.compression.runtime import cpp_backend
from kvserve_v1.compression.runtime.layout import build_compression_layout
from kvserve_v1.compression.runtime.pipeline import CompressionPipeline


def layout(*, dtype=torch.bfloat16, axis="channel", groups=(0,)):
    parts = (SimpleNamespace(dtype=dtype, sections=(SimpleNamespace(axis=axis),)),)
    return SimpleNamespace(layer_parts=parts, fa_group_indices=frozenset(groups))


@pytest.fixture(autouse=True)
def nvidia_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(cpp_backend, "jit_toolchain_error", lambda **_: None)


def test_default_selects_supported_cpp_and_python_override(monkeypatch):
    monkeypatch.delenv("KVSERVE_COMPRESSION_BACKEND", raising=False)
    assert cpp_backend.requested_backend() == "auto"
    device = torch.device("cuda", 0)
    assert cpp_backend.select_backend(layout(), ("quantizer",), device)[0] == "cpp"
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    assert cpp_backend.select_backend(layout(), ("quantizer",), device)[0] == "python"


@pytest.mark.parametrize("candidate", [
    layout(axis="tensor"),
    layout(groups=(0, 1)),
    layout(dtype=torch.float16),
])
def test_auto_uses_python_for_unsupported_layouts(candidate):
    assert cpp_backend.select_backend(candidate, ("quantizer",),
                                      torch.device("cuda", 0), "auto")[0] == "python"
    with pytest.raises(ValueError):
        cpp_backend.select_backend(candidate, ("quantizer",),
                                   torch.device("cuda", 0), "cpp")


def test_auto_does_not_select_cpp_for_cpu_hip_or_raw_aggregate(monkeypatch):
    assert cpp_backend.select_backend(layout(), ("quantizer",),
                                      torch.device("cpu"), "auto")[0] == "python"
    assert cpp_backend.select_backend(layout(), ("aggregate",),
                                      torch.device("cuda", 0), "auto")[0] == "python"
    monkeypatch.setattr(torch.version, "hip", "6.0")
    assert cpp_backend.select_backend(layout(), ("quantizer",),
                                      torch.device("cuda", 0), "auto")[0] == "python"


def test_auto_without_jit_toolchain_uses_standard_python(monkeypatch):
    monkeypatch.setattr(cpp_backend, "jit_toolchain_error", lambda **_: "Ninja missing")
    assert cpp_backend.select_backend(layout(), ("codec",),
                                      torch.device("cuda", 0), "auto")[0] == "python"
    assert cpp_backend.select_backend(layout(), ("codec",),
                                      torch.device("cuda", 0), "cpp")[0] == "cpp"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unknown KVSERVE_COMPRESSION_BACKEND"):
        cpp_backend.requested_backend()


def test_backend_is_chosen_per_runtime_not_per_call(monkeypatch):
    config = CompressionConfig.from_extra_config({"compression": {
        "enable-compression-pipeline": True, "logical_blocks_per_chunk": 1,
    }})
    spec = CacheSpec(0, 1, 2, 8, torch.bfloat16)
    current_layout = build_compression_layout({"layer": spec}, config, {0: 0}, 1)
    cache = torch.zeros((2, 2, 1, 16), dtype=torch.bfloat16)
    stream = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", Mock())
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    first = CompressionPipeline(current_layout, "HND", 1, [], None, stream, {}, {})
    first.bind_kv_caches({"layer": cache})
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    second = CompressionPipeline(current_layout, "HND", 1, [], None, stream, {}, {})
    second.bind_kv_caches({"layer": cache})
    assert first._use_cpp()
    assert not second._use_cpp()
