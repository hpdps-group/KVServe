# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""CPU checks of fused capability selection and slot-size ownership."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from kvserve_v1.compression.runtime import CacheSpec, CompressionConfig, CompressionRuntime
from kvserve_v1.compression.runtime import cpp_backend, fused_runtime, memory
from kvserve_v1.compression.runtime.fused_geometry import select_geometry
from kvserve_v1.compression.runtime.layout import build_compression_layout


@pytest.fixture(autouse=True)
def fused_dependencies_present(monkeypatch):
    # Capability tests are hermetic; dependency failure is tested separately.
    monkeypatch.setattr(fused_runtime, "_fused_dependency_error", lambda: None)


def make_plan(layers=3, heads=4, head_size=128, capacity=64, **settings):
    config = CompressionConfig.from_extra_config({"compression": {
        "enable-compression-pipeline": True,
        "logical_blocks_per_chunk": capacity,
        **settings,
    }})
    specs = {
        f"model.blocks.{i}.attention": CacheSpec(i, 1, heads, head_size, torch.bfloat16)
        for i in range(layers)
    }
    layout = build_compression_layout(specs, config, {i: 0 for i in range(layers)}, layers)
    return SimpleNamespace(config=config, layout=layout)


def test_unmodified_defaults_keep_compression_when_fusion_is_ineligible(monkeypatch):
    config = CompressionConfig.from_extra_config({"compression": {
        "enable-compression-pipeline": True,
    }})
    assert config.logical_blocks_per_chunk == 1
    assert config.pipeline == ("transformer", "quantizer", "codec")
    specs = {"arbitrary.layer": CacheSpec(0, 1, 4, 128, torch.bfloat16)}
    layout = build_compression_layout(specs, config, {0: 0}, 1)
    plan = SimpleNamespace(config=config, layout=layout)
    assert select_geometry(plan, "HND", 1)[0] is None
    standard = Mock()
    monkeypatch.setattr(fused_runtime, "CompressionRuntime", standard)
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0")
    standard.assert_called_once_with(plan, "HND", 1, "cuda:0", execution="both")
    result.allocate_output_slots.assert_called_once_with(0)


def test_auto_uses_standard_when_optional_native_ans_is_missing(monkeypatch):
    plan = make_plan()
    standard = Mock()
    monkeypatch.setattr(fused_runtime, "CompressionRuntime", standard)
    monkeypatch.setattr(fused_runtime, "_fused_dependency_error", lambda: "nvCOMP 6 missing")
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0")
    assert result is standard.return_value
    with pytest.raises(RuntimeError, match="nvCOMP 6 missing"):
        fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", backend="fused")


def test_public_factory_delegates_to_runtime(monkeypatch):
    from kvserve_v1.compression import runtime

    factory = Mock()
    monkeypatch.setattr(fused_runtime, "create_runtime", factory)
    assert runtime.create_runtime("plan", execution="encode") is factory.return_value
    factory.assert_called_once_with("plan", execution="encode")


@pytest.mark.parametrize("layers,heads,capacity", [
    (1, 1, 16), (3, 2, 64), (7, 4, 80), (28, 4, 1024),
    (32, 8, 1024), (61, 16, 256),
])
def test_supported_geometry_is_derived_from_layout(layers, heads, capacity):
    plan = make_plan(layers, heads, capacity=capacity)
    geometry, _ = select_geometry(plan, "HND", 1)
    assert geometry is not None
    assert (geometry.layers, geometry.heads, geometry.head_size, geometry.capacity) == (
        layers, heads, 128, capacity
    )
    assert geometry.packed_bytes(1) == layers * heads * 128 * 16
    assert geometry.packed_bytes(capacity) == layers * heads * 128 * capacity


@pytest.mark.parametrize("layers,low_layers", [
    (1, 0), (2, 1), (3, 1), (28, 9), (32, 11), (63, 21),
])
def test_layer_split_uses_the_actual_model_layer_count(layers, low_layers):
    plan = make_plan(layers, low_key_num_levels=3, high_key_num_levels=11,
                     low_value_num_levels=5, high_value_num_levels=19)
    assert select_geometry(plan, "HND", 1)[0] is not None
    for part, low, high in ((0, 3, 11), (1, 5, 19)):
        levels = [p.sections[0].num_levels for p in plan.layout.parts(part)]
        assert levels == [high] * (layers - low_layers) + [low] * low_layers


@pytest.mark.parametrize("tokens,expected", [(1, 16), (16, 16), (17, 32), (79, 80), (80, 80)])
def test_packed_size_accounts_for_partial_frame_padding(tokens, expected):
    geometry, _ = select_geometry(make_plan(3, 2, capacity=80), "HND", 1)
    assert geometry.packed_bytes(tokens) == 3 * 2 * 128 * expected


@pytest.mark.parametrize("tokens", [0, -1, 81, True, 1.5])
def test_packed_size_rejects_invalid_frame_lengths(tokens):
    geometry, _ = select_geometry(make_plan(capacity=80), "HND", 1)
    with pytest.raises(ValueError):
        geometry.packed_bytes(tokens)


@pytest.mark.parametrize("heads,head_size,capacity", [
    (3, 128, 64), (4, 64, 64), (4, 256, 64),
    (4, 128, 1), (4, 128, 15), (4, 128, 17), (4, 128, 1040),
])
def test_unsupported_geometry_uses_standard_compression(monkeypatch, heads, head_size, capacity):
    plan = make_plan(heads=heads, head_size=head_size, capacity=capacity)
    assert select_geometry(plan, "HND", 1)[0] is None
    standard = Mock()
    monkeypatch.setattr(fused_runtime, "CompressionRuntime", standard)
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", output_slots=2)
    standard.assert_called_once_with(plan, "HND", 1, "cuda:0", execution="both")
    result.allocate_output_slots.assert_called_once_with(2)
    assert plan.config.pipeline == ("transformer", "quantizer", "codec")
    with pytest.raises(ValueError, match="Unsupported fused geometry"):
        fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", backend="fused")


def test_low_shared_memory_uses_standard_compression(monkeypatch):
    plan = make_plan()
    standard = Mock()
    monkeypatch.setattr(fused_runtime, "CompressionRuntime", standard)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(shared_memory_per_block_optin=224 * 1024))
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0")
    standard.assert_called_once_with(plan, "HND", 1, "cuda:0", execution="both")
    result.allocate_output_slots.assert_called_once_with(0)
    with pytest.raises(ValueError, match="225 KiB"):
        fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", backend="fused")


@pytest.mark.parametrize("backend", ["auto", "fused"])
def test_decode_capability_uses_fused_without_large_shared_memory(monkeypatch, backend):
    plan = make_plan()
    fused = Mock()
    monkeypatch.setattr(fused_runtime, "FusedCompressionRuntime", fused)
    # No query of the key encoder's shared-memory capability is required.
    query = Mock(side_effect=AssertionError("decode must not query K encode shared memory"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", query)
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", backend=backend,
                                          execution="decode")
    assert result is fused.return_value
    fused.assert_called_once_with(plan, "HND", 1, "cuda:0", output_slots=0,
                                  graph_lengths=None, execution="decode")
    query.assert_not_called()


@pytest.mark.parametrize("execution", ["both", "encode"])
def test_encode_capability_keeps_shared_memory_requirement(monkeypatch, execution):
    plan = make_plan()
    standard = Mock()
    monkeypatch.setattr(fused_runtime, "CompressionRuntime", standard)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(shared_memory_per_block_optin=99 * 1024))
    result = fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", execution=execution)
    assert result is standard.return_value
    standard.assert_called_once_with(plan, "HND", 1, "cuda:0", execution=execution)
    with pytest.raises(ValueError, match="225 KiB"):
        fused_runtime.create_runtime(plan, "HND", 1, "cuda:0", backend="fused",
                                    execution=execution)


def test_unknown_execution_capability_is_rejected_before_cuda():
    with pytest.raises(ValueError, match="execution capability"):
        fused_runtime.create_runtime(make_plan(), "HND", 1, "cuda:0", execution="restore")


@pytest.mark.parametrize("runtime_type", [CompressionRuntime, fused_runtime.FusedCompressionRuntime])
def test_disabled_execution_direction_rejects_submission(runtime_type):
    runtime = runtime_type.__new__(runtime_type)
    runtime.execution = "decode"
    with pytest.raises(RuntimeError, match="decode-only runtime cannot encode"):
        runtime.encode(0, [[0]], 0, valid_tokens_by_group={0: 1})
    with pytest.raises(RuntimeError, match="decode-only runtime cannot encode"):
        runtime.encode_into(0, [[0]], 0, valid_tokens_by_group={0: 1},
                            output=None, auxiliary=None)
    runtime.execution = "encode"
    with pytest.raises(RuntimeError, match="encode-only runtime cannot decode"):
        runtime.decode(0, [[0]], 0, 1, 1, valid_tokens_by_group={0: 1})


@dataclass
class _InitializationPlan:
    config: object
    layout: object
    reserved_bytes: int = 0


@pytest.mark.parametrize("execution", ["both", "encode", "decode"])
def test_fused_initialization_opts_into_shared_memory_only_for_encode(monkeypatch, execution):
    raw = make_plan()
    plan = _InitializationPlan(raw.config, raw.layout)
    module = SimpleNamespace(bind_shared=Mock(), GraphBudget=Mock())
    monkeypatch.setattr(fused_runtime, "load_fused", lambda device: module)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)

    def init_base(self, base_plan, *args, execution, **kwargs):
        self.plan, self.execution = base_plan, execution
        self.slots = [object()]
        self.stream = SimpleNamespace(synchronize=Mock())

    monkeypatch.setattr(CompressionRuntime, "__init__", init_base)
    monkeypatch.setattr(fused_runtime.FusedCompressionRuntime, "allocate_output_slots", Mock())
    runtime = fused_runtime.FusedCompressionRuntime(plan, "HND", 1, "cuda:0",
                                                    execution=execution)
    assert runtime.execution == execution
    if execution == "decode":
        module.bind_shared.assert_not_called()
    else:
        module.bind_shared.assert_called_once_with(0)


def test_direction_selects_graphs_without_changing_wire_profile(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 2000))
    monkeypatch.setattr(fused_runtime.cpp_backend, "build_meta",
                        lambda *args: torch.tensor([1], dtype=torch.int64))
    monkeypatch.setattr(CompressionRuntime, "bind_kv_caches",
                        lambda self, *args: setattr(self, "wire_profile", "shared-layout"))
    profiles = []
    for capability, directions in (("both", (True, False)), ("encode", (True,)),
                                   ("decode", (False,))):
        runtime = fused_runtime.FusedCompressionRuntime.__new__(fused_runtime.FusedCompressionRuntime)
        runtime.execution = capability
        runtime._state, runtime.device = "new", torch.device("cpu")
        runtime.geometry, _ = select_geometry(make_plan(), "HND", 1)
        runtime._validate_binding = Mock(return_value=64)
        runtime._graph_budget, runtime._graph_budget_bytes = None, 1
        runtime.graph_lengths = (16, 64)
        runtime._gpu_metadata = [[torch.empty(1, dtype=torch.int64) for _ in range(2)]]
        runtime._plans, runtime._plan_owners = {}, []
        slot = SimpleNamespace(completion_event=Mock(),
                               host_indices=[torch.empty(64, dtype=torch.int64)],
                               block_indices=[torch.empty(64, dtype=torch.int64)],
                               arena_a=torch.empty(8, dtype=torch.uint8))
        runtime.slots = [slot]
        runtime.pipelines = [SimpleNamespace(stream=SimpleNamespace(
            synchronize=Mock(), cuda_stream=123), codec=SimpleNamespace(
                _native_ans=SimpleNamespace(max_output=lambda size: 1)))]
        plans = [Mock(), Mock()]
        runtime._module = SimpleNamespace(ExecutionPlan=Mock(side_effect=plans))
        runtime.bind_kv_caches({}, {})
        for plan in plans:
            assert plan.build.call_args_list == [call(n, direction) for n in (16, 64)
                                                  for direction in directions]
        profiles.append(runtime.wire_profile)
    assert len(set(profiles)) == 1


@pytest.mark.parametrize("layout,blocks", [("NHD", 1), ("HND", 2)])
def test_non_hnd_or_multiple_physical_blocks_are_ineligible(layout, blocks):
    assert select_geometry(make_plan(), layout, blocks)[0] is None


def test_head_split_and_unusual_quantization_axes_are_ineligible():
    plan = make_plan(split_type="head")
    # Construct a real split instead of relying on a score file.
    specs = {f"layer.{i}": CacheSpec(i, 1, 4, 128, torch.bfloat16) for i in range(3)}
    plan.layout = build_compression_layout(specs, plan.config, {i: 1 for i in range(3)}, 3)
    assert select_geometry(plan, "HND", 1)[0] is None
    assert select_geometry(make_plan(axis_key="tensor"), "HND", 1)[0] is None
    assert select_geometry(make_plan(axis_value="channel"), "HND", 1)[0] is None


@pytest.mark.parametrize("change", [
    {"layer_name": "wrong-layer"}, {"num_heads": 8}, {"head_size": 64},
    {"block_size": 2}, {"dtype": torch.float16}, {"raw_bytes": 0}, {"u8_offset": 1},
])
def test_inconsistent_kv_parts_are_ineligible(change):
    plan = make_plan()
    parts = list(plan.layout.layer_parts)
    parts[1] = replace(parts[1], **change)
    plan.layout = replace(plan.layout, layer_parts=tuple(parts))
    assert select_geometry(plan, "HND", 1)[0] is None


def test_size_read_excludes_slot_reuse_until_host_value_is_copied():
    entered, resume = threading.Event(), threading.Event()
    lock = threading.Lock()
    runtime = SimpleNamespace(_locks=[lock], _generations=[1])
    codec = SimpleNamespace(_native_ans=SimpleNamespace(compressed_size=lambda: 73))

    def query():
        entered.set()
        assert resume.wait(5)
        return True

    size = fused_runtime._Size(codec, SimpleNamespace(query=query), generation=(runtime, 0, 1))
    outcome = []

    def read_size():
        try:
            outcome.append(size.buffer_size)
        except BaseException as exc:
            outcome.append(exc)

    reader = threading.Thread(target=read_size)
    reader.start()
    try:
        assert entered.wait(5)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        assert not acquired, "slot reuse could overwrite the host size while it is being consumed"
    finally:
        resume.set()
        reader.join(5)
    assert not reader.is_alive()
    assert outcome == [73]
    with lock:
        runtime._generations[0] += 1
        codec._native_ans.compressed_size = lambda: 91
    assert size.buffer_size == 73


def test_size_fails_closed_on_reuse_and_can_retry_before_completion():
    runtime = SimpleNamespace(_locks=[threading.Lock()], _generations=[1])
    event = SimpleNamespace(query=Mock(return_value=False))
    codec = SimpleNamespace(_native_ans=SimpleNamespace(compressed_size=lambda: 42))
    size = fused_runtime._Size(codec, event, generation=(runtime, 0, 1))
    with pytest.raises(RuntimeError, match="GPU completion"):
        _ = size.buffer_size
    event.query.return_value = True
    assert size.buffer_size == 42

    stale = fused_runtime._Size(codec, event, generation=(runtime, 0, 1))
    runtime._generations[0] = 2
    event.query.reset_mock()
    with pytest.raises(RuntimeError, match="slot reuse"):
        _ = stale.buffer_size
    event.query.assert_not_called()


@pytest.mark.parametrize("runtime_type", [CompressionRuntime, fused_runtime.FusedCompressionRuntime])
def test_encode_into_requires_one_complete_runtime_owned_bank(runtime_type):
    runtime = runtime_type.__new__(runtime_type)
    runtime.execution = "both"
    slots = [SimpleNamespace(arena_a=torch.empty(64, dtype=torch.uint8),
                             aux=torch.empty(32, dtype=torch.uint8)) for _ in range(2)]
    banks = [(torch.empty(64, dtype=torch.uint8), torch.empty(32, dtype=torch.uint8))
             for _ in range(2)]
    runtime.slots, runtime.output_slots = slots, banks
    assert runtime._validate_output_bank(0, slots[0].arena_a, slots[0].aux) is slots[0]
    assert runtime._validate_output_bank(0, *banks[1]) is slots[0]

    # Same storage, a different view, a mixed pair, another slot's buffers,
    # and arbitrary caller allocations must all fail before any work starts.
    bad_pairs = [
        (banks[0][0], banks[0][0]),
        (banks[0][0], banks[1][1]),
        (banks[0][0].view_as(banks[0][0]), banks[0][1]),
        (slots[1].arena_a, slots[1].aux),
        (torch.empty_like(banks[0][0]), banks[0][1]),
    ]
    for output, auxiliary in bad_pairs:
        with pytest.raises(ValueError, match="Runtime-owned bank"):
            runtime.encode_into(0, [[0]], 0, valid_tokens_by_group={0: 1},
                                output=output, auxiliary=auxiliary)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_default_probe_warms_actual_hadamard_without_humming(monkeypatch, backend):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", backend)
    monkeypatch.setitem(sys.modules, "humming", None)
    monkeypatch.setitem(sys.modules, "humming.ops", None)
    warm = Mock()
    monkeypatch.setattr(cpp_backend, "hadamard_transform", warm)

    fake_codec = Mock()
    fake_codec.compression_config.side_effect = lambda size: size
    fake_codec.decompression_config.side_effect = lambda config: config
    fake_codec.get_max_comp_buffer_size.return_value = 128
    fake_codec.encode.return_value = SimpleNamespace(buffer_size=16)
    nvidia = ModuleType("nvidia")
    nvidia.nvcomp = SimpleNamespace(Codec=Mock(return_value=fake_codec),
                                    as_array=lambda tensor, **kwargs: tensor)
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    stream = SimpleNamespace(cuda_stream=123, synchronize=Mock())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 2000))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(memory, "register_workspace_allocators", Mock())
    unregister = Mock()
    monkeypatch.setattr(memory, "unregister_workspace_allocators", unregister)

    plan = make_plan(capacity=16)
    specs = {"layer.0": CacheSpec(0, 1, 4, 128, torch.bfloat16)}
    result = memory.build_memory_plan(plan.config, specs, num_layers=1, device=torch.device("cpu"))
    assert result is not None
    warm.assert_called_once()
    source, output = warm.call_args.args
    assert source.shape == output.shape == (1, 128)
    assert source.dtype == output.dtype == torch.bfloat16
    fake_codec.encode.assert_called_once()
    fake_codec.decode.assert_called_once()
    unregister.assert_called_once_with([123])
