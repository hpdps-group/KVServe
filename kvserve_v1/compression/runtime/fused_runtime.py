# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Instance-scoped fused Runtime, shared by framework transfer adapters.

No SGLang/Mooncake imports or process-global pipeline rewrites. Unsupported
geometry uses the normal KVServe compression Runtime through create_runtime().
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib
import logging
from pathlib import Path
import threading

import torch
from torch.utils.cpp_extension import load

from . import cpp_backend
from .codec import EncodedChunk
from .fused_geometry import select_geometry
from .manager import CompressionRuntime

logger = logging.getLogger(__name__)
_module = None
_module_lock = threading.Lock()


def load_fused(device):
    global _module
    with _module_lock:
        if _module is None:
            library = Path(importlib.import_module('nvidia.libnvcomp').__file__).resolve().parent
            source = Path(__file__).resolve().parent / 'csrc' / 'fused'
            with torch.cuda.device(device):
                _module = load(
                    name='kvserve_fused_execution_v1',
                    sources=[str(source / 'bindings.cpp'), str(source / 'kvs_fused.cu')],
                    extra_include_paths=[str(library / 'include')],
                    extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
                    extra_ldflags=['-L' + str(library / 'lib64'), '-l:libnvcomp.so.6',
                                   '-Wl,-rpath,' + str(library / 'lib64'), '-lcudart'],
                    with_cuda=True, verbose=False,
                )
        return _module


class _Size:
    def __init__(self, codec, event, owners=(), generation=None):
        self.codec, self.event, self.owners = codec, event, owners
        self.generation, self.value = generation, None

    @property
    def buffer_size(self):
        if self.value is not None:
            return self.value

        def consume():
            if not self.event.query():
                raise RuntimeError('Encoded size requires GPU completion')
            self.value = self.codec._native_ans.compressed_size()
            return self.value

        if self.generation is None:
            return consume()
        runtime, slot, expected = self.generation
        # A slot owns one host-size cell. Keep its generation check and read
        # atomic with respect to a new encode/decode submission on that slot.
        with runtime._locks[slot]:
            if self.value is not None:
                return self.value
            if runtime._generations[slot] != expected:
                raise RuntimeError('Encoded size was not consumed before slot reuse')
            return consume()


class _NativeCodec:
    def __init__(self, stream, sizes, workspace, module, capacity_sizes):
        self.stream = stream
        self._native_device_size = torch.empty(1, dtype=torch.int64, device=stream.device)
        self._native_host_size = torch.empty(1, dtype=torch.int64, pin_memory=True)
        self._native_workspace = workspace
        self._native_ans = module.NativeANS(
            stream.cuda_stream, workspace.data_ptr(), workspace.numel(),
            self._native_device_size.data_ptr(), self._native_host_size.data_ptr(),
            sorted(set(sizes) | set(capacity_sizes)),
        )

    def close(self):
        self._native_ans.close()


class FusedCompressionRuntime(CompressionRuntime):
    def __init__(self, plan, kv_cache_layout, blocks_per_logical, device, *,
                 output_slots=0, graph_lengths=None, execution='both'):
        if execution not in ('both', 'encode', 'decode'):
            raise ValueError('Unknown KVServe execution capability')
        geometry, reason = select_geometry(plan, kv_cache_layout, blocks_per_logical)
        if geometry is None:
            raise ValueError('Unsupported fused compression geometry: ' + reason)
        self.geometry = geometry
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._state, self._plans, self._codecs = 'new', {}, []
        self._plan_owners, self._graph_budget = [], None
        self._locks = [threading.Lock() for _ in range(plan.config.slot_count)]
        self._generations = [0] * plan.config.slot_count
        # Whole frames dominate serving traffic. Other exact lengths use the
        # same compressed eager kernels; do not instantiate 64 mostly cold
        # graphs per direction/slot merely because a tail length is possible.
        lengths = (geometry.capacity,) if graph_lengths is None else graph_lengths
        self.graph_lengths = tuple(sorted(set(lengths)))
        if any(type(n) is not int or not 0 < n <= geometry.capacity for n in self.graph_lengths):
            raise ValueError('Invalid prebuilt fused Graph length')
        self._module = load_fused(self.device)
        sizes = [geometry.packed_bytes(n) for n in range(16, geometry.capacity + 1, 16)]

        def codec_factory(stream, original_sizes, workspace):
            codec = _NativeCodec(stream, original_sizes, workspace, self._module, sizes)
            self._codecs.append(codec)
            return codec

        with torch.cuda.device(self.device):
            if execution != 'decode':
                self._module.bind_shared(self.device.index)
            super().__init__(plan, kv_cache_layout, blocks_per_logical, self.device,
                             codec_factory=codec_factory, external_execution=True,
                             execution=execution)
            try:
                count = cpp_backend.KVS_META_HEADER_FIELDS + geometry.layers * (
                    cpp_backend.KVS_PART_FIELDS + cpp_backend.KVS_SECTION_FIELDS)
                self._gpu_metadata = [[torch.empty(count, dtype=torch.int64, device=self.device)
                                       for _ in (0, 1)] for _ in self.slots]
                # Reserve CUDA Graph headroom BEFORE the framework sizes native KV.
                # It is released before capture. Fail closed if measured retention
                # exceeds this conservative startup allowance.
                graph_bytes = ((16 << 20) + len(self.slots) * len(self.graph_lengths) * 4 * 65536
                               if self.graph_lengths else 0)
                self._graph_budget_bytes = graph_bytes
                if graph_bytes:
                    self._graph_budget = self._module.GraphBudget(self.device.index, graph_bytes)
                extra = sum(t.numel() * t.element_size() for row in self._gpu_metadata for t in row)
                extra += len(self.slots) * 8 + graph_bytes
                self.plan = replace(self.plan, reserved_bytes=self.plan.reserved_bytes + extra)
                self.allocate_output_slots(output_slots)
                self.stream.synchronize()
            except BaseException:
                self.shutdown()
                raise

    def _validate_binding(self, caches, groups):
        names = {p.layer_name for p in self.layout.parts(0)}
        if set(caches) != names or set(groups) != names or any(groups[n] != 0 for n in names):
            raise ValueError('Fused Runtime requires the planned single full-attention group')
        for name, cache in caches.items():
            if isinstance(cache, tuple):
                if len(cache) != 2:
                    raise ValueError('Expected a K/V cache pair')
                views = cache
            else:
                if cache.ndim != 4 or cache.shape[-1] != 2 * self.geometry.head_size:
                    raise ValueError('Invalid combined K/V cache shape')
                views = tuple(cache.narrow(-1, p * self.geometry.head_size, self.geometry.head_size)
                              for p in (0, 1))
            for part, view in enumerate(views):
                if (view.device != self.device or view.dtype != torch.bfloat16
                        or view.ndim != 4 or tuple(view.shape[1:]) != (self.geometry.heads, 1, 128)
                        or view.shape[0] < 1 or view.stride(-1) != 1):
                    raise ValueError('Invalid fused cache geometry or device: ' + name)
                alignment = 4 if part == 1 else 2
                if view.data_ptr() % alignment or any(
                        view.shape[d] > 1 and (view.stride(d) <= 0
                        or view.stride(d) * 2 % alignment) for d in range(3)):
                    raise ValueError('Invalid fused cache stride/alignment: ' + name)
            if views[0].shape[0] != views[1].shape[0]:
                raise ValueError('K/V token capacities differ')
        return min((v[0] if isinstance(v, tuple) else v).shape[0] for v in caches.values())

    def bind_kv_caches(self, kv_caches, group_indices):
        if self._state != 'new':
            raise RuntimeError('Fused Runtime is bind-once; state=' + self._state)
        cache_capacity = self._validate_binding(kv_caches, group_indices)
        self._state = 'binding'
        try:
            with torch.cuda.device(self.device):
                super().bind_kv_caches(kv_caches, group_indices)
                self.cache_capacity = cache_capacity
                # The reserve was present during native KV sizing. Release it
                # BEFORE capture so graph creation can consume that allowance,
                # instead of requiring a second copy of it at peak usage.
                if self._graph_budget is not None:
                    self._graph_budget.close()
                    self._graph_budget = None
                before, _ = torch.cuda.mem_get_info(self.device)
                for slot_id, pipe in enumerate(self.pipelines):
                    slot = self.slots[slot_id]
                    pipe.stream.synchronize()
                    slot.completion_event.record(pipe.stream)
                    slot.completion_event.synchronize()
                    for part in (0, 1):
                        cpu = cpp_backend.build_meta(pipe, slot, part)
                        gpu = self._gpu_metadata[slot_id][part]
                        gpu.copy_(cpu)
                        self._plan_owners.append(cpu)
                        execution = self._module.ExecutionPlan(
                            cpu.data_ptr(), gpu.data_ptr(), slot.host_indices[0].data_ptr(),
                            slot.block_indices[0].data_ptr(), self.geometry.capacity,
                            self.geometry.heads, part, self.device.index, pipe.stream.cuda_stream,
                        )
                        self._plans[slot_id, part] = execution
                        for n in self.graph_lengths:
                            if self.execution != 'decode':
                                execution.build(n, True)
                            if self.execution != 'encode':
                                execution.build(n, False)
                    if any(pipe.codec._native_ans.max_output(n) > slot.arena_a.numel()
                           for n in {self.geometry.packed_bytes(t) for t in range(16, self.geometry.capacity + 1, 16)}):
                        raise ValueError('ANS output capacity exceeds reserved encoded arena')
                after, _ = torch.cuda.mem_get_info(self.device)
                budget = self._graph_budget_bytes
                if before - after > budget:
                    raise RuntimeError(
                        f'CUDA Graph retention {before - after} exceeded '
                        f'pre-reserved startup budget {budget} bytes')
                logger.info('KVServe Graph startup: lengths=%s retained=%d reserved=%d bytes',
                            self.graph_lengths, max(0, before - after), budget)
                self.standard_wire_profile = self.wire_profile
                self.wire_profile = hashlib.sha256(
                    ('kvserve-u8-layer-stride-bucket16-v1:' + self.wire_profile).encode()).hexdigest()
            self._state = 'bound'
        except BaseException:
            self._state = 'failed'
            raise

    def data_bytes(self, kv_part, tokens=None):
        if kv_part not in (0, 1):
            raise ValueError('Invalid K/V part')
        if tokens is None:
            return self.layout.u8_bytes_by_part[kv_part]
        if set(tokens) != {0}:
            raise ValueError('Fused Runtime requires FA group 0')
        return self.geometry.packed_bytes(tokens[0])

    def _frame(self, slot, groups, part, tokens):
        if self._state != 'bound' or type(slot) is not int or not 0 <= slot < len(self.slots):
            raise RuntimeError('Fused Runtime is unbound, closed, or has an invalid slot')
        if part not in (0, 1) or len(groups) != 1 or set(tokens) != {0}:
            raise ValueError('Invalid fused frame group/part')
        ids = groups[0]
        n = tokens[0]
        self.geometry.packed_bytes(n)
        if (len(ids) != n or len(set(ids)) != n
                or any(type(i) is not int or not 0 <= i < self.cache_capacity for i in ids)):
            raise ValueError('Invalid or duplicate fused token indices')
        if not self.slots[slot].completion_event.query():
            raise RuntimeError('Fused slot reused before GPU completion')
        self.slots[slot].host_indices[0].numpy()[:n] = ids
        return n

    def encode(self, slot_id, logical_ids_by_group, kv_part, *, valid_tokens_by_group):
        self._require_execution('encode')
        slot = self.slots[slot_id]
        return self.encode_into(slot_id, logical_ids_by_group, kv_part,
                                valid_tokens_by_group=valid_tokens_by_group,
                                output=slot.arena_a, auxiliary=slot.aux)

    def encode_into(self, slot_id, logical_ids_by_group, kv_part, *,
                    valid_tokens_by_group, output, auxiliary):
        self._require_execution('encode')
        if type(slot_id) is not int or not 0 <= slot_id < len(self.slots):
            raise ValueError('Invalid fused slot')
        self._validate_output_bank(slot_id, output, auxiliary)
        lock = self._locks[slot_id]
        if not lock.acquire(blocking=False):
            raise RuntimeError('Concurrent submission to one fused slot')
        try:
            n = self._frame(slot_id, logical_ids_by_group, kv_part, valid_tokens_by_group)
            slot, pipe = self.slots[slot_id], self.pipelines[slot_id]
            aux_bytes = self.aux_bytes(kv_part, {0: n})
            for tensor, size in ((output, slot.arena_a.numel()), (auxiliary, aux_bytes)):
                if (tensor.device != self.device or tensor.dtype != torch.uint8
                        or not tensor.is_contiguous() or tensor.numel() < size):
                    raise ValueError('Invalid encoded output/auxiliary bank')
            self._generations[slot_id] += 1
            size = self.geometry.packed_bytes(n)
            aux_dst = 0 if auxiliary.data_ptr() == slot.aux.data_ptr() else auxiliary.data_ptr()
            with torch.cuda.device(self.device):
                self._plans[slot_id, kv_part].encode(
                    pipe.codec._native_ans, slot.arena_b.data_ptr() + self.layout.u8_base,
                    output.data_ptr(), size, slot.completion_event.cuda_event, n,
                    slot.aux.data_ptr(), aux_dst, aux_bytes if aux_dst else 0,
                )
            return EncodedChunk(_Size(pipe.codec, slot.completion_event, (output, auxiliary),
                                     (self, slot_id, self._generations[slot_id])),
                                slot.completion_event, size)
        except BaseException:
            self._state = 'failed'
            raise
        finally:
            lock.release()

    def decode(self, slot_id, logical_ids_by_group, kv_part, codec_bytes, u8_bytes, *,
               valid_tokens_by_group, wire_profile=None):
        self._require_execution('decode')
        if type(slot_id) is not int or not 0 <= slot_id < len(self.slots):
            raise ValueError('Invalid fused slot')
        lock = self._locks[slot_id]
        if not lock.acquire(blocking=False):
            raise RuntimeError('Concurrent submission to one fused slot')
        try:
            n = self._frame(slot_id, logical_ids_by_group, kv_part, valid_tokens_by_group)
            slot, pipe = self.slots[slot_id], self.pipelines[slot_id]
            if wire_profile is None:
                wire_profile = self.wire_profile
            if wire_profile not in (self.wire_profile, self.standard_wire_profile):
                raise ValueError('Invalid received compression profile')
            standard_layout = wire_profile == self.standard_wire_profile
            expected_tokens = self.geometry.capacity if standard_layout else n
            if u8_bytes != self.geometry.packed_bytes(expected_tokens) or not 0 < codec_bytes <= slot.arena_a.numel():
                raise ValueError('Invalid received fused payload size')
            self._generations[slot_id] += 1
            with torch.cuda.device(self.device):
                self._plans[slot_id, kv_part].decode(
                    pipe.codec._native_ans, slot.arena_a.data_ptr(),
                    slot.arena_b.data_ptr() + self.layout.u8_base, u8_bytes,
                    slot.completion_event.cuda_event, n, standard_layout,
                )
            return slot.completion_event
        except BaseException:
            self._state = 'failed'
            raise
        finally:
            lock.release()

    def shutdown(self):
        if self._state == 'closed':
            return
        with ExitStack() as stack:
            for lock in self._locks:
                stack.enter_context(lock)
            with torch.cuda.device(self.device):
                for stream in self.streams:
                    stream.synchronize()
                for execution in self._plans.values():
                    execution.close()
                self._plans.clear()
                for codec in self._codecs:
                    codec.close()
                self._codecs.clear()
                if self._graph_budget is not None:
                    self._graph_budget.close()
                    self._graph_budget = None
                super().shutdown()
                self.output_slots.clear()
            self._state = 'closed'


def _fused_dependency_error():
    """Check the optional native ANS ABI without compiling or allocating CUDA."""
    try:
        library = Path(importlib.import_module('nvidia.libnvcomp').__file__).resolve().parent
    except (ImportError, AttributeError, TypeError):
        return 'the nvCOMP 6 fused runtime package is not installed'
    if not (library / 'lib64' / 'libnvcomp.so.6').is_file():
        return 'the nvCOMP 6 shared library is not installed'
    if not (library / 'include' / 'nvcomp' / 'ans.hpp').is_file():
        return 'the nvCOMP 6 development headers are not installed'
    return cpp_backend.jit_toolchain_error(already_loaded=_module is not None)


def create_runtime(plan, cache_layout, blocks_per_logical, device, *, backend='auto',
                   output_slots=0, graph_lengths=None, execution='both',
                   standard_backend=None):
    if backend not in ('auto', 'standard', 'fused'):
        raise ValueError('Unknown KVServe execution backend')
    if execution not in ('both', 'encode', 'decode'):
        raise ValueError('Unknown KVServe execution capability')
    if standard_backend not in (None, 'python', 'cpp'):
        raise ValueError('Unknown KVServe standard pipeline backend')
    geometry, reason = select_geometry(plan, cache_layout, blocks_per_logical)
    if backend == 'fused' and geometry is None:
        raise ValueError('Unsupported fused geometry: ' + reason)
    if backend != 'standard' and geometry is not None:
        dependency_error = _fused_dependency_error()
        if torch.version.hip is not None:
            dependency_error = 'fused execution requires NVIDIA CUDA'
        if dependency_error is not None:
            if backend == 'fused':
                raise RuntimeError('Fused execution unavailable: ' + dependency_error)
            reason = dependency_error
        else:
            shared = None
            if execution != 'decode':
                with torch.cuda.device(device):
                    shared = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
            if execution == 'decode' or shared >= 225 * 1024:
                return FusedCompressionRuntime(plan, cache_layout, blocks_per_logical, device,
                                               output_slots=output_slots, graph_lengths=graph_lengths,
                                               execution=execution)
            if backend == 'fused':
                raise ValueError('Fused head kernel requires 225 KiB opt-in shared memory')
            reason = 'device does not provide 225 KiB opt-in shared memory'
    logger.info('Using standard KVServe compression execution: %s', reason)
    kwargs = {'execution': execution}
    if standard_backend is not None:
        kwargs['pipeline_backend'] = standard_backend
    runtime = CompressionRuntime(plan, cache_layout, blocks_per_logical, device,
                                 **kwargs)
    runtime.allocate_output_slots(output_slots)
    return runtime
