# Fixed-profile compression runtime

KVServe provides a transport-independent runtime for caller-owned KV caches.
It imports no vLLM module. Cache registration, request lifecycle and slot credit
remain the responsibility of the transfer/storage adapter.

## Installation

Install the package with `runtime` and the nvCOMP extra matching the CUDA major.
Consumers do not need to clone KVServe or add its source tree to `PYTHONPATH`;
the current `v1` source archive is built and installed as a normal wheel:

```bash
uv pip install 'kvserve_v1[runtime,runtime-cu13] @ https://github.com/hpdps-group/KVServe/archive/refs/heads/v1.tar.gz'
# CUDA 12: use runtime-cu12 instead.
export KVSERVE_COMPRESSION_BACKEND=cpp
```

For reproducible deployments, replace `refs/heads/v1` with a full commit SHA
containing this dependency change. Older pinned archives retain their original
dependencies, including Humming.

For KVServe development from a local checkout, use
`uv pip install '.[runtime,runtime-cu13]'` instead.

The runtime extras include ninja and do not install `humming-kernels`.
Both execution backends use the same native fused BF16 Hadamard implementation
on NVIDIA CUDA. A CUDA compiler is required; the extension builds during cache
binding, before requests arrive. Set `KVSERVE_COMPRESSION_BACKEND=cpp` in both
producer and consumer environments to drive the complete layer loop from C++.
The runtime ships its JIT source files in the wheel. Importing `kvserve_v1` no
longer loads the optional legacy vLLM connector; its existing exported name is
resolved when requested.

The execution backend defaults to `python`. Its `transformer` stage uses the
same fused CUDA kernel on main and Mooncake worker threads, with no Humming
dependency or main-thread JIT restriction. It rounds each butterfly to BF16,
preserving the previous C++ and worker-thread numerics, quantized symbols and
wire layout. Unlike the former Python main-thread Humming call, it also uses
this BF16 operation order. Head sizes 1 through 1024 (powers of two) need one
kernel per layer; larger power-of-two sizes retain the two-buffer fallback.
The inverse uses the same normalized transform. No additional device scratch
is allocated in the request path.

Use `scripts/bench_default_hadamard.py` with the same captured BF16 KV to
measure the per-layer transform from the Python default and the C++ layer loop:

```bash
python scripts/bench_default_hadamard.py --input kv.pt --tokens 1024 \
  --output hadamard.json --minimum-gb-s 200
```

The JSON records CUDA graph
event timing and separate uncaptured synchronized wall timing. Sign generation,
gathering, scatter and transport are outside this measurement. Add
`--minimum-gb-s 200` to enforce a measured minimum for both K and V paths.
The [H100 validation](benchmarks/h100-fused-hadamard-20261030.md) records
338 GB/s GPU-event throughput for the 1,024-token Python default, along with
separate host-inclusive wall times and full runtime correctness checks.

The CUDA backend uses cooperative channel/token reductions and quantization,
preserving the fixed-profile BF16 operation order and buffer layout. Each
layer section still uses one quantization kernel and needs no extra global
workspace.

## API version 1

```python
from kvserve_v1.compression.runtime import (
    API_VERSION, CacheSpec, CompressionConfig, CompressionRuntime, build_memory_plan,
)

config = CompressionConfig.from_extra_config({
    "compression": {"enable-compression-pipeline": True,
                    "logical_blocks_per_chunk": 64},
})
# Supply an explicit layer index; names need not follow a model naming convention.
specs = {name: CacheSpec(layer_index, block_size, kv_heads,
                        key_head_size, dtype, value_head_size)}
plan = build_memory_plan(config, specs, blocks_per_logical=physical_factor,
                         tp_size=tp_size, tp_rank=tp_rank,
                         num_layers=num_layers, device=device)
# Reserve plan.reserved_bytes before allocating the inference KV cache.
runtime = CompressionRuntime(plan, "HND", physical_factor, device)
runtime.bind_kv_caches(cache_tensors, layer_to_group)
addresses, lengths = runtime.registered_memory()

encoded = runtime.encode(slot_id, block_ids_by_group, kv_part,
                         valid_tokens_by_group=valid_tokens)
encoded.event.synchronize()  # before the transfer engine reads the payload
# Transfer payload, runtime.aux_bytes(...) and encoded.raw_tail_bytes.
# The receiver must use distinct local buffers and the same wire_profile.
event = receiver.decode(receiver_slot, destination_blocks, kv_part,
                        encoded.codec_bytes, encoded.u8_bytes,
                        valid_tokens_by_group=valid_tokens)
event.synchronize()  # before making destination cache visible or returning credit
```

This is a fixed-profile API. Layout and head selection are planned once;
head orders/signs are initialized once when caches bind. Buffers are bounded,
preallocated and owned by the runtime. Encode/decode never perform a network
operation. Callers can register the single pool and address its slot views.

Supported caches are dense BF16 HND/LBHNC with separate K/V views, including
different K/V head sizes. Complete blocks enter transform/quantizer/ANS; partial
tails are transferred raw. Configure `pipeline=["codec"]` for lossless ANS-only.
Head profiles include TP slicing and replicated KV-head mappings. CUDA supports
a single FullAttention group; unsupported combinations fail explicitly.

## Ownership and compatibility

The adapter must fence the source cache before encode, wait for encode before
network reads, wait for received bytes before decode, and retain each receiver
slot until decode completes. Shutdown is safe only after outstanding transfers
and engine jobs have drained. `shutdown()` fences runtime streams and releases
its codec resources; it cannot stop remote writes.

Profile version 2 fingerprints the static representation and rejects older
profile versions; both endpoints must be upgraded together. Slot count and the
local score-file path do not affect peer representation identity.

nvCOMP uses a process-global allocator. While active, KVServe routes its streams
to fixed workspaces and ordinary external codec streams to independent
allocations, restoring the SDK default after the last runtime stops. Arbitrary
external custom nvCOMP allocators are unsupported.

Numerical/kernel tests live in `tests/runtime`; vLLM separately tests cache
adaptation, Mooncake integration, disabled behavior and lifecycle safety.
