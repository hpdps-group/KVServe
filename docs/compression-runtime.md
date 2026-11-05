# Fixed-profile compression runtime

KVServe provides a transport-independent runtime for caller-owned KV caches.
It imports no vLLM module. Cache registration, request lifecycle and slot credit
remain the responsibility of the transfer/storage adapter.

## Installation

Install the package using the extra matching the CUDA major and desired runtime.
Consumers do not need to clone KVServe or change `PYTHONPATH`. For example, after
publishing this change, replace `<commit>` below with its full commit SHA:

```bash
# Standard runtime, CUDA 13; use runtime-cu12 for CUDA 12.
uv pip install 'kvserve_v1[runtime,runtime-cu13] @ https://github.com/hpdps-group/KVServe/archive/<commit>.tar.gz'

# Optional fused runtime, CUDA 13; use runtime-fused-cu12 for CUDA 12.
uv pip install 'kvserve_v1[runtime-fused-cu13] @ https://github.com/hpdps-group/KVServe/archive/<commit>.tar.gz'
```

For KVServe development from a local checkout, use
`uv pip install '.[runtime,runtime-cu13]'` for the standard runtime, or
`uv pip install '.[runtime-fused-cu13]'` for the fused runtime. The standard
CUDA extra pins nvCOMP 5.3, while the fused extra pins nvCOMP 6.0; do not install
both CUDA extras in the same environment. Older pinned archives retain their
original dependencies.

The runtime extras include Ninja. On NVIDIA CUDA, both standard execution paths
use KVServe's BF16 Hadamard CUDA kernel. A matching CUDA compiler is required;
JIT sources are included in the wheel and compilation occurs before requests.

## Backend selection

`create_runtime()` defaults to `backend="auto"`. For a supported geometry with
nvCOMP 6 and the JIT tools available, it selects the fused runtime; otherwise
it selects the standard compressed runtime. Explicit `backend="fused"` rejects
an unsupported geometry or missing dependency. A runtime or build failure after
selection is reported, not silently converted into an uncompressed transfer.

The fused path currently accepts uniform BF16 Full Attention, HND token views
(`block_size=1` and `blocks_per_logical=1`), head size 128, 1/2/4/8/16 KV
heads, the default Hadamard/quantizer/ANS stage order, channel K and token V
quantization, and a chunk capacity divisible by 16 and at most 1,024 tokens.
The encode direction needs 225 KiB of opt-in shared memory per block; a
decode-only runtime does not impose that encode requirement. Full chunks use
prebuilt CUDA Graphs; other valid lengths use eager fused kernels with the same
codec. P and D may use different hardware, but must agree on the wire profile.

The standard runtime chooses the C++ path by default on a supported NVIDIA
CUDA layout with the JIT tools available: BF16, one Full Attention group, and
channel/token quantization. Unsupported layouts or missing tools use the
standard Python path. A Python path with a Hadamard stage still requires the
CUDA compiler for its own native transform. Set
`KVSERVE_COMPRESSION_BACKEND=python` or `cpp` to override this choice; an
explicit `cpp` rejects unsupported layouts. The setting is captured per runtime
instance and the layout choice is resolved when caches bind. Raw aggregation
retains its separate gather/scatter path. `backend="standard"` forces the
standard runtime and still uses this Python/C++ selector.

Callers can select this per instance without changing the process environment:
pass `standard_backend="cpp"` or `"python"` to `create_runtime`, or
`pipeline_backend` to `CompressionRuntime`. An explicit `"cpp"` rejects
unsupported layouts and propagates JIT failures. The default `None` preserves
the existing automatic/environment-selected behavior. To warm the C++ CUDA
module before inference-cache sizing and include retained driver memory in the
plan, pass `warm_cpp_backend=True` to `build_memory_plan` on NVIDIA CUDA.

`CompressionRuntime(...)` remains the backward-compatible standard-runtime
constructor. Existing callers such as vLLM automatically benefit from the
standard C++ selection when their layout is supported. Callers must explicitly
adopt `create_runtime(...)` and satisfy its token-view contract to select the
new fused path. Standard Python/C++ use the same wire profile; the fused
representation has a different profile. A caller must check peer profiles and
choose a common backend rather than assume `auto` selected the same one on
different machines.

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
    API_VERSION, CacheSpec, CompressionConfig, create_runtime, build_memory_plan,
)

config = CompressionConfig.from_extra_config({
    "compression": {"enable-compression-pipeline": True,
                    "logical_blocks_per_chunk": 64},
})
# Supply an explicit layer index; names need not follow a model naming convention.
# For hybrid models use original global full-attention indices, not a dense
# renumbering, and set num_layers to the total number of model layers.
specs = {name: CacheSpec(layer_index, block_size, kv_heads,
                        key_head_size, dtype, value_head_size)}
plan = build_memory_plan(config, specs, blocks_per_logical=physical_factor,
                         tp_size=tp_size, tp_rank=tp_rank,
                         num_layers=num_layers, device=device)
# Create the runtime before sizing/allocating the inference KV cache. Its owned
# buffers, optional output slots and Graph budget are reflected in runtime.plan.
runtime = create_runtime(plan, "HND", physical_factor, device)
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

The standard runtime supports dense BF16 HND/LBHNC caches with separate K/V
views, including different K/V head sizes. Complete blocks enter the selected
stages; standard-runtime partial tails are transferred raw. The fused runtime
accepts the token-level geometry described above and compresses short frames.
Configure `pipeline=["codec"]` for lossless ANS-only. Head profiles include TP
slicing and replicated KV-head mappings; their compatibility with the fused
path depends on the section and geometry checks above.

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
