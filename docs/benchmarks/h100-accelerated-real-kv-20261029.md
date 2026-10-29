# Accelerated compression stages on real Qwen KV

Measured on 2026-10-29 using an NVIDIA H100 PCIe (80 GB), driver 610.43.02,
PyTorch 2.13.0+cu130 and nvCOMP 5.3.0.16. The GPU was idle before the run;
it was not exclusively reserved. Samples were collected with one-second GPU
monitoring and are included in the [raw results](h100-accelerated-real-kv-20261029.json).

The input was captured from the first LongBench HotpotQA context using local
Qwen2.5-7B-Instruct, BF16 and 8,192 input tokens. Its K and V tensors each
have shape `[28 layers, 8192 tokens, 4 KV heads, 128 head_dim]`. Both cases
below use the same captured tensors. Key uses channel quantization with 12/6
levels; value uses token quantization with 8/4 levels. The last nine layers
use the low-level profile, matching the runtime's layer-split rule.

## Results

Each case has three independent runs. Each run uses five warmups, seven timed
samples, and five operations per sample. Tables show the median of the three
run medians. Rates are decimal GB/s calculated from the indicated byte count;
they are effective input/output rates, not measured GPU memory bandwidth.

For **1,024 tokens**, each K or V contains 29,360,128 BF16 bytes across 28
layers. Quantization produces 14,680,064 uint8 bytes. This corresponds to one
64-block chunk when the block size is 16.

| Stage | K time (ms) | K GB/s | V time (ms) | V GB/s | Throughput basis |
| --- | ---: | ---: | ---: | ---: | --- |
| FWHT + normalization | 0.7557 | 38.85 | 0.7561 | 38.83 | BF16 input |
| Quantize | 0.2694 | 109.00 | 0.1270 | 231.14 | BF16 input |
| Dequantize | 6.3532 | 4.62 | 6.7218 | 4.37 | BF16 output |
| nvCOMP ANS encode | 0.1289 | 113.87 | 0.1225 | 119.88 | uint8 symbols input |
| nvCOMP ANS decode | 0.0715 | 205.22 | 0.0701 | 209.46 | uint8 symbols output |

The compressed payloads were 5,493,964 bytes for K and 4,356,042 bytes for V:
2.672x and 3.370x compression relative to the uint8 symbols. These ratios do
not include quantization metadata, raw tails or transport headers.

For **8,192 tokens**, each K or V contains 234,881,024 BF16 bytes and produces
117,440,512 uint8 bytes:

| Stage | K time (ms) | K GB/s | V time (ms) | V GB/s | Throughput basis |
| --- | ---: | ---: | ---: | ---: | --- |
| FWHT + normalization | 3.5354 | 66.44 | 3.5351 | 66.44 | BF16 input |
| Quantize | 1.7704 | 132.67 | 0.3503 | 670.60 | BF16 input |
| Dequantize | 50.3843 | 4.66 | 6.7320 | 34.89 | BF16 output |
| nvCOMP ANS encode | 0.4401 | 266.88 | 0.4255 | 276.02 | uint8 symbols input |
| nvCOMP ANS decode | 0.2214 | 530.33 | 0.2194 | 535.17 | uint8 symbols output |

The 8,192-token case treats all tokens as one reduction/codec input. It is a
large-payload comparison, not eight independent 1,024-token runtime chunks:
the quantization ranges, compressed representation and launch behavior differ.

## Humming Hadamard comparison

On the same H100 and the same real BF16 K/V fixture, compare the production
CUDA kernels with Humming 0.1.16. Each result is the median of three independent
runs, with seven CUDA-graph event samples per run and five operations per sample.
Each row covers all 28 layers; GB/s is BF16 input bytes divided by elapsed time.
[Raw timings, wall times, input hashes and monitoring](h100-hadamard-humming-20261029.json)
are available separately.

| Tokens | Tensor | Current CUDA ms / GB/s | Humming per layer ms / GB/s | Humming batched ms / GB/s |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | K | 0.7570 / 38.79 | 0.1267 / 231.77 | 0.0779 / 376.67 |
| 1,024 | V | 0.7593 / 38.67 | 0.1268 / 231.51 | 0.0777 / 377.85 |
| 8,192 | K | 3.5325 / 66.49 | 0.6487 / 362.07 | 0.5971 / 393.34 |
| 8,192 | V | 3.5325 / 66.49 | 0.6488 / 362.02 | 0.5972 / 393.33 |

These measurements start from signed, contiguous data and include only the
normalized transform. They exclude gathering, sign multiplication, allocation,
scatter and transport. Synchronized wall time without graph replay for the
1,024-token K case was 0.996 ms for current CUDA, 0.805 ms for per-layer
Humming, and 0.088 ms for batched Humming. These are stage microbenchmarks,
not end-to-end serving speeds. The batched experiment uses one contiguous
28-layer input and does not include the cost of creating such a batch.

KVServe's Python transform calls Humming on the main thread, while the current
Mooncake worker-thread path uses the PyTorch fallback. The C++/CUDA runtime
uses its own kernels; a batched Humming runtime is not implemented. Results
are not bit-identical: for 1,024-token K, mean and maximum absolute differences
between Humming and current CUDA were 0.01295 and 3.0, respectively; for V they
were 0.00329 and 0.25. Humming's per-layer and batched outputs matched exactly
for these inputs. Further end-to-end quality and compatibility validation is
needed before switching the runtime implementation.

Humming is optional for this benchmark; install `humming-kernels` separately
(the measurements used 0.1.16). With the same CUDA build dependencies and a
captured fixture, run:

```bash
python scripts/bench_hadamard_compare.py --input kv.pt --tokens 1024 \
  --output humming-1024.json
```

## Measurement boundaries and validation

- The benchmark wrapper includes the repository's `kvs_compress.cu` and invokes
  its actual FWHT, normalization, K/channel and V/token quant/dequant kernels,
  once per layer. It starts from already gathered and signed data and excludes
  gather/sign, inverse sign, scatter, allocation and network work.
- Kernel event intervals use CUDA graph replay, divided by the operations per
  replay. nvCOMP uses batched stream events and reusable caller-owned output
  buffers/configs; it does not use graph capture. Its event interval may include
  host submission gaps. Separate synchronized wall times and all samples are in
  the JSON. These are stage microbenchmarks, not serving throughput or TTFT.
- FWHT, quantization, metadata and dequantization were checked against BF16
  PyTorch references on the first, middle and last layers. nvCOMP encode/decode
  exactly restored the complete quantized K/V symbol streams.
- A separate check used the complete 28-layer `CompressionRuntime` with the
  CUDA backend and 1,024-token chunk. Its K and V quantized symbols matched the
  standalone benchmark byte-for-byte, and its nvCOMP payload lengths matched.
  The [runtime check](h100-runtime-check-20261029.json) records zero mismatches.
- The production CUDA source SHA-256 is
  `069ba311c769bad086bd71cf4f27518ec95589a7e7ee54be3cfda49c08278e71`.
  Script, wrapper, input and codec-input hashes are recorded in the JSON.

## Reproduce and compare platforms

Use a CUDA/ROCm PyTorch environment, a local Transformers model, and a text
file with at least 8,192 tokens. The native NVIDIA benchmark also needs a CUDA
compiler, ninja and the nvCOMP Python bindings. Model prefill and fixture
creation happen outside all measurements.

```bash
python scripts/capture_kv_fixture.py --model /path/to/Qwen2.5-7B-Instruct \
  --text context.txt --tokens 8192 --output kv.pt
python scripts/bench_accelerated_stages.py --input kv.pt --tokens 1024 \
  --save-symbols symbols-1024.pt --output nvidia-1024.json
python scripts/verify_accelerated_runtime.py --input kv.pt --tokens 1024 \
  --symbols-input symbols-1024.pt --output runtime-check.json
python scripts/bench_accelerated_stages.py --input kv.pt --tokens 8192 \
  --save-symbols symbols-8192.pt --output nvidia-8192.json
```

The fixture contains input-derived model activations. Copy it and the matching
symbol file to the other system. Pass the same `--tokens`, timing settings and
`--symbols-input` there; compare the recorded input and codec-input SHA-256
values. Comparing different codecs should identify both codec names and
compression ratios. If the AMD backend cannot use GPU graph capture, use
`--no-graph` on both platforms for a like-for-like timing boundary.

An external `--backend adapter.py` must define `CODEC_NAME`,
`load_kernels(device)` and `create_codec(q, stream)`. The kernel object exposes
the pointer-based `fwht`, `quant` and `dequant` signatures in
`scripts/bench_runtime_bind.cpp`; its launches must run on the supplied stream.
`create_codec` returns an object with `payload_bytes`, a same-shape uint8
`decoded` tensor, and asynchronous `encode()`/`decode()` methods. It must retain
its input and preallocated output buffers through synchronization. The benchmark
checks the output against the same BF16 reference and requires exact lossless
round-trip. An AMD implementation has not yet been tested here.
