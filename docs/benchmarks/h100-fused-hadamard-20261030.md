# Default fused BF16 Hadamard on H100

Validated on 2026-10-30 on an NVIDIA H100 PCIe 80 GB, physical GPU 1,
PyTorch 2.13.0+cu130 and nvCOMP 5.3.0.16. The GPU was idle before testing;
it was not exclusively reserved. The performance runs include one-second
GPU monitoring in the [raw validation results](h100-fused-hadamard-20261030.json).

The default Python transform and the C++ encode/decode plans now call the
same fused CUDA kernel. For power-of-two head sizes up to 1024 it holds each
row in registers, uses warp shuffles for butterflies, and normalizes before
storing. It preserves the former C++ implementation's rounding to BF16 after
each butterfly. The request path uses preallocated output buffers and no extra
device scratch. Larger power-of-two head sizes retain the two-buffer fallback.
Humming is not required on either the main thread or a Mooncake worker thread.
The overall execution-backend setting continues to select Python or C++.

## Per-layer transform throughput

The input is the same real Qwen2.5-7B-Instruct KV fixture used in the
[previous measurements](h100-accelerated-real-kv-20261029.md), captured from
the first LongBench HotpotQA context. Each K/V has shape
`[28 layers, tokens, 4 KV heads, 128 head_dim]`, in BF16. Its input hashes
match the previous Humming comparison for both token counts.

Each case has three independent process runs, five warmups, seven timed
samples per run and five operations per sample. Tables show the median of
the three run medians. Decimal GB/s counts BF16 input bytes only. These are
28 separate layer calls, with no cross-layer batching.

| Tokens | Execution path | K GPU ms | K GB/s | V GPU ms | V GB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | Python default | 0.0868 | 338.26 | 0.0866 | 338.89 |
| 1,024 | C++ layer loop | 0.0868 | 338.19 | 0.0864 | 339.84 |
| 8,192 | Python default | 0.3347 | 701.70 | 0.3342 | 702.92 |
| 8,192 | C++ layer loop | 0.3356 | 699.97 | 0.3337 | 703.92 |

All six runs passed `--minimum-gb-s 200` for both K/V and both execution
paths. The earlier 1,024-token CUDA implementation measured 38.85/38.83 GB/s
for K/V; the earlier per-layer Humming implementation measured
231.77/231.51 GB/s. Those are historical runs on the same input and GPU, not
a simultaneous comparison. Humming uses different intermediate numerics.

GPU event intervals use CUDA graph replay. Separate synchronized wall timing
uses ordinary calls without graph replay and includes host submission costs:

| Tokens | Execution path | K wall ms | K GB/s | V wall ms | V GB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | Python default | 0.3881 | 75.65 | 0.3854 | 76.18 |
| 1,024 | C++ layer loop | 0.1291 | 227.47 | 0.1281 | 229.14 |
| 8,192 | Python default | 0.4021 | 584.06 | 0.3962 | 592.83 |
| 8,192 | C++ layer loop | 0.3713 | 632.53 | 0.3695 | 635.66 |

Python's per-layer host submission costs remain visible for small inputs.
The complete C++ runtime crosses the Python boundary once per chunk. Both
tables exclude gather, sign multiplication, allocation, quantization, codec,
scatter and network work; they do not measure serving throughput. The
8,192-token case is one large input rather than eight independent chunks.

## Correctness and runtime validation

- The 51 Hadamard GPU tests passed: power-of-two sizes 1 through 2048,
  incomplete block tails, a BF16 butterfly reference, an independent dense
  Sylvester matrix, disjoint buffers, nondefault streams, concurrent workers
  and worker-thread CUDA graph capture with Humming imports blocked.
- The Python and C++ execution backends each passed 68 runtime/config tests,
  including encode/decode, raw tails, head selection, cross-backend wire
  decoding, quantization references and workspace lifecycle checks.
- Both backends encoded the full real 28-layer, 1,024-token K/V fixture from
  a worker thread. All 14,680,064 uint8 symbols per K/V matched the saved
  pre-change CUDA symbols, with zero mismatches. nvCOMP payload sizes also
  remained 5,493,964 bytes for K and 4,356,042 bytes for V.
- Compute Sanitizer memcheck passed 13 selected shape/worker/graph tests and
  reported `ERROR SUMMARY: 0 errors`.
- An existing quantization test assumed identical CPU/CUDA signs for a tied
  zero minimum. The failure was reproduced on unmodified `v1`. Its assertion
  now accepts either sign only for the all-zero minimum; scale and all other
  metadata remain byte-exact. Quantization kernel behavior was not changed.

The production CUDA source SHA-256 is
`26538fc0688dfa9498ca335c99c316b02c1c63dc44415e77e2c4b9d9ef396500`.
The raw JSON records source hashes, commands, test output, every timing sample,
input hashes and GPU monitoring. The standalone 51-case run preceded the
recorded validation commands; memcheck repeats 13 of those cases.

## Reproduce

Install the runtime extras and a CUDA compiler, then use the same captured
fixture and pre-change symbol stream from the earlier benchmark:

```bash
python -m pytest tests/runtime/test_hadamard.py -q
KVSERVE_COMPRESSION_BACKEND=python python -m pytest tests/runtime/test_gpu.py -q
KVSERVE_COMPRESSION_BACKEND=cpp python -m pytest tests/runtime/test_gpu.py -q
python scripts/verify_accelerated_runtime.py --input kv.pt \
  --symbols-input symbols-1024.pt --tokens 1024 --backend python --worker
python scripts/verify_accelerated_runtime.py --input kv.pt \
  --symbols-input symbols-1024.pt --tokens 1024 --backend cpp --worker
python scripts/bench_default_hadamard.py --input kv.pt --tokens 1024 \
  --minimum-gb-s 200 --output fused-1024.json
python scripts/bench_default_hadamard.py --input kv.pt --tokens 8192 \
  --minimum-gb-s 200 --output fused-8192.json
```
