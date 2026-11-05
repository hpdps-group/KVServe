# H200 fixed-profile backend comparison

These are isolated Runtime microbenchmarks, not end-to-end serving results.
They were recorded on 2026-11-05 using an NVIDIA H200 NVL, Torch 2.14.1+cu130,
nvCOMP 6.0, BF16, the default Hadamard/quantizer/ANS profile, 28 layers, four KV
heads, and head size 128. The input was seeded synthetic normal KV, not model
activations. Other users' GPU processes were resident, with approximately 0–2%
SM activity in a short pre-run sample; this was not an exclusive GPU run.

Each backend was warmed up and then measured in alternating order for two
rounds, with three timed repeats per round (four for the 16-token case). The
table is the mean of the two per-round medians. K and V are encoded or decoded
sequentially with completion waits; allocation, initialization, compilation,
model forward, network, and serving scheduling are excluded. GB/s is raw BF16
K+V bytes divided by wall time, not network throughput.

| Layout and tokens | Backend | Encode GB/s | Decode GB/s |
| --- | --- | ---: | ---: |
| 16-token blocks, 1,024 tokens | standard Python | 4.77 | 9.60 |
| 16-token blocks, 1,024 tokens | standard C++ | 42.87 | 44.61 |
| 16-token blocks, 16 tokens | standard Python | 0.079 | 0.150 |
| 16-token blocks, 16 tokens | standard C++ | 0.866 | 0.900 |
| token view, 1,024 tokens | standard Python | 3.83 | 7.41 |
| token view, 1,024 tokens | standard C++ | 37.81 | 39.24 |
| token view, 1,024 tokens | fused C++ | 84.11 | 88.92 |

For the token-view case, fused was approximately 2.22× the standard C++ encode
throughput and 2.27× its decode throughput in this configuration. For the
16-token-block/1,024-token case, standard C++ was approximately 8.99× the
standard Python encode throughput and 4.65× its decode throughput. All runs
checked equal quantized symbols, auxiliary values and restored KV for this
default lossy profile. Equality does not mean recovery of the original BF16 KV.

The measurements used the fused-runtime patch over KVServe `60f749a`, before
the automatic-dispatch changes in this branch. The timed kernels and explicit
backend paths were unchanged by dispatch selection. They do not establish an
end-to-end speedup, a bound for other geometries, or a claim of exclusive GPU
performance. Earlier SGLang testing on H200 did not show a stable cold-prefill
throughput improvement under natural bandwidth.

Raw results: [page16/1024](page16-1024.json), [page16/16](page16-16.json),
[token/1024](token-1024.json). Reproduce the token-view comparison with:

```bash
python scripts/bench_runtime_dispatch.py --block-size 1 --blocks 1024 \
  --backends python cpp fused --rounds 2 --repeats 3 --warmups 1 \
  --output /tmp/kvserve-h200-runtime.json
```

For the page-16 comparison use `--block-size 16 --blocks 64 --backends python
cpp`; use `--blocks 1 --repeats 4 --warmups 2` for the 16-token case. A matching
CUDA toolkit and the fused nvCOMP 6 extra are needed for all three backends.
