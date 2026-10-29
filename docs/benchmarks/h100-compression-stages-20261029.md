# H100 compression-stage reference measurements

Measured on 2026-10-29 on an NVIDIA H100 PCIe, GPU 1, with driver 610.43.02.
Before and after the runs this GPU had 0% reported utilization and 124 MiB of
other GPU memory in use. The measurement was not an exclusive GPU reservation.
PyTorch was 2.10.0+cu128, fast-hadamard-transform 1.1.0 and nvCOMP 5.2.0.13.
The tested KVServe checkout was based on `4316a5e`; the benchmark script SHA-256
was `8632e25efbba82aa7bb00b4aeeb1e4d1133c171869a9e9d78fc32f59adbcc893`.

The input was synthetic BF16 with shape `[64, 16, 4, 128]` (1,024 tokens,
4 heads, head dimension 128), seed 42 and scale 0.35. Channel quantization
used 12 levels; token quantization used 8. Each case ran three times, alternating
case order, with 20 warmups and 100 measured samples per timing type and stage.
The table reports the median of the three GPU-event medians. GB/s is decimal
input or output bytes divided by the event time; it is not hardware memory
bandwidth. Event intervals can include GPU idle gaps from host launches.

| Axis / stage | Event time (ms) | Input GB/s | Output GB/s | Wall time (ms) |
| --- | ---: | ---: | ---: | ---: |
| Channel: transform | 0.0451 | 23.26 | 23.26 | 0.0384 |
| Channel: inverse transform | 0.0443 | 23.65 | 23.65 | 0.0386 |
| Channel: quantize | 0.0803 | 13.06 | 6.53 | 0.0766 |
| Channel: dequantize | 0.0292 | 17.96 | 35.91 | 0.0260 |
| Channel: nvCOMP ANS encode | 0.2092 | 2.51 | 1.03 | 0.1980 |
| Channel: nvCOMP ANS decode | 0.1440 | 1.50 | 3.64 | 0.1322 |
| Token: transform | 0.0445 | 23.58 | 23.58 | 0.0389 |
| Token: inverse transform | 0.0436 | 24.04 | 24.04 | 0.0389 |
| Token: quantize | 0.0721 | 14.53 | 7.27 | 0.0684 |
| Token: dequantize | 0.0289 | 18.12 | 36.25 | 0.0258 |
| Token: nvCOMP ANS encode | 0.2084 | 2.52 | 0.87 | 0.1973 |
| Token: nvCOMP ANS decode | 0.1415 | 1.28 | 3.70 | 0.1306 |

These are interface-level results for the legacy `HadamardTransform`, PyTorch
min/max quantizer and nvCOMP wrapper, not the fused Mooncake CUDA runtime.
Allocation, Python dispatch and wrapper work are part of the interface being
measured. Quantizer metadata is excluded from byte counts. The codec uses a
separately generated, shared uint8 fixture, rather than the preceding measured
quantizer's output. Its 524,288 input bytes produced 215,444 bytes for channel
(2.434x) and 181,016 bytes for token (2.896x). Fixture hashes, individual run
medians, ranges and GPU monitoring samples are in
[`h100-compression-stages-20261029.json`](h100-compression-stages-20261029.json).
The standalone stages are not an end-to-end KV transfer measurement.

To regenerate the input and run one measurement on a CUDA environment with
KVServe, fast-hadamard-transform and nvCOMP installed:

```bash
python scripts/bench_compression_stages.py --make-input k12.pt \
  --axis channel --levels 12
python scripts/bench_compression_stages.py --input k12.pt --axis channel \
  --levels 12 --device cuda:1 --warmup 20 --iterations 100 --output k12.json
# For the token case, generate v8.pt with --axis token --levels 8,
# then use the same axis and levels when measuring it.
```

## Current CUDA quantization kernels

Separately, the current `kvs_compress.cu` quantization kernels were measured
using the earlier standalone kernel harness and prebuilt extensions, on the
same H100 GPU. This used PyTorch 2.13.0+cu130, not the environment in the table
above. Candidate source SHA-256 was
`069ba311c769bad086bd71cf4f27518ec95589a7e7ee54be3cfda49c08278e71`, matching
the current `v1` file; the loaded prebuilt candidate binary SHA-256 was
`814d3dc8492750b595b8bde7ec15bfdcda87310df56eedc3762e6365e7333c65`.
The binary was not rebuilt during this measurement.

Each CUDA graph replay contained 200 kernel calls; eight alternating
old/current repetitions used boundary GPU events. The numbers below are the
median event interval divided by 200. Effective GB/s uses 1,048,576 BF16 input
bytes divided by that time and does not include metadata or represent memory
bandwidth. The old/current outputs passed 20 byte-for-byte comparison cases.

| Current kernel | Tokens × heads × head dimension | Levels | Time (µs) | Input GB/s |
| --- | --- | ---: | ---: | ---: |
| K/channel quantize | 1,024 × 4 × 128 | 12 | 6.257 | 167.58 |
| V/token quantize | 1,024 × 4 × 128 | 8 | 3.568 | 293.87 |

These kernel measurements use a different timing boundary and software
environment from the legacy interface measurements and must not be compared
as if they were the same operation. The complete samples and other shapes are
in [`h100-quant-kernels-20261029.json`](h100-quant-kernels-20261029.json).
The standalone kernel harness was a local validation tool, not a repository
benchmark script. Neither measurement establishes AMD performance.
