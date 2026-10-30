# Pluggable operators: GPU validation, 2026-10-30

Add **signed DCT-II, Haar random orthogonal rotation, and centered Lloyd–Max quantization** as optional modular operators. Keep the existing default profile and fixed-workspace runtime unchanged. These are three algorithm choices sharing two implementation modules.

## Quality and representation

The calibration-free Gaussian codebook uses no model score CSV or trained codebook. Eight Llama questions screened fixed scale/refinement choices. The selected profile (`key_bits=value_bits=3`, K group 128, FP32 means, BF16 scales, `scale_alpha=1.1`, `scale_refinements=0`) was then frozen. Held-out evaluation used 16 different Llama questions and 8 questions on a 27B Qwen3.5-architecture checkpoint (local directory `Qwen3.8-27B`), each with seeds 101/202/303. Three seeds are repeated profiles on those questions, not three times as many unique samples.

Both used LongBench HotpotQA, greedy decoding, concurrency 1 and at most 32 output tokens. Complete prompts were retained under caps of 30,000/20,000 tokens respectively; none of the selected prompts were truncated. Llama ran on one RTX 5090; Qwen used two RTX 5090s through Hugging Face device-map sharding, not vLLM tensor parallelism. PyTorch was 2.13.0+cu132 and Transformers 5.17.0.

The harness roundtrips every prefill FullAttention K/V layer before the final prompt token. New generated KV remains native. Qwen has 16 FullAttention and 48 linear-attention layers; recurrent states remain uncompressed and are cloned independently per evaluation branch. The template cache length is checked after every branch.

| Profile | Llama F1 | Qwen F1 | Llama payload ratio | Qwen FullAttention payload ratio |
|---|---:|---:|---:|---:|
| Uncompressed | 58.54 | 69.64 | 1.00× | 1.00× |
| Hadamard + uniform 3-bit | 55.42 | 69.64 | 4.92× | 5.02× |
| DCT + uniform 3-bit | 56.80 | 69.64 | 4.92× | 5.02× |
| Haar + uniform 3-bit | 57.58 | 69.64 | 4.92× | 5.02× |
| Hadamard + centered codebook | 59.00 | 69.64 | 4.74× | 4.87× |
| DCT + centered codebook | 56.91 | 69.64 | 4.74× | 4.87× |
| Haar + centered codebook | 55.90 | 69.64 | 4.74× | 4.87× |

F1 is the mean across questions and seeds. Uniform is a controlled per-channel K / per-token V min-max 3-bit reference with 128-token K groups, not the deployed hybrid/ANS default profile. Its Hadamard baseline uses normalized Humming FWHT and the same CPU-generated per-head sign convention as DCT. All profiles use the same BitPack codec. Ratios include actual packed bytes, the 16-byte codec header and GPU means/scales; CPU metadata is excluded. Hybrid ratios cover FullAttention KV only, not total recurrent state.

On Llama, FP32 means cost about 3.86% more bytes than the uniform reference; on Qwen the cost is about 2.95%. BF16 means are supported when a lower metadata budget is preferred. The wider no-refinement profile improved the Llama mean F1 over uniform Hadamard here; it does not establish a universal accuracy improvement. DCT/Haar combined with the centered quantizer do not always beat the individual alternatives.

Llama uniform Hadamard/DCT/Haar mean first-logit KL was 0.03978/0.02541/0.02726. Qwen mean KL was 0.00293/0.00270/0.00398; DCT+centered was 0.00169. Qwen F1 was identical across these small runs, so KL differences alone are not a task-accuracy win. Llama F1 varied with seed; the machine-readable summary retains each value. Larger task evaluations remain appropriate before changing any production default.

## Kernel and full-roundtrip cost

GPU times below are warmed CUDA Graph intervals: 5 operations per capture, 20 replays per timing sample, 7 samples per run, 3 runs. Setup/CPU basis generation, H2D constant initialization and compilation are excluded. Synchronized Python wall times were also recorded locally; they include launch/allocation overhead and are not equivalent to the graph intervals. No inference, TTFT or RDMA speedup is claimed.

On actual Qwen KV slices (first/middle/last FullAttention layer; `[2,1,8192,4,256]` BF16), median forward DCT/Haar took about 117.3/111.8 μs, versus 307.4/308.3 μs for FP32 dense-matrix references with identical bases. DCT/Haar use a shared per-layer matrix across heads, represented by two BF16 parts with FP32 accumulation. This is still a dense O(D²) transform, not an FFT DCT.

For the same real slices, the previous centered codebook prototype took 34.2 μs to quantize. The metadata-precision-matched new path took 20.9 μs (about 1.64× faster), and the selected FP32-mean static profile took 20.3 μs. The neutral constructor default with one scale refinement took about 41.6 μs: refinement has a real cost. The old reference receives a prearranged head-major input; its layout-copy cost is excluded, so this is a stage comparison.

Tuning compared dynamic binary lookup, linear scans, and a statically expanded selection tree. Constant threshold addresses avoided dependent GPU gathers. K-channel tiling and a smaller V token tile increased parallelism. The chosen tree passed the same symbol/reconstruction reference checks.

| Synthetic BF16 geometry (T/H/D) | Hadamard + centered + BitPack | DCT + centered + BitPack | Haar + centered + BitPack |
|---|---:|---:|---:|
| 1024/8/128 | 38.16 μs | 35.23 μs | 34.29 μs |
| 1024/2/256 | 20.85 μs | 28.75 μs | 28.72 μs |
| 8192/8/128 | 247.61 μs | 204.66 μs | 200.49 μs |
| 8192/2/256 | 87.77 μs | 149.03 μs | 143.91 μs |

Roundtrips include transform, quantization, BitPack encode/decode, dequantization and inverse, using the same centered static profile in all three columns. At D=128 the new rotations reduced this graph interval; at D=256 they increased it. Real Qwen slices similarly took about 188/297/292 μs for Hadamard/DCT/Haar roundtrips. Keep Hadamard for latency-sensitive D=256 profiles unless another rotation justifies the extra cost through quality. No universal latency replacement is proposed.

Profiler traces for warmed DCT/Haar + centered + BitPack contained **zero H2D/DtoH data copies**. Fixture capture and evaluation scoring were outside that region. Small CPU configuration/metadata handling and CUDA launch overhead remain.

## Integration and checks

- `transformer/orthogonal.py`: one tiled kernel for both new rotations; bounded immutable GPU constant caches and stream ownership.
- `quantizer/centered_codebook.py`: 1–4 bit codebooks, packet-owned mean/scale/config metadata, optional scale refinements, independent output buffer support.
- Existing transformer dispatch and adapter factory select the new components. Existing wire packing serializes nested auxiliary tensors, including chunked layer IDs. Conflicting TileLang-fused selections fail explicitly.
- One new test file; **98 tests passed** together with existing BitPack coverage. Cases cover FP32/FP16/BF16, non-power-of-two dimensions, all bit widths, partial groups, strided tensors, output alias rejection, independent packets, streams/CUDA Graphs, separate managers/adapters, and an actual >500 MiB chunked wire roundtrip.
- A built wheel was installed to an isolated prefix and roundtripped on GPU while imports of vLLM, Humming, FHT, pandas, SciPy and TileLang were blocked. Only the new operators/BitPack are covered by that minimal install; the legacy connector retains its own dependencies.

The three components enter the modular library only. The vLLM Mooncake PR and its pinned fixed-workspace provider are unchanged. Supporting these algorithms in that runtime requires representation/workspace planning and separate connector validation. Mixed-outlier codebooks, dual-axis normalization and QJL remain experimental; none is added merely to increase the component count.

The [operator guide](../pluggable-operators.md) documents configuration and compatibility. The [measurement summary](pluggable-operators-20261030.json) records selected sample IDs, per-seed quality, payload ratios, timings and source/result SHA-256 values. Raw outputs and evaluation scripts are retained with the local experiment artifacts; these are small-sample operator evaluations, not a replacement for a production serving benchmark.

## H100 portability check

The same operator sources at `6e47505` also passed **98 GPU regressions** on an
H100 PCIe (PyTorch 2.13.0+cu130, Transformers 5.19.0), including the stream,
CUDA Graph, output-alias and chunked-wire cases. No kernel changes or profile
retuning were needed. The minimal-dependency GPU roundtrip passed too.

A separate Qwen3-32B smoke used four complete HotpotQA questions (indices 24–27),
seeds 101/202/303, concurrency 1 and at most 32 greedy output tokens. All six
compressed profiles and the uncompressed reference scored **75.00 F1**. Uniform
3-bit profiles achieved **4.92×** payload compression; centered-codebook profiles
achieved **4.74×**, including GPU auxiliary tensors. These four questions do not
establish broad quality equivalence. Mean first-logit KL for uniform
Hadamard/DCT/Haar was 0.03526/0.02146/0.02891; centered-codebook combinations were
0.04754/0.04178/0.08600. In particular, equal F1 does not mean that Haar+centered
has equal distortion. Keep model-specific profile validation.

On actual Qwen3-32B KV slices (`[2,1,8192,8,128]`, first/middle/last layer),
median DCT/Haar forward times were **82.35/77.31 μs**, versus **316.47/313.60 μs**
for the FP32 matrix references. Full centered+BitPack roundtrips took
**415.75/303.33/292.06 μs** with Hadamard/DCT/Haar. These are GPU graph intervals,
not inference latency or transfer speedups.

The quantizer optimization did **not** reproduce the RTX 5090's stage gain at
this geometry: the metadata-matched path took **56.59 μs** versus **56.24 μs** for
the old prototype, effectively unchanged. The selected FP32-mean static profile
took **53.96 μs**, with a different metadata budget. For synthetic H=2/D=256,
that static profile took 29.65 μs versus 52.62 μs for the prototype. Speedup is
hardware- and geometry-dependent. Synthetic D=256 full roundtrips still favored
Hadamard (169.17 μs) over DCT/Haar (203.62/199.92 μs).

Warmed traces again showed zero H2D/DtoH data copies. The
[H100 measurement summary](h100-pluggable-operators-20261030.json) retains
per-seed quality, timings and source hashes. This check does not change the
optional status of the new components or the existing default profile.
