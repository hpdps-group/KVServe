# Pluggable GPU compression operators

The modular operator library includes signed DCT-II and Haar random orthogonal
rotations, plus a centered nonuniform scalar quantizer. They are opt-in choices;
existing Hadamard and hybrid uniform profiles remain unchanged.

```bash
uv pip install '.[operators]'
```

This extra installs PyTorch and Triton. The new operators and BitPack need no
Humming, `fast_hadamard_transform`, model score CSV, or calibration dataset.
The existing connector/adapter still needs its usual compatible vLLM environment.

## Transform choices

```python
from kvserve_v1.compression.transformer import KVServeTransformer

transform = KVServeTransformer(transform_type="dct", seed=13107)
# Alternatively: transform_type="haar".
y = transform.transform(layer_id, kv)
restored = transform.inverse(layer_id, y)
```

`kv` is a CUDA BF16/FP16/FP32 tensor with shape
`[2, blocks, tokens, kv_heads, head_dim]`. Both operators preserve this logical
layout and dtype, accept noncontiguous views, and support head dimensions 1–512.

- **DCT** applies per-head Rademacher signs and an orthonormal DCT-II basis.
- **Haar** applies a seeded, Gaussian-QR random orthogonal matrix. This is a
  Haar-distributed rotation, not a Haar wavelet. A layer shares its matrix across
  heads; different layers use different seeded matrices.

Both use the same tiled Triton tensor-core kernel. Matrices are represented by
two BF16 parts with FP32 accumulation; FP16/FP32 inputs are split as well.
There is no per-head Python loop in the warmed transform. This remains an
O(head_dim²) matrix method, rather than an FFT-based fast DCT, and may cost more
than Hadamard. Choose a rotation on the quality/latency tradeoff for the model.

The immutable basis is generated once on CPU and copied to GPU during geometry
initialization. Compression/decompression of KV data runs on GPU. Warm forward
and inverse for every layer/geometry before timing or CUDA Graph capture, and
complete that initialization before capturing on another stream. Input producer
streams must be fenced by the caller, as with other CUDA tensor operations.

Both endpoints must use the same transform, seed, layer IDs and local head order.
Keep the initialization environment consistent across peers; a seed is not an
interchange format for arbitrary QR implementations. Bases need no per-request
payload. These operators do not implement a heterogeneous-TP remapping protocol.

## Centered codebook quantization

```python
from kvserve_v1.compression.quantizer import CenteredCodebookQuantizer

quantizer = CenteredCodebookQuantizer(
    key_bits=3, value_bits=3, group_size=128,
    scale_refinements=1, scale_alpha=1.0, mean_dtype="float32",
)
symbols, metadata = quantizer.quantize(layer_id, y)
y_restored = quantizer.dequantize(layer_id, symbols, metadata)
```

The quantizer computes local means and scales and selects from a shared Gaussian
Lloyd–Max codebook. Codebooks need no model calibration. K statistics are per
head/channel over bounded token groups; V statistics are per head/token over the
head dimension. An optional least-squares scale update reduces reconstruction
error for the selected symbols. It is an option, not a guarantee of better task
accuracy: evaluate the chosen configuration on held-out model inputs.

| Setting | Supported values |
|---|---|
| `key_bits`, `value_bits` | Integers 1–4 |
| `group_size` | 16, 32, 64, 128, 256 |
| `scale_refinements` | 0, 1, 2 |
| `scale_alpha` | Finite positive multiplier for initial RMS |
| `mean_dtype` | `float32` or `bfloat16` |

Means/scales and the complete decoding configuration belong to each returned
packet (`centered_codebook_v1`). A later encode or configuration update does not
overwrite another request's statistics. Scales are BF16. FP32 means use six
auxiliary bytes per statistics item; BF16 means use four. At 128-token K groups
and head dimension 128, symmetric 3-bit coding approaches 3.375 bits/value with
FP32 means, or 3.25 with BF16 means, before small headers and partial-group
overhead. Include these auxiliary bytes when reporting compression ratios.

An independent contiguous CUDA uint8 `out=` buffer can hold the symbols. The
output alphabet is `[0, 2**bits)` for each K/V part. With BitPack, set
`bitpack_bits >= max(key_bits, value_bits)`. A single fixed-width stream does not
automatically save bytes from unequal K/V bit widths. Invalid decoded symbols
trigger a CUDA assertion; malformed packets must not be treated as trusted data.

## Composition and transport

Select the quantizer through `quantizer_config.impl="centered_codebook"` in the
existing `KVCompressionAdapter`. Select either new transform independently in
`transformer_config.transform_type`; the existing pipeline remains
`transformer → quantizer → codec`. Nested means/scales are packed through the
existing GPU auxiliary-buffer and wire metadata paths. No new wire protocol or
transfer engine is introduced.

```python
config = {
    "pipeline": ["transformer", "quantizer", "codec"],
    "transformer_config": {"transform_type": "dct", "seed": 13107},
    "quantizer_config": {
        "impl": "centered_codebook", "key_bits": 3, "value_bits": 3,
        "group_size": 128, "scale_refinements": 0,
        "scale_alpha": 1.1, "mean_dtype": "float32",
    },
    "codec_config": {"codec_type": "bitpack", "bitpack_bits": 3},
}
```

This static example uses a wider initial scale and skips refinement; it differs
from the quantizer's neutral constructor defaults. Matching DCT/Haar examples
are in `configs/compression/*_centered_bitpack.json`. They are starting points
for model-specific evaluation, not universal accuracy presets.

The H100/Qwen3-32B smoke retained the same F1 on four questions, while some
centered-codebook combinations increased first-logit KL. Equal task scores on
small samples do not establish equal distortion; check the selected composition
on the target model before using it as a production profile. Kernel gains also
depend on GPU and head geometry; see the benchmark report for both RTX 5090 and
H100 measurements.

Use identical transform/codec profiles at both endpoints. `tilelang_fused`
includes the old Hadamard/uniform algorithm and cannot select these operators;
conflicting fused/centered selections raise an error. These additions target
the modular operator library. They do not change the separately versioned,
fixed-workspace `compression.runtime` interface or the vLLM Mooncake PR's pinned
provider. Runtime workspace planning and connector-specific validation are
separate integration work.

Numerical, packet ownership, stream, graph, adapter and chunked wire regression
tests are in `tests/test_pluggable_operators.py`. Model and timing evidence,
including limitations, is recorded in the accompanying benchmark report.
